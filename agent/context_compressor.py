"""内置上下文压缩引擎（对应原版 agent/context_compressor.py）。

把超阈值对话的中间轮次用 LLM 摘要替换（head/tail 保护），失败时降级为
确定性静态摘要或纯修剪。my-hermes 精简版保留核心算法，裁剪了：
telemetry、micro-compact/defrag、滚动摘要、[SKILL_PRUNED] 技能标记、
失败冷却的 DB 持久化（只留内存版）等外围。

方案 B：摘要模型支持独立配置（summary_model_override / summary_base_url /
summary_api_key），通过注入的 ``summary_client_factory`` 创建独立客户端；
未配置时回退主 agent 的 worker client。
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    get_default_context_length,
)

logger = logging.getLogger(__name__)

# ── 常量（对齐原版）──────────────────────────────────────────────────────

COMPRESSED_SUMMARY_METADATA_KEY = "_compressed_summary"
COMPRESSED_SUMMARY_HAS_USER_TURN_KEY = "_compressed_summary_has_user_turn"
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

_MIN_SUMMARY_TOKENS = 2000
_SUMMARY_RATIO = 0.20
_SUMMARY_TOKENS_CEILING = 10_000
_SUMMARY_INPUT_MAX_CHARS = 160_000
_PRUNE_MIN_CHARS = 200
_FALLBACK_TURN_MAX_CHARS = 700
_MAX_TAIL_MESSAGE_FLOOR = 8
_FEASIBILITY_SKIP_MIDDLE_FRACTION = 0.10

_SUMMARY_END_MARKER = (
    "--- END OF CONTEXT SUMMARY — "
    "respond to the message below, not the summary above ---"
)

# 压缩后注入 system prompt 的说明（让模型知道历史已被压缩）
_COMPRESSION_NOTE = (
    "[Note: Some earlier conversation turns have been compacted into a handoff "
    "summary to preserve context space. The current session state may still "
    "reflect earlier work, so build on that summary and state rather than "
    "re-doing work.]"
)

# ── 简化脱敏（原版依赖 agent/redact.py 的 redact_sensitive_text，my-hermes
#    精简版内置正则版：覆盖常见密钥模式 + URL 凭据）─────────────────────

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:api[_-]?key|apikey|token|secret|password|passwd|access[_-]?token)\s*[:=]\s*['\"]?[^\s'\"&]{6,}", re.IGNORECASE),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
)


def _redact_compaction_text(text: Any) -> str:
    """脱敏跨压缩摘要边界的内容。

    摘要会持久化并在后续每次摘要 prompt 中重注入，因此使用严格模式：
    密钥/令牌/密码/URL userinfo 一律替换为 [REDACTED]。
    """
    if text is None:
        return ""
    redacted = str(text)
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    # URL userinfo：https://user:pass@host → https://[REDACTED]@host
    redacted = re.sub(
        r"([a-z][a-z0-9+.\-]*://)([^/@\s]+)@",
        r"\1[REDACTED]@",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


# ── 通用辅助（对齐原版同名函数）────────────────────────────────────────


def _safe_int(value: Any) -> int | None:
    """把任意值安全转 int，失败返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _estimate_msg_budget_tokens(msg: Dict[str, Any], charge_stale_thinking: bool = True) -> int:
    """粗略估算单条消息的 token 开销（原版 :1011）。

    字符数 / 4 估算正文；assistant 工具调用按参数长度累加；tool 消息按
    tool_call_id + 内容估算。my-hermes 无精确 tokenizer，取保守上界。
    """
    total = 0
    content = msg.get("content")
    if isinstance(content, str):
        total += len(content) // 4 + 1
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                total += len(str(part.get("text", ""))) // 4 + 1
            elif isinstance(part, str):
                total += len(part) // 4 + 1
    role = msg.get("role", "")
    if role == "assistant":
        for tc in msg.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = tc.get("function", {}).get("arguments", "")
            else:
                fn = getattr(tc, "function", None)
                args = getattr(fn, "arguments", "") if fn else ""
            total += len(str(args)) // 4 + 8
    elif role == "tool":
        total += 8  # tool_call_id 开销
    return max(1, total)


def _last_assistant_index(messages: List[Dict[str, Any]]) -> int:
    """返回最后一条 assistant 消息的下标，没有则 -1。"""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return -1


def _content_text_for_contains(content: Any) -> str:
    """把 content 归一为字符串（dict/list 内容取 text 字段）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _append_text_to_content(content: Any, text: str) -> Any:
    """向 content 追加文本（保持 str/list 形状）。"""
    if content is None:
        return text
    if isinstance(content, str):
        return content + text
    if isinstance(content, list):
        content = list(content)
        content.append({"type": "text", "text": text})
        return content
    return str(content) + text


def _extract_tool_call_name_and_args(tool_call: Any) -> tuple[str, str]:
    """提取工具调用的名称与参数字符串。"""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function", {})
        return str(fn.get("name", "?")), str(fn.get("arguments", ""))
    fn = getattr(tool_call, "function", None)
    name = getattr(fn, "name", "?") if fn else "?"
    args = getattr(fn, "arguments", "") if fn else ""
    return str(name), str(args)


def _extract_tool_call_id(tool_call: Any) -> str:
    """提取工具调用 ID。"""
    if isinstance(tool_call, dict):
        return str(tool_call.get("id", ""))
    return str(getattr(tool_call, "id", "") or "")


def _dedupe_append(items: list[str], value: str, *, limit: int) -> None:
    """去重追加（静态 fallback 收集文件路径用）。"""
    if value and value not in items:
        items.append(value)
        if len(items) > limit:
            del items[0]


def _collect_path_mentions(text: str, relevant_files: list[str], *, limit: int = 12) -> None:
    """从文本收集形如 path 的提及。"""
    for match in re.finditer(r"(?:^|[\s\"'`(])([./~]?[\w./\-]+\.(?:py|js|ts|go|rs|java|c|h|cpp|hpp|md|json|yaml|yml|toml|txt|sh|conf|cfg|ini))", text):
        _dedupe_append(relevant_files, match.group(1), limit=limit)


def resolve_model_threshold(
    model: str,
    model_thresholds: dict[str, float] | None,
    default: float,
) -> float:
    """按模型解析有效压缩阈值（最长子串匹配优先）。"""
    if not model_thresholds or not model:
        return default
    best_key = ""
    for key in model_thresholds:
        if key in model and len(key) > len(best_key):
            best_key = key
    if best_key:
        return float(model_thresholds[best_key])
    return default


def _summarize_tool_result(tool_name: str, tool_args: str, tool_content: str) -> str:
    """把旧工具结果压缩成一行信息性摘要（对齐原版 :1344 的简化版）。

    形如 ``[terminal] ran `npm test` -> exit 0, 47 lines output``。
    """
    content = _content_text_for_contains(tool_content) or ""
    content = content.strip()
    args = tool_args.strip()
    line_count = len(content.splitlines())
    char_count = len(content)
    if line_count <= 1 and char_count <= 200:
        return f"[{tool_name}] {args or 'ran'} -> {content[:120] or 'ok'}"
    return (
        f"[{tool_name}] {args or 'ran'} -> {line_count} lines / "
        f"{char_count} chars"
    )


class ContextCompressor(ContextEngine):
    """内置压缩引擎：摘要替换中间轮次（head/tail 保护）。"""

    _MIN_CTX_TRIGGER_RATIO = 0.85
    _CONTENT_MAX = 6000
    _CONTENT_HEAD = 4000
    _CONTENT_TAIL = 1500
    _TOOL_ARGS_MAX = 1500
    _TOOL_ARGS_HEAD = 1200
    _SUMMARY_INPUT_MAX = _SUMMARY_INPUT_MAX_CHARS

    def __init__(
        self,
        model: str,
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        protect_last_n: int = 20,
        summary_target_ratio: float = 0.20,
        quiet_mode: bool = False,
        summary_model_override: str = None,
        base_url: str = "",
        api_key: str = "",
        config_context_length: int | None = None,
        provider: str = "",
        api_mode: str = "",
        abort_on_summary_failure: bool = False,
        max_tokens: int | None = None,
        model_thresholds: dict[str, float] | None = None,
        threshold_tokens_cap: Any = None,
        proactive_prune_tokens: int = 0,
        proactive_prune_min_result_chars: int = 8000,
        proactive_prune_min_reclaim_tokens: int = 4096,
        min_tail_user_messages: int = 1,
        agent: Any = None,
        summary_client_factory: Any = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.provider = provider
        self.api_mode = api_mode
        self.quiet_mode = quiet_mode
        # 方案 B：独立摘要模型配置（None = 用主模型）
        self.summary_model = summary_model_override or None
        self.abort_on_summary_failure = abort_on_summary_failure
        self.max_tokens = _safe_int(max_tokens)
        self.min_tail_user_messages = min_tail_user_messages
        self._agent = agent
        self._summary_client_factory = summary_client_factory

        self.model_thresholds = model_thresholds or {}
        self._config_threshold_percent = threshold_percent
        self._base_threshold_percent = resolve_model_threshold(
            model, self.model_thresholds, threshold_percent
        )
        self.threshold_percent = self._base_threshold_percent
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n
        self.summary_target_ratio = summary_target_ratio

        # 上下文长度：显式配置 > 硬编码表 > 默认 256K（对齐原版 0/8/9）
        self._config_context_length = _safe_int(config_context_length) or 0
        self.context_length = self._resolve_context_length()

        # 阈值 tokens（含 max_tokens 输出预留；cap 限制）
        self._threshold_tokens_cap = _safe_int(threshold_tokens_cap)
        self.threshold_tokens = self._compute_threshold_tokens(
            self.context_length, self.threshold_percent, self.max_tokens
        )
        self._apply_threshold_tokens_cap()

        # tail 预算：摘要目标比例 × 上下文长度（自动随窗口缩放）
        self.tail_token_budget = int(self.context_length * self.summary_target_ratio)
        self.max_summary_tokens = min(
            max(_MIN_SUMMARY_TOKENS, int(self.tail_token_budget * _SUMMARY_RATIO * 5)),
            _SUMMARY_TOKENS_CEILING,
        )

        # 工具结果预修剪参数（主动修剪触发，my-hermes 暂只在压缩内使用）
        self.proactive_prune_tokens = proactive_prune_tokens or 0
        self.proactive_prune_min_result_chars = proactive_prune_min_result_chars
        self.proactive_prune_min_reclaim_tokens = proactive_prune_min_reclaim_tokens

        # 运行状态
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self._previous_summary: Optional[str] = None
        self._summary_has_user_turn: Optional[bool] = None
        self._summary_failure_cooldown_until = 0.0
        self._last_summary_error: Optional[str] = None
        self._last_summary_auth_failure = False
        self._last_summary_network_failure = False
        self._last_summary_fallback_used = False
        self._last_summary_dropped_count = 0
        self._last_compress_aborted = False
        self._last_compression_savings_pct: Optional[float] = None
        self._ineffective_compression_count = 0
        self._prellm_skip_count = 0
        self._last_feasibility_skip = False
        self._summary_model_fallen_back = False

    # -- 身份 ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "compressor"

    # -- 上下文长度 / 阈值 ------------------------------------------------

    def _resolve_context_length(self) -> int:
        """上下文长度：显式配置 > 硬编码表 > 默认 256K（对齐原版 0/8/9）。"""
        if self._config_context_length:
            return self._config_context_length
        return get_default_context_length(self.model, self.base_url)

    @property
    def context_length(self) -> int:
        return self._context_length

    @context_length.setter
    def context_length(self, value: int) -> None:
        self._context_length = max(1, int(value or 0))

    @property
    def threshold_tokens(self) -> int:
        return self._threshold_tokens

    @threshold_tokens.setter
    def threshold_tokens(self, value: int) -> None:
        self._threshold_tokens = max(1, int(value or 0))

    @property
    def tail_token_budget(self) -> int:
        return self._tail_token_budget

    @tail_token_budget.setter
    def tail_token_budget(self, value: int) -> None:
        self._tail_token_budget = max(1, int(value or 0))

    @property
    def max_summary_tokens(self) -> int:
        return self._max_summary_tokens

    @max_summary_tokens.setter
    def max_summary_tokens(self, value: int) -> None:
        self._max_summary_tokens = max(1, int(value or 0))

    def _apply_threshold_tokens_cap(self) -> None:
        """应用 threshold_tokens_cap（若有）。"""
        if self._threshold_tokens_cap:
            self._threshold_tokens = min(
                self._threshold_tokens, self._threshold_tokens_cap
            )

    def _effective_threshold_percent(self, context_length: int, threshold_percent: float) -> float:
        """小上下文模型不提前压缩：阈值下限 64K 场景特殊处理。"""
        return threshold_percent

    def _compute_threshold_tokens(
        self, context_length: int, threshold_percent: float, max_tokens: int | None = None,
    ) -> int:
        """计算压缩触发阈值（对齐原版 :2473）。

        有效输入预算 = context_length - max_tokens（provider 预留输出空间）；
        阈值 = 预算 × 比例，下限 64K；当下限吞掉整个窗口时退化为窗口的
        85%（保证小上下文模型也能触发压缩）。
        """
        effective_window = context_length - (max_tokens or 0)
        if effective_window <= 0:
            effective_window = context_length
        pct_value = int(effective_window * threshold_percent)
        floored = max(pct_value, MINIMUM_CONTEXT_LENGTH)
        if effective_window > 0 and floored >= effective_window:
            return max(
                1,
                min(
                    int(effective_window * self._MIN_CTX_TRIGGER_RATIO),
                    effective_window - 1,
                ),
            )
        return floored

    # -- 用量追踪 --------------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """根据 API 响应更新追踪的 token 用量（简化自原版 :2741）。"""
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get(
            "total_tokens", self.last_prompt_tokens + self.last_completion_tokens
        )

    # -- 压缩判定 --------------------------------------------------------

    def should_compress(self, prompt_tokens: int = None) -> bool:
        decision, _reason = self.should_compress_info(prompt_tokens)
        return decision

    def should_compress_info(
        self, prompt_tokens: int = None
    ) -> "tuple[bool, str | None]":
        """返回 ``(should_compress, reason)``。

        reason 非 None 表示"需要压缩但被阻断"：``cooldown:<秒>`` 摘要 LLM
        冷却中；``ineffective`` 连续压缩无效（防抖动）。
        """
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens < self.threshold_tokens:
            return False, None
        if self._automatic_compression_blocked():
            return False, self._compression_block_reason() or "blocked"
        return True, None

    def _compression_block_reason(self) -> "str | None":
        cooldown = self._summary_failure_cooldown_until - time.monotonic()
        if cooldown > 0:
            return f"cooldown:{cooldown:.0f}"
        if self._ineffective_compression_count >= 2:
            return "ineffective"
        return None

    def _automatic_compression_blocked(self) -> bool:
        """摘要 LLM 冷却或连续无效时阻断自动压缩。"""
        if self._summary_failure_cooldown_until > time.monotonic():
            return True
        if self._ineffective_compression_count >= 2:
            return True
        return False

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """中间区域非空才可压缩（对齐原版 :5655）。"""
        compress_start = self._align_boundary_forward(
            messages, self._protect_head_size(messages)
        )
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        return compress_start < compress_end

    # -- 工具结果修剪 ----------------------------------------------------

    def _prune_old_tool_results(
        self, messages: List[Dict[str, Any]], protect_tail_count: int,
        protect_tail_tokens: int | None = None,
        min_prune_chars: int = _PRUNE_MIN_CHARS,
    ) -> tuple[List[Dict[str, Any]], int]:
        """把旧工具结果替换为一行信息性摘要（对齐原版 :3099 的简化版）。

        从尾部反向走，保护最近的 protect_tail_count 条（或预算内消息）；
        保护区外的大工具结果替换为 ``[tool] args -> N lines / N chars``。
        """
        if not messages:
            return messages, 0
        result = [m.copy() for m in messages]
        pruned = 0

        # tool_call_id -> (tool_name, args)
        call_id_to_tool: Dict[str, tuple] = {}
        for msg in result:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    call_id_to_tool[_extract_tool_call_id(tc)] = (
                        _extract_tool_call_name_and_args(tc)
                    )

        if protect_tail_tokens is not None and protect_tail_tokens > 0:
            accumulated = 0
            boundary = len(result)
            min_protect = min(protect_tail_count, len(result), _MAX_TAIL_MESSAGE_FLOOR)
            for i in range(len(result) - 1, -1, -1):
                msg_tokens = _estimate_msg_budget_tokens(result[i])
                if accumulated + msg_tokens > protect_tail_tokens and (len(result) - i) >= min_protect:
                    boundary = i
                    break
                accumulated += msg_tokens
                boundary = i
            budget_protect_count = len(result) - boundary
            protected_count = max(budget_protect_count, min_protect)
            prune_boundary = len(result) - protected_count
        else:
            prune_boundary = len(result) - protect_tail_count

        for i in range(min(prune_boundary, len(result))):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = _content_text_for_contains(msg.get("content"))
            if len(content) < min_prune_chars:
                continue
            tool_name, args = call_id_to_tool.get(
                msg.get("tool_call_id", ""), ("unknown", "")
            )
            msg["content"] = _summarize_tool_result(tool_name, args, content)
            pruned += 1
        return result, pruned

    def prune_tool_results_only(
        self, messages: List[Dict[str, Any]], current_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """只剪旧工具结果（对齐原版 :3391）。"""
        return self._prune_old_tool_results(
            messages,
            protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )

    # -- 摘要生成 --------------------------------------------------------

    def _compute_summary_budget(self, turns_to_summarize: List[Dict[str, Any]]) -> int:
        """摘要预算随内容缩放：内容×20%，下限 2000，上限 10K。"""
        content_tokens = sum(
            _estimate_msg_budget_tokens(m) for m in turns_to_summarize
        )
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    def _serialize_for_summary(self, turns: List[Dict[str, Any]]) -> str:
        """把轮次序列化为给摘要模型的有标签文本（对齐原版 :3532 简化版）。

        全部内容先脱敏；正文按 _CONTENT_MAX 截断；assistant 带工具调用名
        与参数；tool 结果带 tool_call_id。
        """
        parts: list[str] = []
        for msg in turns:
            role = msg.get("role", "unknown")
            content = _content_text_for_contains(msg.get("content"))
            content = _redact_compaction_text(content)
            # 去掉内联推理块（<think> / <reasoning>），避免临时草稿进摘要
            content = re.sub(
                r"<(?:think|reasoning|Thought)[^>]*>.*?</(?:think|reasoning|Thought)>",
                "",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

            if role == "tool":
                tool_id = msg.get("tool_call_id", "")
                if len(content) > self._CONTENT_MAX:
                    content = (
                        content[: self._CONTENT_HEAD]
                        + "\n...[truncated]...\n"
                        + content[-self._CONTENT_TAIL:]
                    )
                parts.append(f"[TOOL RESULT {tool_id}]: {content}")
                continue

            if role == "assistant":
                if len(content) > self._CONTENT_MAX:
                    content = (
                        content[: self._CONTENT_HEAD]
                        + "\n...[truncated]...\n"
                        + content[-self._CONTENT_TAIL:]
                    )
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    tc_parts = []
                    for tc in tool_calls:
                        name, args = _extract_tool_call_name_and_args(tc)
                        args = _redact_compaction_text(args)
                        if len(args) > self._TOOL_ARGS_MAX:
                            args = args[: self._TOOL_ARGS_HEAD] + "..."
                        tc_parts.append(f"  {name}({args})")
                    content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
                parts.append(f"[ASSISTANT]: {content}")
                continue

            if len(content) > self._CONTENT_MAX:
                content = (
                    content[: self._CONTENT_HEAD]
                    + "\n...[truncated]...\n"
                    + content[-self._CONTENT_TAIL:]
                )
            parts.append(f"[{role.upper()}]: {content}")
        return "\n\n".join(parts)

    def _bound_summary_input(self, content: str) -> str:
        """限制摘要输入总长，保留首尾并标记省略中间（对齐原版 :3823）。"""
        if len(content) <= self._SUMMARY_INPUT_MAX:
            return content
        marker_template = (
            "\n\n...[summary input truncated: omitted "
            "{omitted:,} chars from the middle to keep compression prompt bounded]...\n\n"
        )
        marker = marker_template.format(omitted=len(content))
        remaining = max(self._SUMMARY_INPUT_MAX - len(marker), 0)
        head_chars = int(remaining * 0.45)
        tail_chars = remaining - head_chars
        omitted = max(len(content) - head_chars - tail_chars, 0)
        marker = marker_template.format(omitted=omitted)
        remaining = max(self._SUMMARY_INPUT_MAX - len(marker), 0)
        head_chars = int(remaining * 0.45)
        tail_chars = remaining - head_chars
        tail = content[-tail_chars:].lstrip() if tail_chars else ""
        return content[:head_chars].rstrip() + marker + tail

    def _call_summary_llm(self, messages: List[Dict[str, Any]], *, model: str) -> Any:
        """调用摘要 LLM（方案 B）。

        - 配置了 summary_client_factory（独立摘要模型）→ 用它；
        - 否则回退主 agent 的 worker client（同 key/endpoint）；
        - 再否则用 create_openai_client 自建（按 self.base_url/api_key）。
        """
        if self._summary_client_factory is not None:
            client = self._summary_client_factory()
        elif self._agent is not None and hasattr(self._agent, "_make_worker_client"):
            client = self._agent._make_worker_client()
        else:
            from agent.agent_runtime_helpers import create_openai_client

            client = create_openai_client(
                self._agent,
                {
                    "api_key": self.api_key or "",
                    "base_url": self.base_url or "",
                },
                reason="summary",
                shared=False,
            )
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _generate_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        focus_topic: Optional[str] = None,
        memory_context: str = "",
    ) -> Optional[str]:
        """生成结构化摘要（对齐原版 :3886 的裁剪版）。

        - 序列化中间轮次（脱敏/截断/去 think 块）；
        - 结构化模板 prompt（Goal/Constraints/Completed Actions/...）；
        - 有 previous summary 时走迭代更新路径；
        - 摘要模型失败（且配置了独立模型）→ 回退主模型一次；
        - 失败进入冷却，避免摘要 LLM 429 时死循环。
        """
        if time.monotonic() < self._summary_failure_cooldown_until:
            logger.debug(
                "Skipping context summary during cooldown (%.0fs remaining)",
                self._summary_failure_cooldown_until - time.monotonic(),
            )
            return None
        if focus_topic:
            focus_topic = _redact_compaction_text(focus_topic)
        if self._previous_summary:
            self._previous_summary = _redact_compaction_text(self._previous_summary)

        summary_budget = self._compute_summary_budget(turns_to_summarize)
        content_to_summarize = self._serialize_for_summary(turns_to_summarize)
        content_to_summarize = self._bound_summary_input(content_to_summarize)

        _preamble = (
            "You are a summarization agent creating a context checkpoint. "
            "Treat the conversation turns below as source material for a "
            "compact record of prior work. "
            "Produce only the structured summary; do not add a greeting, "
            "preamble, or prefix. "
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary — replace any that appear "
            "with [REDACTED]. Note that credentials were present, but do not "
            "preserve their values."
        )
        _template = """## Goal
[The user's original goal or current objective]

## Constraints & Preferences
[Runtime, configuration, and technical constraints only. Do not invent user preferences.]

## Completed Actions
[Numbered list of concrete actions taken — include tool used, target, and outcome.
Format each as: N. ACTION target — outcome [tool: name]
Be specific with file paths, commands, line numbers, and results.]

## Active State
[Current working state — include modified/created files, test status, running processes]

## Blocked
[Any blockers, errors, or issues not yet resolved. Include exact error messages.]

## Key Decisions
[Important technical decisions and WHY they were made]

## Resolved Questions
[None, or answered questions]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Critical Context
[Any specific values, error messages, configuration details, or data that would be lost without explicit preservation. NEVER include API keys, tokens, passwords, or credentials — write [REDACTED] instead.]

Target ~{budget} tokens. Be CONCRETE — include file paths, command outputs, error messages, line numbers, and specific values. Avoid vague descriptions like "made some changes" — say exactly what changed.
Write only the summary body. Do not include any preamble or prefix."""

        if self._previous_summary:
            bounded_previous = self._bound_summary_input(self._previous_summary)
            prompt = f"""{_preamble}

You are updating a context compaction summary. A previous compaction produced the summary below. New conversation turns have occurred since then and need to be incorporated.

PREVIOUS SUMMARY:
{bounded_previous}

NEW TURNS TO INCORPORATE:
{content_to_summarize}

Update the summary using this exact structure. PRESERVE all existing information that is still relevant. ADD new completed actions to the numbered list (continue numbering). Move answered questions to "Resolved Questions". Update "Active State" to reflect current state. Remove information only if it is clearly obsolete.

{_template.format(budget=summary_budget)}"""
        else:
            prompt = f"""{_preamble}

Create a structured checkpoint summary for the conversation after earlier turns are compacted. The summary should preserve enough detail for continuity without re-reading the original turns.

TURNS TO SUMMARIZE:
{content_to_summarize}

Use this exact structure:

{_template.format(budget=summary_budget)}"""

        if focus_topic:
            prompt += (
                f'\n\nFOCUS TOPIC: "{focus_topic}"\n'
                "This compaction should PRIORITISE preserving all information "
                "related to the focus topic above. For content NOT related to "
                "the focus topic, summarise more aggressively. Even for the "
                "focus topic, NEVER preserve API keys, tokens, passwords, or "
                "credentials — use [REDACTED]."
            )

        messages = [{"role": "user", "content": prompt}]
        attempts = [
            self.summary_model or self.model,  # 独立摘要模型或主模型
        ]
        # 配置了独立摘要模型且尚未回退过 → 主模型作为第二次尝试
        if (
            self.summary_model
            and self.summary_model != self.model
            and not self._summary_model_fallen_back
        ):
            attempts.append(self.model)

        last_error: Optional[Exception] = None
        for model in attempts:
            try:
                response = self._call_summary_llm(messages, model=model)
                message = response.choices[0].message
                content = (
                    message.get("content")
                    if isinstance(message, dict)
                    else getattr(message, "content", message)
                )
                if not isinstance(content, str):
                    content = str(content) if content else ""
                content = content.strip()
                if content:
                    self._summary_model_fallen_back = model != (self.summary_model or self.model)
                    self._last_summary_error = None
                    self._last_summary_auth_failure = False
                    self._last_summary_network_failure = False
                    return content
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Summary LLM call failed (model=%s): %s", model, exc
                )
                if _is_summary_access_or_quota_error(exc):
                    self._last_summary_auth_failure = True
                    break

        # 全部失败：进入冷却 + 记录错误
        self._summary_failure_cooldown_until = time.monotonic() + 30.0
        self._last_summary_error = str(last_error) if last_error else "unknown"
        return None

    def _build_static_fallback_summary(
        self,
        turns_to_summarize: List[Dict[str, Any]],
        reason: str | None = None,
    ) -> str:
        """确定性兜底摘要：本地提取用户问题/工具动作/文件/错误（对齐 :3620 简化版）。"""
        user_asks: list[str] = []
        assistant_actions: list[str] = []
        tool_actions: list[str] = []
        relevant_files: list[str] = []
        blockers: list[str] = []

        def _compact_fallback_turn(value: Any) -> str:
            text = _redact_compaction_text(_content_text_for_contains(value))
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > _FALLBACK_TURN_MAX_CHARS:
                text = text[:_FALLBACK_TURN_MAX_CHARS - 15].rstrip() + " ...[truncated]"
            return text

        for msg in turns_to_summarize:
            role = msg.get("role", "unknown")
            text = _compact_fallback_turn(msg.get("content"))
            _collect_path_mentions(text, relevant_files)
            if role == "user" and text and "echo" not in text[:20].lower():
                _dedupe_append(user_asks, text[:300], limit=8)
            elif role == "assistant" and text:
                _dedupe_append(assistant_actions, text[:300], limit=8)
            elif role == "tool" and text:
                _dedupe_append(tool_actions, text[:300], limit=8)
            if re.search(r"\berror\b|\bexception\b|\bfailed\b", text, re.IGNORECASE):
                _dedupe_append(blockers, text[:300], limit=5)

        lines = ["[CONTEXT SUMMARY]: Earlier turns were compacted automatically."]
        if user_asks:
            lines.append("\nUser asks:")
            lines += [f"- {a}" for a in user_asks]
        if assistant_actions:
            lines.append("\nAssistant actions:")
            lines += [f"- {a}" for a in assistant_actions]
        if tool_actions:
            lines.append("\nTool results:")
            lines += [f"- {t}" for t in tool_actions]
        if relevant_files:
            lines.append("\nRelevant files: " + ", ".join(relevant_files))
        if blockers:
            lines.append("\nBlockers/errors:")
            lines += [f"- {b}" for b in blockers]
        if reason:
            lines.append(f"\n[Compression note: {reason}]")
        return "\n".join(lines)

    def _strip_summary_prefix(self, summary: str) -> str:
        if summary.startswith(LEGACY_SUMMARY_PREFIX):
            return summary[len(LEGACY_SUMMARY_PREFIX):].lstrip()
        return summary

    def _with_summary_prefix(self, summary: str) -> str:
        return f"{LEGACY_SUMMARY_PREFIX} {summary}"

    # -- 边界定位 --------------------------------------------------------

    def _protect_head_size(self, messages: List[Dict[str, Any]]) -> int:
        """head 大小：system（若有）+ protect_first_n 条。"""
        start = 1 if messages and messages[0].get("role") == "system" else 0
        return min(len(messages), start + max(1, self.protect_first_n))

    def _align_boundary_forward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """把边界推进到完整轮次起点（tool 消息归属其 assistant 之前）。"""
        while idx < len(messages) and messages[idx].get("role") == "tool":
            idx += 1
        return idx

    def _align_boundary_backward(self, messages: List[Dict[str, Any]], idx: int) -> int:
        """把边界回退到完整轮次起点。"""
        while idx > 0 and messages[idx].get("role") == "tool":
            idx -= 1
        return idx

    def _find_last_user_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """head_end 之后最后一条 user 消息下标，无则 -1。"""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "user":
                return i
        return -1

    def _find_last_assistant_message_idx(
        self, messages: List[Dict[str, Any]], head_end: int
    ) -> int:
        """head_end 之后最后一条 assistant 消息下标，无则 -1。"""
        for i in range(len(messages) - 1, head_end - 1, -1):
            if messages[i].get("role") == "assistant":
                return i
        return -1

    def _ensure_last_user_message_in_tail(
        self, messages: List[Dict[str, Any]], cut_idx: int, head_end: int
    ) -> int:
        """确保最近一条 user 消息在 tail 中（调整 cut_idx）。"""
        idx = self._find_last_user_message_idx(messages, head_end)
        if idx >= 0 and idx >= cut_idx:
            return cut_idx
        if idx >= 0:
            return idx
        return cut_idx

    def _find_turn_pair_end(
        self, messages: List[Dict[str, Any]], user_idx: int
    ) -> int:
        """从 user 消息找轮次结束（到下一个 user 前，含后续 assistant/tool）。"""
        for i in range(user_idx + 1, len(messages)):
            if messages[i].get("role") == "user":
                return i
        return len(messages)

    def _find_tail_cut_by_tokens(
        self, messages: List[Dict[str, Any]], head_end: int,
        token_budget: int | None = None,
    ) -> int:
        """从尾部反向累计 token 直到预算，返回 tail 起点下标（对齐 :5505 简化版）。

        - 预算为第一准则，但保留最近消息数下限（protect_last_n，上限 8）；
        - 预算可超 1.5×，避免在超大消息中间切割；
        - 不切割 tool_call/result 组；确保最近 user 消息在 tail。
        """
        if token_budget is None:
            token_budget = self.tail_token_budget
        n = len(messages)
        available_tail = max(0, n - head_end - 1)
        min_tail_floor = max(3, min(self.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR))
        compressible_tail_cap = max(3, available_tail - 2)
        min_tail = (
            min(min_tail_floor, compressible_tail_cap, available_tail)
            if available_tail > 1 else 0
        )
        soft_ceiling = int(token_budget * 1.5)
        accumulated = 0
        cut_idx = n

        for i in range(n - 1, head_end - 1, -1):
            msg_tokens = _estimate_msg_budget_tokens(messages[i])
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i
            # 不切割 tool 消息（归属其 assistant 轮次）
            if messages[i].get("role") == "tool":
                j = i - 1
                while j >= head_end and messages[j].get("role") not in ("assistant", "user"):
                    j -= 1
                if j >= head_end and messages[j].get("role") == "assistant":
                    cut_idx = j
                    i = j
                    break

        cut_idx = self._ensure_last_user_message_in_tail(messages, cut_idx, head_end)
        return cut_idx

    def _find_context_summaries(
        self, messages: List[Dict[str, Any]], start: int, end: int
    ) -> list[tuple[int, str]]:
        """在 [start, end) 找带压缩 metadata 的摘要消息（简化重水合）。"""
        hits: list[tuple[int, str]] = []
        for i in range(start, min(end, len(messages))):
            msg = messages[i]
            if msg.get(COMPRESSED_SUMMARY_METADATA_KEY):
                content = _content_text_for_contains(msg.get("content"))
                if content:
                    hits.append((i, content))
        return hits

    # -- 孤儿 tool 对清理 ------------------------------------------------

    def _sanitize_tool_pairs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理孤儿 tool_call / tool_result 对（对齐原版 :4985 简化版）。

        删除没有对应 assistant tool_call 的 tool 消息，以及没有对应 tool
        结果的 assistant tool_call（防 API 报 ID 不匹配）。
        """
        result = []
        known_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = _extract_tool_call_id(tc)
                    if cid:
                        known_ids.add(cid)
        for msg in messages:
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid and tid not in known_ids:
                    continue  # 孤儿 tool 结果：删除
            result.append(msg)
        # 删除调用了但结果被删光的 assistant tool_call（保留 name/args 但清空）
        for msg in result:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg = dict(msg)
                msg["tool_calls"] = [
                    tc for tc in msg["tool_calls"]
                    if not _extract_tool_call_id(tc) or _extract_tool_call_id(tc) in known_ids
                ]
        return result

    # -- 主流程 ----------------------------------------------------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """压缩对话消息：head/tail 保护 + 中间摘要替换（对齐原版 :6423 裁剪版）。

        流程：
          1. 预清理：剪旧工具结果（无 LLM）
          2. 边界：head（system + 前 N 条）与 tail（token 预算）
          3. 中间序列化 → LLM 摘要（结构化模板）
          4. 摘要失败 → 静态 fallback（或 abort_on_summary_failure 时中止）
          5. 组装：head + summary + tail（role 交替安全），清理孤儿对
        """
        # 复位每次调用的状态
        self._last_summary_fallback_used = False
        self._last_summary_dropped_count = 0
        self._last_feasibility_skip = False
        self._last_summary_error = None
        self._last_compress_aborted = False

        if not messages:
            return messages
        n_messages = len(messages)
        if n_messages <= max(4, self.protect_first_n + self.protect_last_n):
            self._ineffective_compression_count += 1
            return messages

        # Phase 1：剪旧工具结果（便宜、无 LLM）
        messages, pruned_count = self._prune_old_tool_results(
            messages,
            protect_tail_count=self.protect_last_n,
            protect_tail_tokens=self.tail_token_budget,
        )
        if pruned_count and not self.quiet_mode:
            logger.info("Pre-compression: pruned %d old tool result(s)", pruned_count)

        # Phase 2：边界定位
        compress_start = self._protect_head_size(messages)
        compress_start = self._align_boundary_forward(messages, compress_start)
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)

        if compress_start >= compress_end:
            self._ineffective_compression_count += 1
            if not self.quiet_mode:
                logger.warning(
                    "Compression skipped: compress_start (%d) >= compress_end (%d)",
                    compress_start, compress_end,
                )
            return messages

        turns_to_summarize = messages[compress_start:compress_end]

        # 简化重水合：找已有 summary 作为 previous（迭代更新）
        previous_summary_before = self._previous_summary
        summary_hits = self._find_context_summaries(messages, compress_start, compress_end)
        if summary_hits:
            if not self._previous_summary:
                self._previous_summary = "\n\n".join(body for _, body in summary_hits)

        if not self.quiet_mode:
            logger.info(
                "Context compression triggered: summarizing turns %d-%d "
                "(%d turns), protecting %d head messages",
                compress_start + 1, compress_end, len(turns_to_summarize),
                compress_start,
            )

        # Phase 3：生成摘要（LLM 或静态 fallback）
        if not force and self._ineffective_compression_count >= 1:
            middle_tokens = sum(
                _estimate_msg_budget_tokens(m) for m in turns_to_summarize
            )
            if middle_tokens < int(self.threshold_tokens * _FEASIBILITY_SKIP_MIDDLE_FRACTION):
                self._last_feasibility_skip = True
                self._prellm_skip_count += 1
                summary = None
            else:
                summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)
        else:
            summary = self._generate_summary(turns_to_summarize, focus_topic=focus_topic)

        if not summary and not self._last_feasibility_skip and self.abort_on_summary_failure:
            self._last_compress_aborted = True
            self._previous_summary = previous_summary_before
            if not self.quiet_mode:
                logger.warning(
                    "Summary generation failed — aborting compression "
                    "(compression.abort_on_summary_failure=true)."
                )
            return messages

        # Phase 4：组装压缩后的消息列表
        if not summary:
            n_dropped = compress_end - compress_start
            self._last_summary_dropped_count = n_dropped
            self._last_summary_fallback_used = True
            summary = self._build_static_fallback_summary(
                turns_to_summarize,
                reason=None if self._last_feasibility_skip else self._last_summary_error,
            )
        else:
            summary = self._strip_summary_prefix(summary)

        # head：system prompt 注入压缩说明
        compressed: List[Dict[str, Any]] = []
        for i in range(compress_start):
            msg = dict(messages[i])
            if i == 0 and msg.get("role") == "system":
                existing = msg.get("content")
                if _COMPRESSION_NOTE not in _content_text_for_contains(existing):
                    msg["content"] = _append_text_to_content(
                        existing,
                        "\n\n" + _COMPRESSION_NOTE
                        if isinstance(existing, str) and existing
                        else _COMPRESSION_NOTE,
                    )
            compressed.append(msg)

        tail_messages: List[Dict[str, Any]] = [
            dict(m) for m in messages[compress_end:]
        ]

        # role 交替：summary 与 head/tail 相邻可见角色保持 user/assistant 交替
        last_head_role: Optional[str] = "user"
        if compressed:
            for m in reversed(compressed):
                role = m.get("role")
                if role in ("user", "assistant"):
                    last_head_role = role
                    break
        first_tail_role: Optional[str] = None
        if tail_messages:
            for m in tail_messages:
                role = m.get("role")
                if role in ("user", "assistant"):
                    first_tail_role = role
                    break
        _force_user_leading = (
            compress_start == 0
            or last_head_role == "system"
            or not any(
                m.get("role") == "user" and _content_text_for_contains(m.get("content")).strip()
                for m in compressed + tail_messages
            )
        )
        if (
            last_head_role is None
            or last_head_role in {"assistant", "tool"}
            or _force_user_leading
        ):
            summary_role = "user"
        else:
            summary_role = "assistant"
        _merge_summary_into_tail = False
        if first_tail_role is not None and summary_role == first_tail_role:
            flipped = "assistant" if summary_role == "user" else "user"
            if (
                flipped != last_head_role
                and last_head_role is not None
                and not _force_user_leading
            ):
                summary_role = flipped
            else:
                _merge_summary_into_tail = bool(tail_messages)

        if not _merge_summary_into_tail:
            summary = summary + "\n\n" + _SUMMARY_END_MARKER
            compressed.append({
                "role": summary_role,
                "content": summary,
                COMPRESSED_SUMMARY_METADATA_KEY: True,
                COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: True,
            })
        else:
            # 合并进首条 tail 消息（避免相邻同 role）
            first = dict(tail_messages[0])
            first["content"] = (
                _content_text_for_contains(first.get("content"))
                + "\n\n[PRIOR CONTEXT — COMPACTION SUMMARY BELOW]\n\n"
                + summary
                + "\n\n"
                + _SUMMARY_END_MARKER
            )
            tail_messages[0] = first

        result = compressed + tail_messages
        result = self._sanitize_tool_pairs(result)

        self.compression_count += 1
        self._previous_summary = summary
        self._last_compression_savings_pct = (
            1.0 - len(result) / max(1, n_messages)
            if n_messages else 0.0
        )
        # 有效压缩一次后清零无效计数
        self._ineffective_compression_count = 0
        return result


def _is_summary_access_or_quota_error(exc: Exception) -> bool:
    """判断摘要 LLM 异常是否为终态访问/配额错误（不重试）。"""
    err_text = str(exc).lower()
    markers = (
        "authentication", "401", "402", "403", "invalid api key",
        "insufficient_quota", "quota exceeded", "permission denied",
        "missing credential", "api key not found",
    )
    return any(m in err_text for m in markers)
