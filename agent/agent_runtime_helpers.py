"""运行时辅助（精简移植版）。

client 创建/关闭 + strip_think_blocks（interim 旁白交付用）
+ restore_primary_runtime（每轮开始恢复主 provider，对应原版
agent_runtime_helpers.py:1459 的精简版）。
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import base_url_host_matches


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent

    return run_agent


def create_openai_client(
    agent, client_kwargs: dict, *, reason: str, shared: bool
) -> Any:

    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify

    client_kwargs = dict(client_kwargs)
    # 验证网络
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)

    httpx_verify = resolve_httpx_verify(
        ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg
    )
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))

    # TODO 判断provider 是不是copilot或者gemini,创建专属client

    # 注入 TCP 保活机制，防止 Agent 永久挂起
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(
            client_kwargs.get("base_url", ""),
            verify=httpx_verify,
        )
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http

    # 禁用 SDK 默认重试，通过 client_kwargs.setdefault("max_retries", 0) 强制关闭 OpenAI SDK 的内置重试。
    client_kwargs.setdefault("max_retries", 0)

    # TODO 自动补全 GitHub Copilot 必需的请求头，防止路由错误 (Header Injection for Copilot)

    client = _ra().OpenAI(**client_kwargs)

    _ra().logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client


def restore_primary_runtime(agent) -> bool:
    """回合开始时恢复主 provider（对应原版 agent_runtime_helpers.py:1459 精简版）。

    长会话里同一个 AIAgent 跨多轮复用。若上一轮激活了 fallback，不恢复的话
    一次瞬时故障会把整个会话钉在回退 provider 上。每轮开始调用本函数让
    fallback 成为"回合内"行为：

    - 未激活 fallback：只重置回退链游标（防 #20465：链耗尽但未激活时
      游标卡在链尾，阻塞后续所有回退尝试），返回 False；
    - 激活过但主 provider 仍在限流冷却（_rate_limited_until 未到）：保持
      回退，返回 False；
    - 否则从 _primary_runtime 快照恢复主 provider 的 model/provider/
      base_url/api_key/_client_kwargs，并用 create_openai_client 重建共享
      client，重置回退状态，返回 True。

    精简版不移植原版的 credential_pool / 压缩引擎 / 传输缓存恢复——
    my-hermes 无这些组件。
    """
    if not getattr(agent, "_fallback_activated", False):
        agent._fallback_index = 0
        return False

    if getattr(agent, "_rate_limited_until", 0) > time.monotonic():
        return False  # 主 provider 仍在冷却，保持回退

    rt = getattr(agent, "_primary_runtime", None)
    if not rt:
        # 快照缺失（异常路径）：清状态，退回主 provider 属性不动
        agent._fallback_activated = False
        agent._fallback_index = 0
        return False

    try:
        agent.model = rt["model"]
        agent.provider = rt["provider"]
        agent.base_url = rt["base_url"]
        # 恢复主 provider 的 API 协议（快照由 try_activate_fallback 保存；
        # 缺省兜底 chat_completions，兼容未含该字段的旧快照）
        agent.api_mode = rt.get("api_mode") or getattr(
            agent, "api_mode", "chat_completions"
        )
        agent.api_key = rt["api_key"]
        agent._client_kwargs = dict(rt["client_kwargs"])
        agent.client = create_openai_client(
            agent, dict(rt["client_kwargs"]), reason="restore_primary", shared=True
        )
    except Exception as e:  # pragma: no cover - 恢复失败保持回退
        _ra().logger.warning("Restore primary runtime failed: %s", e)
        return False

    agent._fallback_activated = False
    agent._fallback_index = 0
    return True


def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:

    from agent.message_sanitization import apply_reasoning_content_policy

    apply_reasoning_content_policy(
        source_msg, api_msg, agent._needs_thinking_reasoning_pad()
    )


# ─── strip_think_blocks（对应原版 agent/agent_runtime_helpers.py:809 精简版）───

_REASONING_TAG_NAMES = (
    "think",
    "thinking",
    "reasoning",
    "REASONING_SCRATCHPAD",
    "thought",
)
_TOOL_CALL_TAG_NAMES = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)

# 闭合推理块：<think>…</think>（含全部变体，大小写不敏感）
_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)
# 闭合工具调用 XML 块：<tool_call …>…</tool_call> 等
_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)
# Gemma 风格 <function name="…">…</function>（仅在行首或标点后，避免误伤正文）
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*"
    r"<function\b[^>]*\bname\s*=[^>]*>"
    r"(?:(?:(?!</function>).)*)</function>",
    re.DOTALL | re.IGNORECASE,
)
# 未闭合推理块（某些端点丢弃闭合标签）：从开标签到串尾整体剥掉
_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf"(?:^|\n)[ \t]*<(?:{'|'.join(_REASONING_TAG_NAMES)})\b[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)
# 游离的孤儿开/闭标签
_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf"</?(?:{'|'.join(_REASONING_TAG_NAMES)})>\s*",
    re.IGNORECASE,
)
# 漏网的闭合标签
_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf"</(?:{'|'.join(_TOOL_CALL_TAG_NAMES)}|function)>\s*",
    re.IGNORECASE,
)


def strip_think_blocks(agent, content: str) -> str:
    """从 content 中移除推理/思考块，只返回可见文本。

    对应原版 agent/agent_runtime_helpers.py:809 的精简版，处理四类：
      1. 闭合标签对（<think>…</think> 及 thinking/reasoning/
         REASONING_SCRATCHPAD/thought 变体，大小写不敏感）；
      2. 未闭合的开标签（从开标签到串尾剥掉）；
      3. 游离的孤儿开/闭标签；
      4. 开放模型在 content 里直接输出的工具调用 XML 块
         （tool_call / function_call / Gemma 风格 <function name=…>）。

    content 可能是 str、结构块列表（Anthropic 风格）或 dict。
    """
    if not content:
        return ""
    # 非字符串内容：展平为可见文本（跳过 thinking/reasoning 类型块）
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub("", content)
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub("", content)
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub("", content)
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub("", content)
    content = _ORPHAN_REASONING_TAG_PATTERN.sub("", content)
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub("", content)
    return content
