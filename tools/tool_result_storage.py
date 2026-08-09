"""工具结果持久化 —— 保留大输出而不是截断（精简移植版）。

对应原版 hermes-agent 的 tools/tool_result_storage.py（254 行）。

防上下文窗口溢出的三道防线：
1. **单工具输出上限**（工具内部）：search_files 等工具在返回前自己截断
   输出。这是第一道防线，也是工具作者唯一可控的。
2. **单结果持久化**（maybe_persist_tool_result）：工具返回后若输出超过
   该工具注册的阈值（registry.get_max_result_size），完整输出写入临时
   目录（/tmp/hermes-results/{tool_use_id}.txt），上下文内替换为预览 +
   文件路径引用。
3. **单轮聚合预算**（enforce_turn_budget）：单个 assistant 回合的所有
   工具结果收集完后，若总量超过 turn_budget（200K），把最大的未持久化
   结果落盘直到总量回到预算内。

精简版砍掉：
- env / sandbox / environments 相关（my-hermes 没有这些系统，env 恒为
  None）；_resolve_storage_dir 直接返回 STORAGE_DIR，_write_to_sandbox
  改用本地文件写入（不再经 env.execute），写入失败回退内联截断。
- 死代码 _heredoc_marker / HEREDOC_MARKER（原版中未被调用）。

generate_preview / maybe_persist_tool_result / enforce_turn_budget 的
签名与内联截断路径与原版一致。
"""

import hashlib
import logging
import os
import re

from tools.budget_config import (
    DEFAULT_PREVIEW_SIZE_CHARS,
    BudgetConfig,
    DEFAULT_BUDGET,
)

logger = logging.getLogger(__name__)
PERSISTED_OUTPUT_TAG = "<persisted-output>"
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"
STORAGE_DIR = "/tmp/hermes-results"
_BUDGET_TOOL_NAME = "__budget_enforcement__"
_UNSAFE_RESULT_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")
_MAX_RESULT_FILENAME_STEM = 120


def _resolve_storage_dir(env) -> str:
    """返回临时存储目录（精简版恒为 STORAGE_DIR）。

    原版会尝试 env.get_temp_dir()（沙箱/远端后端）；my-hermes 没有
    environments 系统，env 恒为 None，直接返回默认目录即可。
    """
    return STORAGE_DIR


def _safe_result_filename(tool_use_id: str) -> str:
    """返回工具结果 ID 对应的安全文件名（对应原版 :65）。"""
    raw_id = str(tool_use_id or "tool_result")
    safe_stem = _UNSAFE_RESULT_FILENAME_CHARS.sub("_", raw_id).strip("._-")
    changed = safe_stem != raw_id

    if not safe_stem:
        safe_stem = "tool_result"
        changed = True

    if changed or len(safe_stem) > _MAX_RESULT_FILENAME_STEM:
        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]
        safe_stem = safe_stem[:_MAX_RESULT_FILENAME_STEM].rstrip("._-") or "tool_result"
        safe_stem = f"{safe_stem}_{digest}"

    return f"{safe_stem}.txt"


def generate_preview(content: str, max_chars: int = DEFAULT_PREVIEW_SIZE_CHARS) -> tuple[str, bool]:
    """在 max_chars 内按最后一个换行截断。返回 (preview, has_more)。"""
    if len(content) <= max_chars:
        return content, False
    truncated = content[:max_chars]
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        truncated = truncated[:last_nl + 1]
    return truncated, True


def _write_to_sandbox(content: str, remote_path: str, env) -> bool:
    """把内容写入临时目录（本地文件实现），成功返回 True。

    原版通过 env.execute() 把内容经 stdin 推给沙箱/远端后端；
    my-hermes 没有 environments 系统，直接用本地文件写入
    （/tmp/hermes-results/{tool_use_id}.txt），失败返回 False，
    由调用方回退内联截断。
    """
    try:
        storage_dir = os.path.dirname(remote_path)
        os.makedirs(storage_dir, exist_ok=True)
        with open(remote_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return True
    except OSError as exc:
        logger.debug("无法写入工具结果到 %s: %s", remote_path, exc)
        return False


def _build_persisted_message(
    preview: str,
    has_more: bool,
    original_size: int,
    file_path: str,
) -> str:
    """构造 <persisted-output> 替换块（对应原版 :109）。"""
    size_kb = original_size / 1024
    if size_kb >= 1024:
        size_str = f"{size_kb / 1024:.1f} MB"
    else:
        size_str = f"{size_kb:.1f} KB"

    msg = f"{PERSISTED_OUTPUT_TAG}\n"
    msg += f"This tool result was too large ({original_size:,} characters, {size_str}).\n"
    msg += f"Full output saved to: {file_path}\n"
    msg += "Use the read_file tool with offset and limit to access specific sections of this output.\n\n"
    msg += f"Preview (first {len(preview)} chars):\n"
    msg += preview
    if has_more:
        msg += "\n..."
    msg += f"\n{PERSISTED_OUTPUT_CLOSING_TAG}"
    return msg


def maybe_persist_tool_result(
    content: str,
    tool_name: str,
    tool_use_id: str,
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
    threshold: int | float | None = None,
) -> str:
    """Layer 2：把超大结果持久化到临时目录，返回预览 + 路径。

    写盘失败或无 env 时回退内联截断。

    Args:
        content: 原始工具结果字符串。
        tool_name: 工具名（用于阈值查询）。
        tool_use_id: 本次工具调用的唯一 ID（用作文件名）。
        env: 环境实例。my-hermes 没有 environments 系统，恒为 None
             （保留参数以对齐原版签名）。
        config: 控制阈值与预览大小的 BudgetConfig。
        threshold: 显式阈值覆盖；优先于 config 解析结果。

    Returns:
        内容小则原样返回；大则返回 <persisted-output> 替换块或内联截断。
    """
    effective_threshold = threshold if threshold is not None else config.resolve_threshold(tool_name)

    if effective_threshold == float("inf"):
        return content

    if len(content) <= effective_threshold:
        return content

    storage_dir = _resolve_storage_dir(env)
    remote_path = f"{storage_dir}/{_safe_result_filename(tool_use_id)}"
    preview, has_more = generate_preview(content, max_chars=config.preview_size)

    try:
        if _write_to_sandbox(content, remote_path, env):
            logger.info(
                "持久化超大工具结果：%s（%s，%d 字符 → %s）",
                tool_name, tool_use_id, len(content), remote_path,
            )
            return _build_persisted_message(preview, has_more, len(content), remote_path)
    except Exception as exc:
        logger.warning("写入工具结果失败 %s: %s", tool_use_id, exc)

    logger.info(
        "内联截断超大工具结果：%s（%d 字符，无落盘）",
        tool_name, len(content),
    )
    return (
        f"{preview}\n\n"
        f"[Truncated: tool response was {len(content):,} chars. "
        f"Full output could not be saved to sandbox.]"
    )


def enforce_turn_budget(
    tool_messages: list[dict],
    env=None,
    config: BudgetConfig = DEFAULT_BUDGET,
) -> list[dict]:
    """Layer 3：强制单个回合内所有工具结果的聚合预算。

    若总字符数超过预算，优先把最大的未持久化结果落盘（本地写文件）直到
    回到预算内。已持久化的结果跳过。

    原地修改列表并返回它。
    """
    candidates = []
    total_size = 0
    for i, msg in enumerate(tool_messages):
        content = msg.get("content", "")
        size = len(content)
        total_size += size
        if PERSISTED_OUTPUT_TAG not in content:
            candidates.append((i, size))

    if total_size <= config.turn_budget:
        return tool_messages

    candidates.sort(key=lambda x: x[1], reverse=True)

    for idx, size in candidates:
        if total_size <= config.turn_budget:
            break
        msg = tool_messages[idx]
        content = msg["content"]
        tool_use_id = msg.get("tool_call_id", f"budget_{idx}")

        replacement = maybe_persist_tool_result(
            content=content,
            tool_name=_BUDGET_TOOL_NAME,
            tool_use_id=tool_use_id,
            env=env,
            config=config,
            threshold=0,
        )
        if replacement != content:
            total_size -= size
            total_size += len(replacement)
            tool_messages[idx]["content"] = replacement
            logger.info(
                "预算强制：持久化工具结果 %s（%d 字符）",
                tool_use_id, size,
            )

    return tool_messages
