"""压缩编排层（精简移植版）—— 对应原版 agent/conversation_compression.py。

原版 4133 行，my-hermes 精简版只保留核心编排：
- check_compression_model_feasibility：辅助摘要模型上下文探测（启动时接线）；
- compress_context：压缩事件编排——压缩前通知外部记忆 provider、调用
  压缩引擎、失败回滚、压缩后系统提示失效 + 按需重建，返回
  ``(compressed_messages, new_system_prompt)``；
- 辅助：_supported_compression_kwargs / sanitize_memory_context /
  _snapshot_compressor_attempt_state / _restore_compressor_attempt_state。

砍掉（my-hermes 无对应系统，调用点保留注释占位）：
- SessionDB 压缩锁 / CompressionCommitFence / 锁租约刷新 / 会话轮换 /
  in-place DB 提交 / commit_memory_session（等 DB 接入后补）；
- 遥测 / 活动心跳 / codex app-server 路由 / 图片压缩恢复。

预留扩展点：
- 外部 memory provider：compress_context 内 ``on_pre_compress`` 调用点，
  ``agent._memory_manager`` 非 None 时生效（my-hermes 恒 None → 跳过）；
- DB session：compress_context 内「会话提交」占位（未来 SessionDB 接入
  后在此补 archive_and_compact / 轮换 publish_compression_child）。
"""

from __future__ import annotations

import copy
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 压缩摘要模型上下文窗口的下限（对齐原版 MINIMUM_CONTEXT_LENGTH）。
from agent.model_metadata import MINIMUM_CONTEXT_LENGTH


def sanitize_memory_context(memory_context: str) -> str:
    """为上下文引擎/LLM 出站边界准备 provider 上下文（对应原版
    context_engine.py:40 sanitize_memory_context 的简化版）。

    原版会做敏感文本脱敏（redact_sensitive_text），my-hermes 无 redact
    模块，只保留 strip + 超长截断。
    """
    sanitized = (memory_context or "").strip()
    if len(sanitized) <= 4000:
        return sanitized
    return sanitized[:1500] + "\n…[truncated]…\n" + sanitized[-1500:]


def _snapshot_compressor_attempt_state(compressor: Any) -> dict[str, Any]:
    """快照压缩器本轮 attempt 的关键状态（压缩失败/中止时回滚用）。

    对应原版 conversation_compression.py:284 的精简版：my-hermes 压缩器
    只暴露 _previous_summary / _summary_has_user_turn 两个跨尝试字段。
    """
    return {
        "_previous_summary": getattr(compressor, "_previous_summary", None),
        "_summary_has_user_turn": getattr(compressor, "_summary_has_user_turn", None),
    }


def _restore_compressor_attempt_state(compressor: Any, snapshot: dict[str, Any]) -> None:
    """恢复压缩器 attempt 状态（对应原版 :305）。"""
    for key, value in snapshot.items():
        try:
            setattr(compressor, key, value)
        except Exception:  # 只读属性等异常情况，尽力而为
            pass


def _supported_compression_kwargs(
    compress_fn: Any,
    *,
    current_tokens: Optional[int],
    focus_topic: Optional[str],
    force: bool,
    memory_context: str,
) -> dict:
    """只返回引擎 callable 接受的压缩 kwargs（对应原版 :1367，照抄）。

    上下文引擎插件可能早于可选宿主契约存在。调用前检查签名，避免捕获
    内部 TypeError 并让有状态的压缩器执行两次。
    """
    candidates = {
        "current_tokens": current_tokens,
        "focus_topic": focus_topic,
        "force": force,
    }
    if memory_context:
        candidates["memory_context"] = memory_context
    try:
        parameters = inspect.signature(compress_fn).parameters
    except (TypeError, ValueError):
        # current_tokens 自 ContextEngine ABC 引入以来就是契约的一部分。
        # 对不可检查签名的 callable（C 扩展等）保留最旧的调用形态。
        return {"current_tokens": current_tokens}

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_kwargs:
        return candidates
    return {name: value for name, value in candidates.items() if name in parameters}


def check_compression_model_feasibility(agent: Any) -> None:
    """启动时探测辅助压缩模型上下文窗口能否容纳主模型压缩阈值。

    对应原版 conversation_compression.py:1597 的精简版：
    - 摘要模型 = compressor.summary_model（独立配置）或主模型；
    - 用 model_metadata.get_default_context_length 解析摘要模型上下文长度；
    - 低于 MINIMUM_CONTEXT_LENGTH(64K) → 警告（原版 raise ValueError 中止
      会话；my-hermes 无状态通道，降级为警告 + 记录 _compression_warning，
      压缩将退化为静态摘要/修剪）；
    - 低于主模型压缩阈值 → 自动降低会话阈值（对齐原版 auto-correct，
      新阈值 >= 64K，因为上面已过下限检查）。

    在 agent_init 挂载压缩器后调用一次；compress_context 首次压缩时惰性
    补调（原版语义：短会话从不触发压缩则省掉探测开销）。
    """
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return
    try:
        from agent.model_metadata import get_default_context_length

        aux_model = getattr(compressor, "summary_model", None) or getattr(agent, "model", "") or ""
        aux_base_url = getattr(compressor, "base_url", "") or ""
        aux_context = get_default_context_length(aux_model, aux_base_url)

        if aux_context < MINIMUM_CONTEXT_LENGTH:
            msg = (
                f"⚠ 压缩摘要模型 {aux_model} 的上下文窗口 {aux_context:,} tokens "
                f"低于最低要求 {MINIMUM_CONTEXT_LENGTH:,}。压缩将跳过 LLM 摘要、"
                f"直接修剪中间轮次。可在 config.yaml 的 compression 段设置 "
                f"summary_model / context_length 修正。"
            )
            agent._compression_warning = msg
            logger.warning(
                "Auxiliary compression model %s context %d < MINIMUM %d — "
                "summaries unavailable.",
                aux_model, aux_context, MINIMUM_CONTEXT_LENGTH,
            )
            return

        threshold = compressor.threshold_tokens
        if aux_context < threshold:
            old_threshold = threshold
            # 自动修正：降低会话阈值让压缩在本会话真正可用。
            # 摘要请求是单条 user 提示（无系统提示、无工具），
            # new_threshold == aux_context 是安全的。
            compressor.threshold_tokens = max(aux_context, MINIMUM_CONTEXT_LENGTH)
            msg = (
                f"⚠ 压缩摘要模型 {aux_model} 的上下文窗口 {aux_context:,} tokens "
                f"小于主模型压缩阈值 {old_threshold:,}。已自动将会话压缩阈值"
                f"降低到 {compressor.threshold_tokens:,}。"
            )
            agent._compression_warning = msg
            logger.warning(
                "Aux compression model %s context %d < threshold %d — "
                "lowered session threshold to %d.",
                aux_model, aux_context, old_threshold, compressor.threshold_tokens,
            )
    except Exception as exc:
        logger.warning("Compression model feasibility check failed: %s", exc)


def _existing_system_prompt(agent: Any, system_message: str) -> str:
    """返回现有缓存系统提示，没有则现场构建（对应原版多处 _existing_sp 模式）。"""
    existing = getattr(agent, "_cached_system_prompt", None)
    if existing:
        return existing
    from agent.system_prompt import build_system_prompt

    return build_system_prompt(agent, system_message)


def compress_context(
    agent: Any,
    messages: list,
    system_message: str,
    *,
    approx_tokens: Optional[int] = None,
    task_id: str = "default",
    focus_topic: Optional[str] = None,
    force: bool = False,
) -> Tuple[list, str]:
    """压缩对话上下文，返回 ``(compressed_messages, new_system_prompt)``。

    对应原版 conversation_compression.py:2150 compress_context 的精简版。
    压缩中止/无进展时返回原消息 + 现有系统提示（调用方以
    ``len(returned) == len(input)`` 检测 no-op）。
    """
    # 1. 快照压缩器 attempt 状态（失败/中止时回滚，对齐原版 :2214）
    _compressor_attempt_snapshot = _snapshot_compressor_attempt_state(
        agent.context_compressor
    )
    agent._last_compression_attempt_recorded = True
    agent._last_compression_attempt_in_place = None

    # TODO 原版在此处理 codex app-server 路由（my-hermes 无 codex，跳过）

    # 2. 惰性可行性探测：首次压缩前探测辅助摘要模型上下文（对应原版 :2268）。
    #    探测失败不影响压缩（check 内部吞异常），标记只在成功后置位。
    if not getattr(agent, "_compression_feasibility_checked", False):
        check_compression_model_feasibility(agent)
        agent._compression_feasibility_checked = True

    # 3. in-place 压缩（原版 config compression.in_place，默认 True）。
    #    my-hermes 无 SessionDB，天然恒 in-place（不轮换 session_id）。
    in_place = bool(getattr(agent, "compression_in_place", True))

    # TODO 原版在此：SessionDB 压缩锁获取 / 锁刷新 / 会话恢复 / 采用
    #     live child（my-hermes 无 SessionDB，DB 接入后在此补锁与恢复）。

    # 4. 压缩前：外部 memory provider 提取洞见（预留扩展点）。
    #    原版 :2825 on_pre_compress——provider 从即将被丢弃的消息里提取
    #    洞见，返回文本注入压缩摘要 prompt 让洞见随摘要存活。
    memory_context = ""
    if getattr(agent, "_memory_manager", None):
        try:
            _maybe_ctx = agent._memory_manager.on_pre_compress(messages)
            if isinstance(_maybe_ctx, str):
                memory_context = sanitize_memory_context(_maybe_ctx)
        except Exception:
            pass

    # 5. 调用压缩引擎（按签名过滤 kwargs，兼容未来插件引擎）
    compress_fn = agent.context_compressor.compress
    compress_kwargs = _supported_compression_kwargs(
        compress_fn,
        current_tokens=approx_tokens,
        focus_topic=focus_topic,
        force=force,
        memory_context=memory_context,
    )
    messages_before_compression = copy.deepcopy(messages)
    try:
        compressed = compress_fn(messages, **compress_kwargs)
    except BaseException:
        # 任何压缩调用异常：回滚 attempt 状态与消息，再向上抛
        _restore_compressor_attempt_state(
            agent.context_compressor, _compressor_attempt_snapshot
        )
        if messages != messages_before_compression:
            messages[:] = copy.deepcopy(messages_before_compression)
        raise

    # 6. 中止 / 无进展检测：返回原消息 + 现有系统提示，不碰会话
    if getattr(agent.context_compressor, "_last_compress_aborted", False):
        # 摘要 LLM 失败且 abort_on_summary_failure=True：向用户解释并跳过
        _restore_compressor_attempt_state(
            agent.context_compressor, _compressor_attempt_snapshot
        )
        _err = (
            getattr(agent.context_compressor, "_last_summary_error", None)
            or "unknown error"
        )
        if getattr(agent, "_last_compression_summary_warning", None) != _err:
            agent._last_compression_summary_warning = _err
            logger.warning(
                "Compression aborted: %s. No messages were dropped — "
                "conversation continues unchanged.",
                _err,
            )
        return messages, _existing_system_prompt(agent, system_message)

    if compressed == messages_before_compression:
        # 语义等值而非对象同一性比较：legacy/插件引擎可能返回等值副本
        _restore_compressor_attempt_state(
            agent.context_compressor, _compressor_attempt_snapshot
        )
        logger.info(
            "Compression made no progress (session=%s) — skipping boundary rewrite.",
            getattr(agent, "session_id", "") or "none",
        )
        return messages, _existing_system_prompt(agent, system_message)

    # 7. 压缩后：系统提示失效 + 按需重建（对齐原版 :3208-3255）。
    #    内置记忆是压缩会重载的唯一系统提示输入（invalidate 里
    #    load_from_disk）；若重载后的记忆块已逐字包含在缓存提示里则保留
    #    原提示（保住前缀缓存），记忆变了才重建渲染新快照。有外部
    #    memory provider 时强制走重建（其提示块可能在 on_pre_compress
    #    里变化——原版语义）。
    cached_system_prompt = getattr(agent, "_cached_system_prompt", None)
    from agent.system_prompt import (
        cached_prompt_reflects_builtin_memory as _prompt_reflects_memory,
        invalidate_system_prompt as _invalidate_system_prompt,
    )

    _invalidate_system_prompt(agent)
    if (
        cached_system_prompt is not None
        and getattr(agent, "_memory_manager", None) is None
        and _prompt_reflects_memory(agent, cached_system_prompt)
    ):
        new_system_prompt = cached_system_prompt
        agent._cached_system_prompt = cached_system_prompt
    else:
        from agent.system_prompt import build_system_prompt

        new_system_prompt = build_system_prompt(agent, system_message)
        agent._cached_system_prompt = new_system_prompt

    # TODO 原版在此：SessionDB 会话提交——
    #     1. commit_memory_session(messages)：转录被摘要替换前触发记忆提取；
    #     2. in_place：archive_and_compact 软归档旧轮次 + 插入压缩后消息；
    #     3. 轮换（in_place=False）：publish_compression_child 建子会话、
    #        轮换 agent.session_id。
    #     my-hermes 无 SessionDB，DB 接入后在此补会话落库。_last_compaction_in_place
    #     语义：my-hermes 恒 in-place（不轮换 id）。
    del in_place
    agent._last_compaction_in_place = True

    # 8. 收尾：记录粗略 token 估算（对齐原版 :3658）
    try:
        _compressed_est = sum(
            _rough_message_tokens(m) for m in compressed
        ) + (len(new_system_prompt or "") // 4)
        agent.context_compressor.last_compression_rough_tokens = _compressed_est
    except Exception:
        pass

    logger.info(
        "context compression done: session=%s messages=%d→%d",
        getattr(agent, "session_id", "") or "none",
        len(messages_before_compression),
        len(compressed),
    )
    return compressed, new_system_prompt


def _rough_message_tokens(message: Dict[str, Any]) -> int:
    """粗略估算单条消息 token（压缩收尾日志用，非精确）。"""
    text = message.get("content", "")
    if isinstance(text, str):
        return len(text) // 4
    return 0
