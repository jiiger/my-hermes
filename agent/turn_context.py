"""每回合序言（turn prologue）：把一次对话的输入组装成循环可用的上下文。

对应原版 agent/turn_context.py 的 build_turn_context；精简版去掉了
session DB、上下文压缩预检、插件钩子、记忆提醒等，只保留核心组装。
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TurnContext:
    """Values produced by the turn prologue and consumed by the turn loop."""

    # Sanitized inbound message (surrogates stripped).
    user_message: str
    # Clean message preserved for transcripts / memory queries (no nudge injection).
    original_user_message: Any
    # Working message list for this turn (loop appends to it).
    messages: List[Dict[str, Any]]
    # May be reset to None by preflight compression (new session created).
    conversation_history: Optional[List[Dict[str, Any]]]
    # Cached system prompt active for this turn (may be rebuilt by compression).
    active_system_prompt: Optional[str]
    # Task / turn identifiers.
    effective_task_id: str
    turn_id: str
    # Index of the current user turn within ``messages``.
    current_turn_user_idx: int
    # Whether the post-turn memory review should fire (精简版恒为 False).
    should_review_memory: bool = False
    # Context contributed by ``pre_llm_call`` plugins (精简版恒为空串).
    plugin_user_context: str = ""
    # External-memory prefetch result (精简版恒为空串).
    ext_prefetch_cache: str = ""


def build_turn_context(
    agent,
    user_message: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
    task_id: Optional[str],
    stream_callback,
    persist_user_message: Optional[Any],
    persist_user_timestamp: Optional[float] = None,
    *,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
) -> TurnContext:
    """每回合一次的设置（序言），返回循环的输入上下文。

    从 conversation_loop 模块传入的回调函数是显式注入的，避免本模块与
    agent.conversation_loop 形成循环导入。

    流程：
    1. 安装安全的 stdin/stdout（线程安全防护）；
    2. 净化用户消息（去孤立代理对字符 surrogate）；
    3. 组装 messages = 对话历史副本 + 当前用户消息；
    4. 准备系统提示（会话内缓存，无则现场构建）；
    5. 生成任务/轮次 ID。
    """

    # 线程安全的 stdio 防护（守护线程/后台任务环境下避免 stdin/stdout 竞态）
    install_safe_stdio()

    # 挂载流式回调：调用方传入的 stream_callback 在本轮生效，
    # 主循环的流式分支（interruptible_streaming_api_call）会逐增量触发它
    if stream_callback is not None:
        agent._stream_callback = stream_callback

    # 净化用户消息：去掉孤立代理对字符，防止下游编码错误
    user_message = sanitize_surrogates(user_message)

    # 复制历史，避免修改调用方的列表（对应原版 turn_context.py:524）
    messages = list(conversation_history) if conversation_history else []

    # 组装当前用户消息（可携带展示类型元数据，仅影响转录展示，模型仍收到原文）
    user_msg: Dict[str, Any] = {"role": "user", "content": user_message}
    if persist_user_display_kind:
        user_msg["display_kind"] = persist_user_display_kind
        if persist_user_display_metadata:
            user_msg["display_metadata"] = persist_user_display_metadata
    messages.append(user_msg)
    # 当前用户消息在 messages 中的下标（压缩/续写等场景需要定位它）
    current_turn_user_idx = len(messages) - 1

    # 保留干净的用户消息原文（持久化/记忆查询用，不做任何注入）
    original_user_message = (
        persist_user_message if persist_user_message is not None else user_message
    )

    # 系统提示：会话内缓存，没有则现场构建（对应原版 turn_context.py:641-642）
    if getattr(agent, "_cached_system_prompt", None) is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)
    active_system_prompt = getattr(agent, "_cached_system_prompt", None)

    # 任务 / 轮次 ID：调用方没传任务ID就现场生成
    effective_task_id = task_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    # 回显预览（安静模式跳过；原版用 agent._safe_print，精简版直接 print）
    if not getattr(agent, "quiet_mode", False):
        _preview = summarize_user_message_for_log(user_message)
        print(f"💬 Starting conversation: '{_preview}'")

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
    )

