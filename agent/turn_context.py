"""每回合序言（turn prologue）：把一次对话的输入组装成循环可用的上下文。

对应原版 agent/turn_context.py 的 build_turn_context；精简版去掉了
session DB、上下文压缩预检、插件钩子、记忆提醒等，只保留核心组装。
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.iteration_budget import IterationBudget
from tools.interrupt import set_interrupt

logger = logging.getLogger(__name__)


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

    # Between-turns MCP 刷新（对齐原版 agent/turn_context.py:509）：
    # 上一轮之后才连上的 MCP server（慢 HTTP/npx/uvx 冷启动通常 2-6s，
    # 错过 agent 构建前的有界等待）在本轮快照里落地。本轮 tools= 前缀
    # 尚未组装，刷新只扩展全新请求前缀，缓存安全。无 MCP 时 no-op。
    # import-cost 门：tools.mcp_tool 会拉进整个 mcp 包（实测 ~0.4s），
    # 零 MCP server 配置的用户首轮不该走重导入路径——MCP 工具只能由
    # 已 import tools.mcp_tool 的代码注册，若它不在 sys.modules 就没有
    # 可刷新的东西，直接跳过。
    try:
        if not getattr(agent, "_skip_mcp_refresh", False):
            import sys as _sys

            if "tools.mcp_tool" in _sys.modules:
                from tools.mcp_tool import (
                    has_registered_mcp_tools,
                    refresh_agent_mcp_tools,
                )

                if has_registered_mcp_tools():
                    refresh_agent_mcp_tools(agent, quiet_mode=True)
    except Exception:
        logger.debug("between-turns MCP tool refresh skipped", exc_info=True)

    # 净化用户消息：去掉孤立代理对字符，防止下游编码错误
    user_message = sanitize_surrogates(user_message)
    if isinstance(persist_user_message, str):
        persist_user_message = sanitize_surrogates(persist_user_message)

    # 挂载 persist override 状态（只影响写库行，不改 live messages；
    # 下标在 user 消息入列后更新，对应原版 turn_context.py:537-539）
    agent._persist_user_message_idx = None
    agent._persist_user_message_override = persist_user_message
    agent._persist_user_message_timestamp = persist_user_timestamp

    # 会话历史默认来源：调用方显式传入的 conversation_history 优先；
    # 未传入但配置了 SessionDB 时，从 state.db 恢复既有消息作为默认历史
    # （对应原版 turn_context.py:465-468 的恢复路径）。
    if conversation_history is None:
        _db = getattr(agent, "_session_db", None)
        _session_id = getattr(agent, "session_id", None)
        if _db is not None and _session_id:
            try:
                # 恢复当前活动上下文：默认只读 active=1 消息（压缩归档的
                # 旧消息不进入工作上下文；对应原版
                # get_messages_as_conversation 语义）。
                _restored = _db.get_messages_as_conversation(_session_id)
                if _restored:
                    conversation_history = _restored
            except Exception:
                # 恢复失败不阻断回合：退化为无历史
                conversation_history = None

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
    agent._persist_user_message_idx = current_turn_user_idx

    # 保留干净的用户消息原文（持久化/记忆查询用，不做任何注入）
    original_user_message = (
        persist_user_message if persist_user_message is not None else user_message
    )

    # 系统提示：会话内缓存，没有则现场构建（对应原版 turn_context.py:641-642）
    if getattr(agent, "_cached_system_prompt", None) is None:
        restore_or_build_system_prompt(agent, system_message, conversation_history)
    active_system_prompt = getattr(agent, "_cached_system_prompt", None)

    # 幂等创建 SessionDB 会话行（此时 _cached_system_prompt 已就绪，
    # 快照能带上非 NULL 系统提示；对应原版 turn_context.py:721-739）。
    # 失败只警告，不阻断回合——首次 flush 会重试。
    try:
        agent._ensure_db_session()
    except Exception:
        pass

    # 外部记忆 provider：通知新回合 + 回合前召回（prefetch_all）。
    # 跳过琐碎提示（问候/确认）——零信号文本不值得召回（对齐原版
    # turn_context.py:1248-1270）。召回结果存入 ext_prefetch_cache，
    # 由 conversation_loop 围栏注入当前用户消息。
    ext_prefetch_cache = ""
    if getattr(agent, "_memory_manager", None):
        try:
            agent._memory_manager.on_turn_start(0, str(user_message))
        except Exception:
            pass
        from agent.memory_provider import is_trivial_prompt

        if not is_trivial_prompt(str(user_message)):
            try:
                ext_prefetch_cache = agent._memory_manager.prefetch_all(
                    str(user_message)
                ) or ""
            except Exception:
                pass

    # 后台记忆提炼触发（对齐原版 turn_context.py:684-692）：memory 工具
    # 启用且每 N 轮（memory.nudge_interval）触发一次；回合后由
    # conversation_loop 据此 spawn background review。
    should_review_memory = False
    if (
        getattr(agent, "_memory_nudge_interval", 0) > 0
        and "memory" in getattr(agent, "valid_tool_names", set())
        and getattr(agent, "_memory_store", None)
    ):
        agent._turns_since_memory = getattr(agent, "_turns_since_memory", 0) + 1
        if agent._turns_since_memory >= agent._memory_nudge_interval:
            should_review_memory = True
            agent._turns_since_memory = 0

    # 任务 / 轮次 ID：调用方没传任务ID就现场生成
    effective_task_id = task_id or str(uuid.uuid4())
    turn_id = str(uuid.uuid4())

    # 回显预览（安静模式跳过；原版用 agent._safe_print，精简版直接 print）
    if not getattr(agent, "quiet_mode", False):
        _preview = summarize_user_message_for_log(user_message)
        print(f"💬 Starting conversation: '{_preview}'")

    # 每轮对话开始时重建独立迭代预算（对应原版 turn_context.py:500）。
    # 同一个 agent 连续多轮对话时，每轮都拿到满额预算，互不蚕食；
    # 原版同样以「换新对象」而非 reset() 实现轮次间预算重置。
    agent.iteration_budget = IterationBudget(agent.max_iterations)

    # 每回合重置内置记忆的整合失败预算（对应原版 turn_context.py:571，
    # issue #42405）：防循环上限计「连续失败」，跨回合清零。
    _reset_consol = getattr(
        getattr(agent, "_memory_store", None), "reset_consolidation_failures", None
    )
    if callable(_reset_consol):
        _reset_consol()

    # 记录执行线程，让 interrupt()/clear_interrupt() 能把工具级中断信号
    # 精确限定到本 agent 的线程（对应原版 turn_context.py:1237-1245）。
    agent._execution_thread_id = threading.current_thread().ident

    # 清除陈旧的按线程中断状态，同时保留一个待处理的中断。
    set_interrupt(False, agent._execution_thread_id)
    if agent._interrupt_requested:
        set_interrupt(True, agent._execution_thread_id)
        agent._interrupt_thread_signal_pending = False
    else:
        agent._interrupt_message = None

    return TurnContext(
        user_message=user_message,
        original_user_message=original_user_message,
        messages=messages,
        conversation_history=conversation_history,
        active_system_prompt=active_system_prompt,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        current_turn_user_idx=current_turn_user_idx,
        ext_prefetch_cache=ext_prefetch_cache,
        should_review_memory=should_review_memory,
    )

