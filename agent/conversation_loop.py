"""对话循环：run_conversation 转发器的真正实现（精简版）。

结构与原版 agent/conversation_loop.py 对应：
- 序言：每回合一次性设置（build_turn_context 完成，本文件负责
  组装调用参数、定义回调函数、解包结果、初始化循环状态）；
- 主循环：TODO 下一步实现（API 调用 + 工具执行）。
"""

from typing import Any, Dict, List, Optional

from agent.process_bootstrap import _install_safe_stdio
from agent.turn_context import build_turn_context


def _restore_or_build_system_prompt(
    agent: Any,
    system_message: Optional[str],
    conversation_history: Optional[List[Dict[str, Any]]],
) -> None:
    """构建或复用系统提示（简化版，对应原版 conversation_loop.py:475）。

    原版会从 session DB 恢复上次的系统提示以命中前缀缓存；精简版没有
    session DB，规则退化为：
    - agent._cached_system_prompt 已存在 → 直接复用（会话内缓存）；
    - 否则现场构建：优先用调用方传入的 system_message，
      其次用 agent._build_system_prompt（若存在），最后用内置默认提示。
    结果写回 agent._cached_system_prompt，供本轮循环使用。
    """
    del conversation_history  # 精简版不做基于历史的状态恢复
    if getattr(agent, "_cached_system_prompt", None):
        return
    builder = getattr(agent, "_build_system_prompt", None)
    prompt = (
        builder(system_message)
        if callable(builder)
        else (system_message or "You are a helpful assistant.")
    )
    agent._cached_system_prompt = prompt


def _sanitize_surrogates(text: Any) -> Any:
    """去掉字符串中的孤立代理对字符（surrogate），防止下游编码错误。

    对应原版 agent.message_sanitization 的职责，精简版内联实现。
    """
    if not isinstance(text, str):
        return text
    return "".join(
        ch if not (0xD800 <= ord(ch) <= 0xDFFF) else "\ufffd" for ch in text
    )


def _summarize_user_message_for_log(message: Any, limit: int = 60) -> str:
    """把用户消息压成一行预览，供回显/日志使用。"""
    text = str(message).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def run_conversation(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    moa_config: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    完成一个完整的conversation

    参数:
    user_message(str):用户的消息/问题
    system_message(str):自定义系统消息(可选，如果已提供，则覆盖临时系统提示)
    conversation_history(List[Dict]):之前的对话消息(可选)
    task_id(str):此任务的唯一标识符，用于在并发任务之间隔离虚拟机(VM)(可选，如果未提供，则自动生成)
    stream_callback:流式处理期间，每次文本增量更新时调用的可选回调函数。
    persist_user_message:当user_message包含仅限API的合成前缀时，可选的干净用户消息。
    persist_user_timestamp:可选的平台事件时间戳。
    persist_user_display_kind:合成用户轮次的可选展示类型。
    persist_user_display_metadata:该事件的可选载荷。
    moa_config:多模型配置（精简版已裁剪 MoA，参数保留仅作占位）。

    返回:字典:包含最终回复和消息历史的完整对话结果
    """

    # ── 序言（每回合一次性设置）──

    # 压缩状态复位：上一轮若发生过就地压缩，其状态标记不能带到本轮。
    # 精简版暂无压缩功能，先保留占位（对应原版 1290-1292 行）
    agent._last_compaction_in_place = False
    agent._last_compression_attempt_recorded = False
    agent._last_compression_attempt_in_place = None

    # MoA（多模型聚合）已裁剪：原版在此解码 moa_config。
    # env 凭据刷新已裁剪：原版在此调用 agent._try_refresh_env_client_credentials()。

    # 每回合设置交给 build_turn_context：净化输入、组装 messages、
    # 准备系统提示、生成任务/轮次ID。返回循环所需的输入上下文
    _ctx = build_turn_context(
        agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        persist_user_timestamp,
        persist_user_display_kind=persist_user_display_kind,
        persist_user_display_metadata=persist_user_display_metadata,
        restore_or_build_system_prompt=_restore_or_build_system_prompt,
        install_safe_stdio=_install_safe_stdio,
        sanitize_surrogates=_sanitize_surrogates,
        summarize_user_message_for_log=_summarize_user_message_for_log,
    )

    # 从上下文解包本轮数据（循环与收尾都会用到）
    user_message = _ctx.user_message
    ouser_message = _ctx.user_message
    original_user_message = _ctx.original_user_message
    messages = _ctx.messages
    conversation_history = _ctx.conversation_history
    active_system_prompt = _ctx.active_system_prompt
    effective_task_id = _ctx.effective_task_id
    turn_id = _ctx.turn_id
    current_turn_user_idx = _ctx.current_turn_user_idx
    _should_review_memory = _ctx.should_review_memory
    _plugin_user_context = _ctx.plugin_user_context
    _ext_prefetch_cache = _ctx.ext_prefetch_cache

    # 本轮已交付的流式片段集合（供去重；精简版暂无流式，先占位）
    agent._delivered_interim_texts = set()
    # 增量持久化失败标记（配置的 SessionDB 追加失败只影响本轮；精简版暂无 DB）
    agent._incremental_persistence_failed = False

    # 循环状态初始化（对应原版 1352-1360 行）
    api_call_count = 0  # 已发起的 API 调用次数（对比 max_iterations 用）
    final_response = None  # 最终回复文本
    interrupted = False  # 是否被用户/上层中断
    failed = False  # 本轮是否失败
    length_continue_retries = 0  # finish_reason=length 时的续写重试计数

    # ══════════════════════════════════════════════════════════════
    # 主循环（下一步实现）
    #
    # 原版骨架（对应原版 1415 行起）：
    #   while (api_call_count < agent.max_iterations
    #          and agent.iteration_budget.remaining > 0) or agent._budget_grace_call:
    #       if agent._interrupt_requested:
    #           interrupted = True
    #           break
    #       api_call_count += 1
    #       api_messages = [系统提示] + messages      # 组装请求消息
    #       tools_for_api = agent.tools              # 工具 schema
    #       response = agent.client.chat.completions.create(
    #           model=agent.model, messages=api_messages, tools=tools_for_api)
    #       if response.tool_calls:
    #           messages.append(assistant 消息)
    #           agent._execute_tool_calls(response, messages,
    #                                     effective_task_id, api_call_count)
    #       else:
    #           final_response = response.content
    #           break
    #   最后返回 {"final_response": final_response, "messages": messages,
    #             "interrupted": interrupted, "failed": failed, ...}
    # ══════════════════════════════════════════════════════════════
    raise NotImplementedError(
        "主循环尚未实现：下一步补 while 循环（API 调用 + 工具执行），骨架见上方注释"
    )
