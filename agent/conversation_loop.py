"""对话循环：run_conversation 转发器的真正实现（精简版）。

结构与原版 agent/conversation_loop.py 对应：
- 序言：每回合一次性设置（build_turn_context 完成，本文件负责
  组装调用参数、定义回调函数、解包结果、初始化循环状态）；
- 主循环：中断/预算检查 → api_messages 组装 → API 调用（带重试）→
  工具执行 / 拿到最终回答退出。
"""

import time
from typing import Any, Dict, List, Optional

from agent.middleware import run_llm_execution_middleware
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
    return "".join(ch if not (0xD800 <= ord(ch) <= 0xDFFF) else "\ufffd" for ch in text)


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
    length_continue_retries = (
        0  # finish_reason=length 时的续写重试计数（精简版不实现续写，先占位）
    )
    _turn_exit_reason = "completed"  # 轮次退出原因（中断/预算/API失败/正常完成）
    # TODO fallback：每轮开始恢复主 provider（原版 restore_primary_runtime，
    #     功能待实现——agent_init 已保留 _fallback_chain 等状态与 fallback_model 参数）

    # ══════════════════════════════════════════════════════════════
    # 主循环：每轮 = "调 API → 有工具调用就执行工具继续下一轮，
    #               否则拿到最终回答退出"。最多跑 max_iterations 轮，
    #               受迭代预算（iteration_budget.remaining）约束。
    # ══════════════════════════════════════════════════════════════
    while (
        api_call_count < agent.max_iterations and agent.iteration_budget.remaining > 0
    ) or agent._budget_grace_call:
        # ① 中断检查：协作式中断——用户请求中断后，本轮到这就退出。
        #    正在执行的工具调用不会被强杀，等它自然结束
        if agent._interrupt_requested:
            interrupted = True
            _turn_exit_reason = "interrupted_by_user"
            if not agent.quiet_mode:
                agent._safe_print("\n⚡ Breaking out of tool loop due to interrupt...")
            break

        # ② 计数 + 预算消耗：正常轮次消耗一次迭代预算；
        #    _budget_grace_call（宽限调用）不占预算，用一次就清掉
        api_call_count += 1
        agent._api_call_count = api_call_count
        if agent._budget_grace_call:
            agent._budget_grace_call = False
        elif not agent.iteration_budget.consume():
            _turn_exit_reason = "budget_exhausted"
            if not agent.quiet_mode:
                agent._safe_print(
                    f"\n⚠️  Iteration budget exhausted "
                    f"({agent.iteration_budget.used}/{agent.iteration_budget.max_total} iterations used)"
                )
            break

        # ③ 组装发给 API 的消息：messages 是工作副本（带内部字段），
        #    发给 API 前要剥掉内部字段（display_kind/api_context/_row_id 等）
        api_messages = []
        for idx, msg in enumerate(messages):
            api_msg = msg.copy()

            # api_context：某些场景下消息的"API 专用内容"（如语音前缀），
            # 与转录展示的 content 不同。先取出，再按规则决定是否覆盖
            _api_content = api_msg.pop("api_context", None)
            api_msg.pop("display_kind", None)
            api_msg.pop("display_metadata", None)
            api_msg.pop("_row_id", None)

            if idx == current_turn_user_idx and msg.get("role") == "user":
                # 当前用户消息：若存在 api_context 且是有效字符串，覆盖 content
                if isinstance(_api_content, str) and _api_content:
                    api_msg["content"] = _api_content
            elif (
                isinstance(_api_content, str)
                and _api_content
                and msg.get("role") in ("user", "assistant")
            ):
                api_msg["content"] = _api_content

            # 推理内容回填：把轨迹里的 reasoning 复制成 API 的
            # reasoning_content（DeepSeek/Kimi 等 thinking 模式需要）
            agent._copy_reasoning_content_for_api(msg, api_msg)

            # 以下字段只用于轨迹存储/展示，部分严格 API 会拒绝：
            api_msg.pop("reasoning", None)  # 已复制到 reasoning_content，删掉原字段
            api_msg.pop("finish_reason", None)  # 不接受 finish_reason（如 Mistral）
            api_msg.pop("_thinking_prefill", None)  # 内部思考预填充标记

            api_messages.append(api_msg)

        # ④ 系统提示：拼到最前面。临时提示词（ephemeral）追加在系统提示之后
        effective_system = active_system_prompt or ""
        if getattr(agent, "ephemeral_system_prompt", None):
            effective_system = (
                effective_system + "\n\n" + str(agent.ephemeral_system_prompt).strip()
            )
        if effective_system:
            # 注意：是 [system] + api_messages 拼接，不是覆盖！
            # （原版 1713 行；这里修复了历史消息被丢弃的问题）
            api_messages = [
                {"role": "system", "content": effective_system}
            ] + api_messages

        # ⑤ 预填充消息（prefill）：插在系统提示之后、对话历史之前，
        #    仅本次 API 调用生效（如自动续写的引导语）
        for idx, pfm in enumerate(getattr(agent, "prefill_messages", None) or []):
            sys_offset = (
                1 if (api_messages and api_messages[0].get("role") == "system") else 0
            )
            api_messages.insert(sys_offset + idx, pfm.copy())

        # ⑥ 工具 schema：精简版直接使用 agent.tools（OpenAI 格式）；
        #    为空则不带 tools 参数（纯对话模式）
        tools_for_api = getattr(agent, "tools", None)

        # ⑦ API 调用准备
        api_start_time = time.time()
        retry_count = 0
        max_retries = getattr(agent, "_api_max_retries", 3)  # 失败重试上限
        finish_reason = "stop"
        response = None
        api_request_id = f"{turn_id}:api:{api_call_count}"  # 每次调用一个请求ID
        agent._current_api_request_id = api_request_id

        # ⑧ 内层循环：真正的 API 调用 + 失败重试（可中断 + 中间件 + 回退）。
        #    有流式消费者（stream_callback）→ 走流式，逐增量触发回调；
        #    否则走非流式。两条路径都可在调用期间响应中断

        def _perform_api_call(_api_kwargs: Dict[str, Any]):
            """实际发起 API 调用（中间件链的最内层）。

            有流式消费者 → 流式（逐增量触发回调）；否则非流式。
            """
            if agent._has_stream_consumers():
                return agent._interruptible_streaming_api_call(_api_kwargs)
            return agent._interruptible_api_call(_api_kwargs)

        while retry_count < max_retries:
            try:
                api_kwargs: Dict[str, Any] = {
                    "model": agent.model,
                    "messages": api_messages,
                }
                if tools_for_api:
                    api_kwargs["tools"] = tools_for_api
                # 经 LLM 执行中间件链发起调用（未注册任何中间件时零开销直通）
                response = run_llm_execution_middleware(
                    api_kwargs,
                    _perform_api_call,
                    agent=agent,
                    api_call_count=api_call_count,
                )
                break  # 成功拿到响应，退出重试循环
            except InterruptedError:
                # 用户中断：不重试，整轮退出（与循环顶部的中断检查殊途同归）
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                break
            except Exception as exc:
                retry_count += 1
                if retry_count >= max_retries:
                    # 重试耗尽：本轮标记失败，错误信息作为最终回复返回。
                    # TODO fallback：原版此处分类错误（classify_api_error）后
                    #     尝试切换到回退 provider（try_activate_fallback），
                    #     功能待实现（agent_init 已保留 fallback_model 参数）
                    failed = True
                    _turn_exit_reason = "api_failed"
                    final_response = f"API 调用失败（已重试 {max_retries} 次）: {exc}"
                    break
                if not getattr(agent, "quiet_mode", False):
                    agent._safe_print(
                        f"⚠️ API 调用失败（第 {retry_count}/{max_retries} 次）: "
                        f"{exc}，准备重试..."
                    )

        if failed or interrupted:
            break
        if response is None:
            # 防御：重试循环因故退出但没有响应（正常不会走到）
            failed = True
            _turn_exit_reason = "api_failed"
            final_response = "API 调用失败：所有重试均未获得响应"
            break

        # ⑨ 归一化响应：OpenAI chat.completions 的标准结构
        #    （choices[0].message + choices[0].finish_reason）
        assistant_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 内容归一化：部分 OpenAI 兼容服务器（llama-server 等）把 content
        #    返回为 dict/list，统一转成字符串，避免下游 .strip() 崩溃
        if assistant_message.content is not None and not isinstance(
            assistant_message.content, str
        ):
            raw = assistant_message.content
            if isinstance(raw, dict):
                assistant_message.content = (
                    raw.get("text", "") or raw.get("content", "") or str(raw)
                )
            elif isinstance(raw, list):
                parts = []
                for part in raw:
                    if isinstance(part, str):
                        parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                assistant_message.content = "\n".join(parts)
            else:
                assistant_message.content = str(raw)

        # ⑩ 分支：有工具调用 → 执行工具后继续下一轮；
        #     没有 → 模型给出了最终回答，收尾退出
        if assistant_message.tool_calls:
            if not getattr(agent, "quiet_mode", False):
                agent._safe_print(
                    f"🔧 Processing {len(assistant_message.tool_calls)} tool call(s)..."
                )

            # 把 assistant 消息（含 tool_calls）归档进 messages：
            # 工具结果会以 role="tool" 追加在它后面，下一轮模型才能看到
            tool_calls_payload = []
            for tc in assistant_message.tool_calls:
                tool_calls_payload.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": tool_calls_payload,
            })

            # 顺序执行工具调用：结果以 role="tool" 消息追加进 messages
            # （对应原版 6365 行）
            agent._execute_tool_calls(
                assistant_message, messages, effective_task_id, api_call_count
            )
        else:
            # 最终回答：把 assistant 消息归档进 messages（保持会话历史
            #    完整，下一轮对话能接上），取出 content 作为最终回复
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
            })
            final_response = assistant_message.content
            _turn_exit_reason = "completed"
            break

    # 循环自然退出（没拿到最终回答、也没中断/失败）→ 只能是预算或
    # 迭代次数耗尽，归一化为 budget_exhausted（对应原版 finalize_turn）
    if final_response is None and not interrupted and not failed:
        _turn_exit_reason = "budget_exhausted"

    # ══════════════════════════════════════════════════════════════
    # 收尾：组装结果 dict（对应原版 finalize_turn 的简化替代）
    # ══════════════════════════════════════════════════════════════
    # turn 结束时消费中断标志（对应原版 turn_finalizer.py:693 的
    # clear_interrupt()）。这样用户的一次中断只影响当前轮，不会
    # 污染下一轮对话。
    agent.clear_interrupt()

    return {
        "final_response": final_response,
        "messages": messages,
        "interrupted": interrupted,
        "failed": failed,
        "api_call_count": api_call_count,
        "turn_exit_reason": _turn_exit_reason,
    }
