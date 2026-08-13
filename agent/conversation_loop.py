"""对话循环：run_conversation 转发器的真正实现（精简版）。

结构与原版 agent/conversation_loop.py 对应：
- 序言：每回合一次性设置（build_turn_context 完成，本文件负责
  组装调用参数、定义回调函数、解包结果、初始化循环状态）；
- 主循环：中断/预算检查 → api_messages 组装 → API 调用（带重试）→
  工具执行 / 拿到最终回答退出。
"""

import time
from typing import Any, Dict, List, Optional

from agent.error_classifier import classify_api_error
from agent.process_bootstrap import _install_safe_stdio
from agent.conversation_compression import conversation_history_after_compression
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
    agent._last_compaction_in_place = False
    agent._last_compression_attempt_recorded = False
    agent._last_compression_attempt_in_place = None
    # 错误触发压缩的次数（turn 级，跨外层迭代累计，对应原版 :1579）
    compression_attempts = 0

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

    # 本轮已交付的中间（interim）旁白文本集合（供去重）。流式响应已实现
    # （_interruptible_streaming_api_call 逐增量触发回调），但"工具循环
    # 中间旁白交付"（interim_assistant_callback 去重推送）尚未移植——
    # 本属性占位，待 interim 交付实现后激活（对齐原版 run_agent.py:6313）。
    agent._delivered_interim_texts = set()
    # 增量持久化失败标记（SessionDB 追加失败只影响本轮，下次 flush 重试）
    agent._incremental_persistence_failed = False
    # 当前 user 消息已入列：API 调用前先落库一次（也保证压缩前原始历史
    # 已持久化，压缩后重建的消息不会重复写入）。
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        agent._incremental_persistence_failed = True

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
    try:
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
    
            # ①½ 主动压缩检查：上一轮 API 响应后已 update_from_response，
            #    这里用真实用量判定是否超阈值；命中则压缩消息历史后重新组装。
            _compressor = getattr(agent, "context_compressor", None)
            # 主动工具结果剪枝（对齐原版 conversation_loop.py:6930）：大窗口
            # 模型远在压缩阈值前回收旧 tool result 的重发成本。确定性、无
            # LLM 调用；未配置 proactive_prune_tokens 或回收量不达标时 no-op
            # （返回原对象）。剪枝会重写已发送历史 → 使 flush 前缀快照失效，
            # 下次 flush 全量重扫。
            _prune = getattr(_compressor, "prune_tool_results_only", None)
            if callable(_prune):
                try:
                    _pruned_msgs, _pruned_n = _prune(
                        messages,
                        current_tokens=getattr(
                            _compressor, "last_prompt_tokens", None
                        ),
                    )
                    if _pruned_n and _pruned_msgs is not None and _pruned_msgs is not messages:
                        messages = _pruned_msgs
                        # 剪枝已 archive_and_compact 持久化：把 flush baseline
                        # 同步为剪枝后列表（对齐压缩后
                        # conversation_history_after_compression 的语义）。
                        # 否则压缩重建出的无 marker 消息在浅拷贝后既不在
                        # history_ids 也无 marker，会被 flush 重复 INSERT。
                        conversation_history = list(_pruned_msgs)
                        agent._db_flush_scan_prefix = None
                except Exception:
                    pass  # 剪枝失败不阻断回合
            if _compressor is not None and _compressor.should_compress():
                if not agent._interrupt_requested:
                    # 压缩编排层：内部完成压缩前记忆通知、引擎调用、失败回滚、
                    # 压缩后系统提示失效 + 按需重建（对齐原版
                    # conversation_compression.py:compress_context），返回
                    # (压缩后消息, 新系统提示)。
                    from agent.conversation_compression import compress_context
    
                    _new_messages, _new_system_prompt = compress_context(
                        agent,
                        messages,
                        system_message,
                        approx_tokens=getattr(_compressor, "last_prompt_tokens", None) or None,
                    )
                    if len(_new_messages) < len(messages):
                        messages = _new_messages
                        # 压缩后以压缩结果为 flush baseline：archive_and_compact
                        # 已把压缩后消息写入 DB active 集，后续 flush 只盖章
                        # 不重写（对齐原版 conversation_history_after_compression）
                        conversation_history = conversation_history_after_compression(
                            agent, messages
                        )
                        # 压缩后本轮循环立即使用新系统提示（对齐原版重试路径语义）
                        active_system_prompt = _new_system_prompt
                        # 压缩保护 tail（含最近 user 消息）；若下标越界则重置
                        # 到最后一条 user 消息（保守，避免指向被压缩掉的中间轮）
                        if current_turn_user_idx >= len(messages):
                            for _i in range(len(messages) - 1, -1, -1):
                                if messages[_i].get("role") == "user":
                                    current_turn_user_idx = _i
                                    break
                            else:
                                current_turn_user_idx = max(0, len(messages) - 1)
    
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
                    # 外部记忆 provider 召回结果围栏注入当前用户消息
                    # （对齐原版 turn_context.py:1294 _ext_prefetch_cache 注入）
                    if _ext_prefetch_cache:
                        from agent.memory_manager import build_memory_context_block
    
                        _mem_block = build_memory_context_block(_ext_prefetch_cache)
                        if _mem_block:
                            _cur = api_msg.get("content")
                            if isinstance(_cur, str):
                                api_msg["content"] = f"{_cur}\n\n{_mem_block}"
                            elif isinstance(_cur, list):
                                api_msg["content"] = _cur + [
                                    {"type": "text", "text": f"\n\n{_mem_block}"}
                                ]
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

            # ⑦½ API 调用前持久化：上一轮的工具调用/结果（或本轮 user 消息）
            #    已入列，先落库再发请求（失败只标记，不阻断 API）。
            try:
                agent._flush_messages_to_session_db(messages, conversation_history)
            except Exception:
                agent._incremental_persistence_failed = True
    
            # ⑦ API 调用准备
            api_start_time = time.time()
            retry_count = 0
            max_retries = getattr(agent, "_api_max_retries", 3)  # 失败重试上限
            finish_reason = "stop"
            response = None
            api_request_id = f"{turn_id}:api:{api_call_count}"  # 每次调用一个请求ID
            agent._current_api_request_id = api_request_id
            # 错误触发压缩的次数上限（对应原版 max_compression_attempts，缺省 3）
            # 与"压缩后回退外层循环重试"标志（对齐原版 _retry.restart_with_compressed_messages）
            _max_compression_attempts = getattr(agent, "max_compression_attempts", 3)
            _restart_with_compressed_messages = False
    
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
                    # 直接发起 API 调用（原版经 LLM 执行中间件链；my-hermes
                    # 无插件系统、无任何注册方，中间件已砍掉）
                    response = _perform_api_call(api_kwargs)
                    # 成功后记录真实用量：主动压缩判定依赖它
                    _usage = getattr(response, "usage", None)
                    if _usage is not None and _compressor is not None:
                        # OpenAI SDK 的 usage 是对象（CompletionUsage），
                        # 归一化为 dict 再交给压缩机（对齐原版做法）
                        if not hasattr(_usage, "get"):
                            _usage = {
                                "prompt_tokens": getattr(_usage, "prompt_tokens", 0),
                                "completion_tokens": getattr(
                                    _usage, "completion_tokens", 0
                                ),
                                "total_tokens": getattr(_usage, "total_tokens", None),
                            }
                        _compressor.update_from_response(_usage)
                    break  # 成功拿到响应，退出重试循环
                except InterruptedError:
                    # 用户中断：不重试，整轮退出（与循环顶部的中断检查殊途同归）
                    interrupted = True
                    _turn_exit_reason = "interrupted_by_user"
                    break
                except Exception as exc:
                    # 错误触发压缩：上下文溢出类错误（context_overflow /
                    # payload_too_large）标记 should_compress=True，压缩历史后
                    # 设置标志并跳出内层循环，由外层退款 + 重新组装重试
                    # （对齐原版 restart_with_compressed_messages 语义）
                    if (
                        _compressor is not None
                        and compression_attempts < _max_compression_attempts
                    ):
                        _classified = classify_api_error(
                            exc,
                            provider=getattr(agent, "provider", "") or "",
                            model=getattr(agent, "model", "") or "",
                            approx_tokens=getattr(_compressor, "last_prompt_tokens", 0)
                            or 0,
                            context_length=getattr(_compressor, "context_length", 0) or 0,
                            num_messages=len(messages) if messages else 0,
                        )
                        if _classified.should_compress:
                            compression_attempts += 1
                            # 与主动压缩统一走 compress_context：内部完成
                            # 摘要生成、失败回滚、SessionDB archive_and_compact
                            # 提交与 baseline 状态更新（对齐原版所有压缩路径
                            # 都经 _compress_context 的语义）。
                            from agent.conversation_compression import compress_context

                            _new_messages, _new_system_prompt = compress_context(
                                agent,
                                messages,
                                system_message,
                                approx_tokens=getattr(
                                    _compressor, "last_prompt_tokens", None
                                )
                                or None,
                            )
                            if len(_new_messages) < len(messages):
                                messages = _new_messages
                                active_system_prompt = _new_system_prompt
                                # 压缩后 baseline 同步（见上方主动压缩处）
                                conversation_history = conversation_history_after_compression(
                                    agent, messages
                                )
                                _restart_with_compressed_messages = True
                                if not getattr(agent, "quiet_mode", False):
                                    agent._safe_print(
                                        f"🧹 上下文超限，已压缩历史并重试 "
                                        f"（第 {compression_attempts}/{_max_compression_attempts} 次）"
                                    )
                                break
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
    
            # ⑧½ 错误触发压缩后的重试：压缩已更新 messages，退款本次调用
            #    （计数 + 预算），重新锚定当前用户消息下标，回退外层循环重新
            #    组装请求再调 API（对齐原版 :6002-6032）
            if _restart_with_compressed_messages:
                _restart_with_compressed_messages = False
                api_call_count -= 1
                agent._api_call_count = api_call_count
                agent.iteration_budget.refund()
                if current_turn_user_idx >= len(messages):
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            current_turn_user_idx = _i
                            break
                    else:
                        current_turn_user_idx = max(0, len(messages) - 1)
                continue
    
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
                _interim_msg = {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": tool_calls_payload,
                }
                messages.append(_interim_msg)
                # interim 中间旁白交付：模型在调工具时说的"让我先看看…"这类
                # 旁白实时推给 UI（interim_assistant_callback + 去重），
                # 对齐原版 conversation_loop.py:6279 的调用点。
                if getattr(agent, "interim_assistant_callback", None) is not None:
                    agent._emit_interim_assistant_message(_interim_msg)
    
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
                # 最终回答落库（finally 还会兜底一次，marker 去重）
                try:
                    agent._flush_messages_to_session_db(messages, conversation_history)
                except Exception:
                    agent._incremental_persistence_failed = True
                break
    finally:
        # 尽力 flush 到 SessionDB（所有退出路径：完成/中断/预算/异常）。
        # 持久化失败只标记 _incremental_persistence_failed，绝不影响
        # 模型主循环的结果与异常传播。
        try:
            agent._flush_messages_to_session_db(messages, conversation_history)
            # 供 close() 兜底 flush 的最新消息快照
            agent._session_messages = messages
        except Exception:
            agent._incremental_persistence_failed = True
        # API 调用计数统计（最小版，对齐原版 update_token_counts 的
        # api_call_count 语义）：每轮一次性把本轮的 API 调用次数累加写库。
        # 原版是每次 API 调用后 queue_token_counts 入队、后台线程写；
        # my-hermes 裁剪了 usage 记账体系，先以轮为单位补统计（后期完整
        # 移植 usage 后由 queue/update_token_counts 取代本段）。
        if api_call_count and getattr(agent, "_session_db", None) \
                and getattr(agent, "session_id", None):
            try:
                agent._session_db.add_session_call_count(
                    agent.session_id, api_call_count
                )
            except Exception:
                pass

    # 循环自然退出（没拿到最终回答、也没中断/失败）→ 只能是预算或
    # 迭代次数耗尽，归一化为 budget_exhausted（对应原版 finalize_turn）
    if final_response is None and not interrupted and not failed:
        _turn_exit_reason = "budget_exhausted"

    # ══════════════════════════════════════════════════════════════
    # 收尾：组装结果 dict（对应原版 finalize_turn 的简化替代）
    # ══════════════════════════════════════════════════════════════
    # 外部记忆 provider：回合完成 → sync_all（持久化对话）+ queue_prefetch_all
    # （预热下一轮召回）。中断回合跳过（对齐原版
    # run_agent._mirror_completed_turn_to_memory，#15218）。
    _mm = getattr(agent, "_memory_manager", None)
    if _mm and final_response and not interrupted and original_user_message:
        try:
            from agent.memory_provider import is_trivial_prompt

            _user_text = str(original_user_message)
            _mm.sync_all(
                _user_text,
                str(final_response),
                session_id=agent.session_id or "",
                messages=messages,
            )
            if not is_trivial_prompt(_user_text):
                _mm.queue_prefetch_all(
                    _user_text, session_id=agent.session_id or ""
                )
        except Exception:
            pass  # 外部记忆是 best-effort，失败不阻断回合收尾

    # ══════════════════════════════════════════════════════════════
    # 后台记忆/技能提炼（对齐原版 turn_finalizer.py:733-765）：
    #   回复交付后 spawn background review，绝不与用户任务竞争。
    #   memory 触发由 turn_context 计算（每 N 轮）；skill 触发按累计
    #   tool 迭代数（_iters_since_skill += 本轮迭代）达阈值。
    # ══════════════════════════════════════════════════════════════
    _should_review_skills = False
    if (
        getattr(agent, "_skill_nudge_interval", 0) > 0
        and "skill_manage" in getattr(agent, "valid_tool_names", set())
    ):
        agent._iters_since_skill = (
            getattr(agent, "_iters_since_skill", 0) + api_call_count
        )
        if agent._iters_since_skill >= agent._skill_nudge_interval:
            _should_review_skills = True
            agent._iters_since_skill = 0

    if (
        final_response
        and not interrupted
        and (_should_review_memory or _should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=_should_review_memory,
                review_skills=_should_review_skills,
            )
        except Exception:
            pass  # Background review is best-effort

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
        # 增量持久化附加信息：对话本身的 failed 语义不变，
        # 这里单独暴露 SessionDB 写入是否失败及原因
        "incremental_persistence_failed": agent._incremental_persistence_failed,
        "persistence_error": getattr(agent, "_last_persistence_error", None),
    }
