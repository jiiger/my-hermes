"""工具执行器 —— 顺序 + 并发派发（精简移植版）。

对应原版 hermes-agent 的 agent/tool_executor.py（2338 行 → 精简版）。

原版 AIAgent 的两个方法（_execute_tool_calls_sequential /
_execute_tool_calls_concurrent）在这里作为接受父 AIAgent 作为首参的
模块级函数实现；run_agent.py:318 的 _execute_tool_calls 转发器调用它们。

精简版保留：
- execute_tool_calls_concurrent（原版 :686）与 execute_tool_calls_sequential
  （原版 :1531）两个入口；
- 私有辅助 _budget_for_agent（:78）/ _parse_tool_arguments（:115）/
  _max_workers_for_tool_batch（:203）/ _run_tool（:895）/
  _append_cancelled_tool_results（:1511）；
- 并发用 stdlib concurrent.futures.ThreadPoolExecutor，调度入口用
  tools/thread_context.py 的 propagate_context_to_thread 包装
  （my-hermes 版保留此行为）；
- 结果按原始调用顺序收集，以 role="tool" 消息追加；
- 工具调用走 agent._tool_impls[name](**args)（my-hermes 既有契约），
  结果 str() 兜底，未注册工具写错误串、异常 fail-open；
- 超阈值结果经 maybe_persist_tool_result 持久化，批末
  enforce_turn_budget 聚合（finalize）。

精简版改动：
- execute_tool_calls_segmented（原版 :2274）已移植为精简版：砍掉
  get_active_env / _incremental_persistence_failed /
  _apply_pending_steer_to_tool_results，execution_cwd 恒为 None；

砍掉（my-hermes 无对应系统，抄了直接 ImportError）：
- relay_tools / mcp_tool / environments.base / daemon_pool / tool_search /
  clarify_tool / memory_tool / read_preview_tool / read_terminal_tool /
  session_search / hermes_cli.middleware 等一切懒加载外部依赖；
- 并发超时（HERMES_CONCURRENT_TOOL_TIMEOUT_S）与 start-order gate
  （原版用于保持跨 worker 派发顺序；精简版只按 index 收集结果，
  不跨 worker 串行化派发）；
- spinner 分支：agent._should_emit_quiet_tool_messages() /
  _should_start_quiet_spinner() 不存在 → 跳过（原版 KawaiiSpinner 动画）。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from typing import Any, Optional

from agent.tool_dispatch_helpers import _plan_tool_batch_segments
from tools.budget_config import DEFAULT_BUDGET, BudgetConfig, budget_for_context_window
from tools.thread_context import propagate_context_to_thread
from tools.tool_result_storage import enforce_turn_budget, maybe_persist_tool_result

logger = logging.getLogger(__name__)

# 并发工具执行的 worker 线程数上限（对应原版 _MAX_TOOL_WORKERS）。
_MAX_TOOL_WORKERS = 8
# 原版图像生成并发上限；精简版无 hermes_cli.config，保留默认值。
_DEFAULT_IMAGE_PARALLEL_REQUESTS = 4


def _budget_for_agent(agent) -> BudgetConfig:
    """解析按 agent 上下文窗口缩放的 BudgetConfig（对应原版 :78）。

    大上下文模型保持历史 100K/200K 字符默认；小模型按窗口比例缩预算。
    my-hermes 的 agent 没有 context_compressor，恒回退默认预算。
    """
    try:
        ctx = getattr(getattr(agent, "context_compressor", None), "context_length", None)
        return budget_for_context_window(int(ctx)) if ctx else DEFAULT_BUDGET
    except Exception:
        return DEFAULT_BUDGET


def _parse_tool_arguments(raw_arguments: Any) -> tuple[dict, Optional[str]]:
    """解析模型发出的参数，不做修复或强制转换（对应原版 :115）。"""
    try:
        arguments = json.loads(raw_arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = None
    if isinstance(arguments, dict):
        return arguments, None
    return {}, json.dumps(
        {
            "error": "Invalid tool arguments",
            "message": "Tool arguments must be a valid JSON object; tool was not executed.",
        },
        ensure_ascii=False,
    )


def _max_workers_for_tool_batch(runnable_calls) -> int:
    """返回并发工具批的 worker 上限（对应原版 :203）。"""
    if not runnable_calls:
        return 0
    max_workers = _MAX_TOOL_WORKERS
    if any(call[2] == "image_generate" for call in runnable_calls):
        max_workers = min(max_workers, _DEFAULT_IMAGE_PARALLEL_REQUESTS)
    return min(len(runnable_calls), max_workers)


def _run_tool(
    agent,
    index: int,
    tool_call,
    function_name: str,
    function_args: dict,
    results: list,
    results_lock: threading.Lock,
) -> None:
    """并发 worker 函数：调用工具并把结果写入 results[index]。

    对应原版 _run_tool（:895）；精简版砍掉 middleware / guardrail /
    file-mutation 记录 / 活动心跳 / 终端 post-call 等钩子。
    异常 fail-open：工具未注册 → 错误串；抛异常 → 错误串。
    """
    start = time.time()
    tool_impls = getattr(agent, "_tool_impls", {})
    impl = tool_impls.get(function_name)
    if impl is None:
        result = f"错误: 未注册的工具 {function_name}"
        is_error = True
    else:
        try:
            if function_name == "memory":
                # 内置记忆工具需要 agent 持有的 MemoryStore（模型不会传
                # store 参数）。对齐原版 agent/tool_executor.py:1795 的
                # memory 分支——其余工具仍走通用 impl(**args) 契约。
                from tools.memory_tool import memory_tool as _memory_tool

                result = _memory_tool(
                    action=function_args.get("action"),
                    target=function_args.get("target", "memory"),
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    operations=function_args.get("operations"),
                    store=getattr(agent, "_memory_store", None),
                )
                # 内置 memory 写入成功后镜像给外部记忆 provider
                # （对齐原版 agent/tool_executor.py:1797）。
                _mm = getattr(agent, "_memory_manager", None)
                if _mm is not None:
                    try:
                        _mm.notify_memory_tool_write(result, function_args)
                    except Exception:
                        pass
            else:
                result = impl(**function_args)
        except KeyboardInterrupt:
            # Ctrl+C 优雅中断（对齐原版 agent/tool_executor.py:1073）：
            # 转成协作式中断请求，工具标记为已取消，不冒泡退出
            agent.interrupt("keyboard interrupt")
            result = "工具执行被用户中断 (keyboard interrupt)"
            is_error = True
        except Exception as exc:
            result = f"工具执行异常: {type(exc).__name__}: {exc}"
            is_error = True
        else:
            is_error = False
    duration = time.time() - start
    with results_lock:
        results[index] = (function_name, function_args, str(result), duration, is_error)


def _append_cancelled_tool_results(messages: list, tool_calls, *, reason: str) -> None:
    """为中断/取消的工具调用追加占位 tool 结果消息（对应原版 :1511）。"""
    for tc in tool_calls:
        name = tc.function.name
        # 保持 my-hermes 既有消息契约：{role, tool_call_id, name, content}
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": name,
            "content": f"[Tool execution cancelled — {name} was skipped due to {reason}]",
        })


def execute_tool_calls_concurrent(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    *,
    finalize: bool = True,
) -> None:
    """并发执行多个工具调用（stdlib ThreadPoolExecutor）。

    对应原版 :686。结果按原始调用顺序收集并追加进 messages，API 看到
    的顺序与模型发出的一致。

    ``finalize=False`` 跳过批末聚合预算执行 —— 原版用于分段调度器；
    分段接线后 execute_tool_calls_segmented 会以 False 传入各段，最终
    由整轮收尾统一聚合一次。
    """
    tool_calls = assistant_message.tool_calls
    num_tools = len(tool_calls)

    # 每轮解析一次上下文缩放的预算（便宜，避免循环内重复构建）
    _tool_budget = _budget_for_agent(agent)

    # ── 预检：中断 ──
    if getattr(agent, "_interrupt_requested", False):
        if not getattr(agent, "quiet_mode", True):
            print(f"{getattr(agent, 'log_prefix', '')}⚡ Interrupt: skipping {num_tools} tool call(s)")
        _append_cancelled_tool_results(messages, tool_calls, reason="user interrupt")
        return

    # ── 解析参数 ──
    parsed_calls = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )
        if malformed_args_result is not None:
            # my-hermes 契约：参数解析失败按空参执行（与原内联版一致）
            function_args = {}
        parsed_calls.append((tool_call, function_name, function_args))

    tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
    if not getattr(agent, "quiet_mode", True) and getattr(agent, "tool_progress_mode", "all") != "off":
        print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")

    # ── 并发执行 ──
    # TODO: 原版在此启动 KawaiiSpinner 动画（agent/tool_executor.py:1096，
    # 条件为 agent._should_emit_quiet_tool_messages() and
    # agent._should_start_quiet_spinner()）；my-hermes 的 AIAgent 没有这两个
    # 方法，spinner 未接入。agent/display.py 的 KawaiiSpinner 已移植可用，
    # 将来补上这两个方法后接线即可。
    results: list = [None] * num_tools
    runnable_calls = [
        (i, tc, name, args) for i, (tc, name, args) in enumerate(parsed_calls)
    ]
    if runnable_calls:
        max_workers = _max_workers_for_tool_batch(runnable_calls)
        results_lock = threading.Lock()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        try:
            futures = []
            for i, tc, name, args in runnable_calls:
                # 用 propagate_context_to_thread 把 agent 回合的 ContextVars
                # 传播进 worker 线程（保留原版 GHSA-qg5c-hvr5-hjgr 行为）。
                f = executor.submit(
                    propagate_context_to_thread(_run_tool),
                    agent, i, tc, name, args, results, results_lock,
                )
                futures.append(f)
            concurrent.futures.wait(futures)
        finally:
            executor.shutdown(wait=True)

    # ── 按原始顺序收集并追加 ──
    for i, (tc, name, args) in enumerate(parsed_calls):
        r = results[i]
        if r is None:
            function_result = f"Error executing tool '{name}': thread did not return a result"
            tool_duration = 0.0
        else:
            function_name, function_args, function_result, tool_duration, _is_error = r
            name, args = function_name, function_args
        # 超阈值持久化（env 恒 None → 本地写盘 / 内联截断）
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=name,
            tool_use_id=tc.id,
            config=_tool_budget,
        )
        # TODO: 改用 make_tool_result_message 富字段构造消息（原版
        # agent/tool_executor.py:1424），把 _tool_output_risk（由
        # tools.threat_patterns.scan_for_threats 生成）带进最终消息；
        # 当前为保 my-hermes 旧测试的消息格式（{role, tool_call_id, name,
        # content}）用了精简格式，风险标记未在运行链路消费。届时需同步
        # 更新 tests/test_agent_state.py 的精确断言。
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": name,
            "content": function_result,
        })

        if not getattr(agent, "quiet_mode", True) and getattr(agent, "tool_progress_mode", "all") != "off":
            if getattr(agent, "verbose_logging", False):
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
            else:
                preview = function_result[:200] if len(function_result) > 200 else function_result
                print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {preview}")

    # ── 批末聚合预算 ──
    if finalize and num_tools > 0:
        enforce_turn_budget(messages[-num_tools:], config=_tool_budget)


def execute_tool_calls_sequential(
    agent,
    assistant_message,
    messages: list,
    effective_task_id: str,
    api_call_count: int = 0,
    *,
    finalize: bool = True,
) -> None:
    """顺序执行工具调用（原版行为；单调用或交互工具使用）。

    对应原版 :1531。``finalize=False`` 跳过批末聚合预算执行 —— 分段
    调度器（execute_tool_calls_segmented）会以 False 传入各段，最终由
    整轮收尾统一聚合一次。
    """
    # 每轮解析一次上下文缩放的预算
    _tool_budget = _budget_for_agent(agent)
    tool_calls = assistant_message.tool_calls or []
    for i, tool_call in enumerate(tool_calls, 1):
        # 安全：每个工具开始前检查中断。用户在上一个工具执行期间发 "stop"，
        # 就不再启动任何工具 —— 立即全部跳过。
        if getattr(agent, "_interrupt_requested", False):
            remaining_calls = tool_calls[i - 1:]
            if remaining_calls:
                if not getattr(agent, "quiet_mode", True):
                    print(f"{getattr(agent, 'log_prefix', '')}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)")
                _append_cancelled_tool_results(
                    messages, remaining_calls, reason="user interrupt",
                )
            break

        function_name = tool_call.function.name
        function_args, malformed_args_result = _parse_tool_arguments(
            tool_call.function.arguments
        )
        if malformed_args_result is not None:
            # my-hermes 契约：参数解析失败按空参执行（与原内联版一致）
            function_args = {}

        tool_start_time = time.time()
        tool_impls = getattr(agent, "_tool_impls", {})
        impl = tool_impls.get(function_name)
        if impl is None:
            function_result = f"错误: 未注册的工具 {function_name}"
        else:
            try:
                function_result = impl(**function_args)
            except KeyboardInterrupt:
                # Ctrl+C 优雅中断（对齐原版 agent/tool_executor.py:2156）
                agent.interrupt("keyboard interrupt")
                function_result = "工具执行被用户中断 (keyboard interrupt)"
            except Exception as exc:
                function_result = f"工具执行异常: {type(exc).__name__}: {exc}"
        tool_duration = time.time() - tool_start_time

        function_result = str(function_result)
        # 超阈值持久化（env 恒 None → 本地写盘 / 内联截断）
        function_result = maybe_persist_tool_result(
            content=function_result,
            tool_name=function_name,
            tool_use_id=tool_call.id,
            config=_tool_budget,
        )
        # TODO: 改用 make_tool_result_message 富字段构造消息（原版
        # agent/tool_executor.py:2175），把 _tool_output_risk（由
        # tools.threat_patterns.scan_for_threats 生成）带进最终消息；
        # 当前为保 my-hermes 旧测试的消息格式（{role, tool_call_id, name,
        # content}）用了精简格式，风险标记未在运行链路消费。届时需同步
        # 更新 tests/test_agent_state.py 的精确断言。
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": function_result,
        })

        if not getattr(agent, "quiet_mode", True) and getattr(agent, "tool_progress_mode", "all") != "off":
            if getattr(agent, "verbose_logging", False):
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
            else:
                preview = function_result[:200] if len(function_result) > 200 else function_result
                print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {preview}")

    # ── 批末聚合预算 ──
    num_tools_seq = len(tool_calls)
    if finalize and num_tools_seq > 0:
        enforce_turn_budget(messages[-num_tools_seq:], config=_tool_budget)


def execute_tool_calls_segmented(agent, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0, segments=None) -> None:
    """把混合工具调用批按有序的并行/顺序段执行（原版 :2274 精简版）。

    ``segments`` 是 ``_plan_tool_batch_segments`` 生成的 ``(kind, calls)``
    规划：并行安全的极大连续运行走并发路径，barrier 调用走顺序路径，
    严格保持模型原始调用顺序。因为段是连续的，每条工具结果仍按发出顺序
    逐条追加，任何调用都不会早于前面的 barrier 结束 —— 与完全顺序执行
    拥有相同的顺序和副作用边界，同时在安全区间内恢复 I/O 并行。

    整轮收尾（聚合预算）在这里对整个批只做一次；各段执行器以
    ``finalize=False`` 运行，避免多段轮次重复聚合预算。

    中断语义：各段执行器开头都检查 ``agent._interrupt_requested`` 并为
    每条调用追加取消/跳过结果，因此第 k 段中断会排空 k+1..n 段而不
    执行它们，同时为每个 tool_call_id 保留一条结果。

    精简版改动（相对原版 :2274）：
    - my-hermes 没有 get_active_env：``segments is None`` 时以
      execution_cwd=None 调用规划器（planner 内部回退 Path.cwd()）；
    - my-hermes 没有 _incremental_persistence_failed 与
      _apply_pending_steer_to_tool_results：对应检查与 /steer 收尾调用
      直接去掉；
    - enforce_turn_budget 用 my-hermes 现有签名（无 env 参数，见 :48
      导入与 :257 调用）。
    """
    from types import SimpleNamespace

    if segments is None:
        # my-hermes 无 get_active_env，execution_cwd 恒为 None（planner
        # 内部回退 Path.cwd()）
        segments = _plan_tool_batch_segments(
            assistant_message.tool_calls, execution_cwd=None
        )

    for kind, calls in segments:
        segment_message = SimpleNamespace(tool_calls=list(calls))
        if kind == "parallel":
            execute_tool_calls_concurrent(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )
        else:
            execute_tool_calls_sequential(
                agent, segment_message, messages, effective_task_id, api_call_count,
                finalize=False,
            )

    # ── 整轮收尾（聚合预算，对整个批只做一次）──────────
    total_tools = len(assistant_message.tool_calls)
    if total_tools > 0:
        _tool_budget = _budget_for_agent(agent)
        enforce_turn_budget(messages[-total_tools:], config=_tool_budget)
