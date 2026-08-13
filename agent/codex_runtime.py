"""Responses API 流式调用（精简版，对应原版 agent/codex_runtime.py）。

用 ``responses.create(stream=True)`` 的 SSE 事件流聚合最终响应，再经
responses_adapter._normalize_codex_response 归一化成 chat 兼容对象
（choices[0].message + usage），主循环/工具循环/fallback 无需区分来源。

精简差异（相对原版 codex_runtime.py）：
- 不移植 relay_llm 观测、单写者围栏、断流续跑（重试）、commentary 展示；
- 推理 delta 不推送展示（my-hermes 无推理显示通道），但完整 reasoning
  item 仍经 output_item.done 收集，交给归一化保留；
- 中断模式对齐 chat_completion_helpers：后台线程消费事件流，主线程
  轮询中断/stale 并关闭 client。
"""

import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional


# ── 事件字段读取（兼容 SDK 对象与 dict 两种形态）──────────────────────


def _event_field(event: Any, name: str, default: Any = None) -> Any:
    """从事件取字段：先取属性，再回退 dict（原版 codex_runtime._event_field）。"""
    value = getattr(event, name, None)
    if value is None and isinstance(event, dict):
        value = event.get(name, default)
    return value if value is not None else default


def _item_field(item: Any, name: str, default: Any = None) -> Any:
    """从输出项取字段（SDK 对象或 dict）。"""
    value = getattr(item, name, None)
    if value is None and isinstance(item, dict):
        value = item.get(name, default)
    return value if value is not None else default


# ── 事件流消费（聚合最终响应）────────────────────────────────────────


_TERMINAL_EVENT_TYPES = {
    "response.completed",
    "response.incomplete",
    "response.failed",
}


def _consume_codex_event_stream(
    event_iter: Any,
    *,
    model: str,
    on_text_delta: Optional[Callable[[str], None]] = None,
    on_first_delta: Optional[Callable[[], None]] = None,
    interrupt_check: Optional[Callable[[], bool]] = None,
) -> SimpleNamespace:
    """消费 Responses SSE 事件流并聚合出最终响应（原版 976 行的精简版）。

    返回 SimpleNamespace：output / output_text / usage / status / id / model。
    - 文本 delta 聚合进 output_text，且经 on_text_delta 推送（有工具调用时
      抑制推送，避免工具轮碎话打断展示）；
    - commentary/analysis 阶段文本不进 content（走 reasoning 通道，本版
      不推送）；
    - output_item.done 收集完整输出项（含 reasoning/function_call）；
    - 终态事件（completed/incomplete/failed）记录 usage/status/id 后结束。
    """
    collected_output_items: List[Any] = []
    collected_text_deltas: List[str] = []
    has_tool_calls = False
    first_delta_fired = False
    active_message_phase: Optional[str] = None
    terminal_status = "completed"
    terminal_usage: Any = None
    terminal_response_id: Optional[str] = None
    saw_terminal = False
    saw_interrupt = False

    for event in event_iter:
        if interrupt_check is not None and interrupt_check():
            saw_interrupt = True
            break

        event_type = _event_field(event, "type", "")
        if not isinstance(event_type, str):
            event_type = ""

        # error 帧：把 provider 真实失败原因抛给上层
        if event_type == "error":
            err = _event_field(event, "error") or _event_field(event, "message") or "unknown error"
            raise RuntimeError(f"Responses stream error: {err}")

        if event_type == "response.output_item.added":
            item = _event_field(event, "item")
            item_type = _item_field(item, "type", "")
            if item_type == "message":
                phase = _item_field(item, "phase", None)
                active_message_phase = phase.strip().lower() if isinstance(phase, str) else None
            else:
                active_message_phase = None
            if "function_call" in str(item_type):
                has_tool_calls = True
            continue

        if "output_text.delta" in event_type:
            delta_text = _event_field(event, "delta", "")
            if not delta_text:
                continue
            # commentary/analysis 是回合中的叙述，不是最终答案：不进 content
            if active_message_phase in {"commentary", "analysis"}:
                continue
            collected_text_deltas.append(delta_text)
            if not has_tool_calls:
                if not first_delta_fired:
                    first_delta_fired = True
                    if on_first_delta is not None:
                        try:
                            on_first_delta()
                        except Exception:
                            pass
                if on_text_delta is not None:
                    try:
                        on_text_delta(delta_text)
                    except Exception:
                        pass
            continue

        # 工具调用增量：只标记有工具，具体项在 output_item.done 收集
        if "function_call" in event_type:
            has_tool_calls = True
            continue

        # 推理增量：不推送展示（my-hermes 无推理通道）；完整 item 由 done 收集
        if "reasoning" in event_type and "delta" in event_type:
            continue

        if event_type == "response.output_item.done":
            done_item = _event_field(event, "item")
            if done_item is not None:
                collected_output_items.append(done_item)
            continue

        if event_type in _TERMINAL_EVENT_TYPES:
            saw_terminal = True
            resp_obj = _event_field(event, "response")
            if resp_obj is not None:
                terminal_usage = _item_field(resp_obj, "usage", terminal_usage)
                rid = _item_field(resp_obj, "id", None)
                if rid is not None:
                    terminal_response_id = rid
                rstatus = _item_field(resp_obj, "status", None)
                if isinstance(rstatus, str):
                    terminal_status = rstatus
            if event_type == "response.incomplete":
                terminal_status = terminal_status or "incomplete"
            elif event_type == "response.failed":
                terminal_status = terminal_status or "failed"
            break

    # 合成 output：优先用 output_item.done 收集的项；纯文本流（无工具）
    # 且没有输出项时，用聚合的 delta 合成 message 项。
    if collected_output_items:
        output = list(collected_output_items)
    elif collected_text_deltas and not has_tool_calls:
        assembled = "".join(collected_text_deltas)
        output = [SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=assembled)],
        )]
    else:
        output = []

    if not saw_terminal and not output and not saw_interrupt:
        raise RuntimeError("Responses stream did not emit a terminal response")

    return SimpleNamespace(
        output=output,
        output_text="".join(collected_text_deltas),
        usage=terminal_usage,
        status=terminal_status,
        id=terminal_response_id,
        model=model,
    )


# ── usage 映射（Responses input/output tokens → chat prompt/completion）──


def _map_usage(usage: Any) -> Optional[SimpleNamespace]:
    """把 Responses usage 映射成 chat 兼容的 prompt/completion tokens。"""
    if usage is None:
        return None

    def _tokens(in_key: str, out_key: str) -> int:
        v = _item_field(usage, in_key, None)
        if v is None:
            v = _item_field(usage, out_key, None)
        return int(v) if v is not None else 0

    prompt = _tokens("input_tokens", "prompt_tokens")
    completion = _tokens("output_tokens", "completion_tokens")
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


# ── 主入口：流式执行一次 Responses 请求，返回 chat 兼容响应对象 ──────


def run_codex_stream(
    agent,
    api_kwargs: dict,
    *,
    on_first_delta: Optional[Callable[[], None]] = None,
    stale_timeout: Optional[float] = None,
) -> Any:
    """执行一次 Responses 流式请求并返回 chat 兼容响应对象。

    结构对齐 chat_completion_helpers 的中断模式：后台线程消费 SSE 事件流，
    主线程轮询中断/stale 并关闭 client。返回对象形状与非流式
    ``response.choices[0].message`` 一致（content / tool_calls /
    finish_reason / usage），主循环无需区分来源。
    """
    if getattr(agent, "_interrupt_requested", False):
        raise InterruptedError("Agent interrupted before Codex stream")

    if stale_timeout is None:
        stale_timeout = getattr(agent, "_api_stale_timeout", 180.0)

    result: Dict[str, Any] = {"response": None, "error": None}
    client_holder = {"client": None}
    client_lock = threading.Lock()

    def _worker() -> None:
        client = None
        event_iter = None
        try:
            # ① 建 worker client（与 chat_completions 中断模式一致）
            factory = getattr(agent, "_make_worker_client", None)
            if factory is None:
                from agent.chat_completion_helpers import _make_worker_client as _mwc

                client = _mwc(agent)
            else:
                client = factory()
            with client_lock:
                client_holder["client"] = client

            # ② 参数转换：chat messages/tools → Responses input/tools
            from agent.responses_adapter import (
                _chat_messages_to_responses_input,
                _normalize_codex_response,
                _responses_tools,
            )

            r_kwargs = dict(api_kwargs)
            r_kwargs["input"] = _chat_messages_to_responses_input(
                api_kwargs.get("messages")
            )
            r_kwargs["tools"] = _responses_tools(api_kwargs.get("tools"))
            r_kwargs.pop("messages", None)
            r_kwargs["stream"] = True

            # ③ 打开事件流并消费聚合
            event_iter = client.responses.create(**r_kwargs)

            from agent.chat_completion_helpers import _fire_stream_delta

            final = _consume_codex_event_stream(
                event_iter,
                model=api_kwargs.get("model", ""),
                on_text_delta=lambda t: _fire_stream_delta(agent, t),
                on_first_delta=on_first_delta,
                interrupt_check=lambda: bool(
                    getattr(agent, "_interrupt_requested", False)
                ),
            )

            # ④ 归一化成 chat 兼容对象
            assistant_message, finish_reason = _normalize_codex_response(final)
            result["response"] = SimpleNamespace(
                id=final.id,
                model=final.model,
                choices=[SimpleNamespace(
                    message=assistant_message,
                    finish_reason=finish_reason,
                )],
                usage=_map_usage(final.usage),
            )
        except Exception as exc:
            result["error"] = exc
        finally:
            if event_iter is not None:
                close = getattr(event_iter, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, daemon=True, name="codex-stream-worker")
    thread.start()

    # 主线程轮询：中断标志 + stale 超时（同 chat_completions 流式）
    from agent.chat_completion_helpers import _abort_worker_client

    start = time.monotonic()
    while thread.is_alive():
        thread.join(timeout=0.25)
        if getattr(agent, "_interrupt_requested", False):
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise InterruptedError("Agent interrupted during Codex stream")
        if time.monotonic() - start > stale_timeout:
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise TimeoutError(
                f"Responses stream did not respond within {stale_timeout:.0f}s"
            )

    if getattr(agent, "_interrupt_requested", False):
        raise InterruptedError("Agent interrupted during Codex stream")
    if result["error"] is not None:
        raise result["error"]
    return result["response"]
