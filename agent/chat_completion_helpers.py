"""聊天补全辅助：可中断的 API 调用（对应原版 agent/chat_completion_helpers.py）。

当前只实现第二步的 interruptible_api_call（非流式 + 中断 + stale 超时）；
流式（interruptible_streaming_api_call）、回退（try_activate_fallback）等
后续步骤按计划补进本文件。
"""

import threading
import time
from typing import Any, Dict, Optional


def _make_worker_client(agent) -> Any:
    """为后台工作线程创建独立的 OpenAI 客户端（默认工厂）。

    为什么不用共享的 agent.client：中断/stale 时主线程会 close() 请求
    客户端来打断阻塞中的 HTTP 调用；如果 close 的是共享 client，本轮
    后续请求全部报废。worker-local client 每次调用独立创建、用完即关，
    中断只影响本次请求。

    测试可以给 agent 挂实例属性 ``_make_worker_client`` 覆盖本工厂。
    """
    from agent.agent_runtime_helpers import create_openai_client

    client_kwargs: Dict[str, Any] = {
        "api_key": getattr(agent, "api_key", None) or "",
        "base_url": getattr(agent, "base_url", "") or "",
    }
    return create_openai_client(agent, client_kwargs, reason="worker", shared=False)


def _abort_worker_client(client_lock: threading.Lock, client_holder: dict) -> None:
    """关闭 worker 客户端连接，打断阻塞中的请求（失败静默）。"""
    with client_lock:
        client = client_holder.get("client")
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def interruptible_api_call(
    agent,
    api_kwargs: dict,
    *,
    stale_timeout: Optional[float] = None,
) -> Any:
    """非流式 API 调用，可在调用期间响应中断（对应原版 :663）。

    结构（精简自原版）：
    - 后台线程执行 ``chat.completions.create(**api_kwargs)``；
    - 主线程以 0.25s 间隔轮询：``agent._interrupt_requested`` 置位 →
      关闭 worker 连接让请求尽快失败 → 抛 InterruptedError；
    - stale 超时（默认取 agent._api_stale_timeout，180s）：请求太久没
      返回 → 同样关闭连接 → 抛 TimeoutError（防 read=None 的 keepalive
      client 永久挂起）。

    返回：原始响应对象（与直接调用 create 完全同构，后续逻辑不变）。
    """
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before API call")

    if stale_timeout is None:
        stale_timeout = getattr(agent, "_api_stale_timeout", 180.0)

    # worker 结果槽：后台线程写，主线程读（GIL 保证可见性足够）
    result = {"response": None, "error": None}
    # worker 客户端持有者：主线程中断时从这里取 client 并 close
    client_holder = {"client": None}
    client_lock = threading.Lock()

    def _worker() -> None:
        client = None
        try:
            factory = getattr(agent, "_make_worker_client", None)
            if factory is None:
                # 兜底工厂（非 AIAgent 实例的场景）：模块级工厂需要 agent 参数
                client = _make_worker_client(agent)
            else:
                # 实例工厂：AIAgent._make_worker_client() 或测试注入的 lambda
                client = factory()
            with client_lock:
                client_holder["client"] = client
            result["response"] = client.chat.completions.create(**api_kwargs)
        except Exception as exc:
            result["error"] = exc
        finally:
            # worker 自己收尾也 close（幂等：中断路径已 close 过）
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, daemon=True, name="api-call-worker")
    thread.start()

    # 主线程轮询：中断标志 + stale 超时
    start = time.monotonic()
    while thread.is_alive():
        thread.join(timeout=0.25)
        if agent._interrupt_requested:
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise InterruptedError("Agent interrupted during API call")
        if time.monotonic() - start > stale_timeout:
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise TimeoutError(
                f"API call did not respond within {stale_timeout:.0f}s"
            )

    # worker 已结束：有错误就抛，否则返回响应。
    # 注意：worker 可能在中断置位前后一瞬完成——再查一次中断标志，
    # 避免"中断被吞"的竞态（worker 正常结束但用户其实按了 Ctrl+C）
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted during API call")
    if result["error"] is not None:
        raise result["error"]
    return result["response"]


# ── 流式（第三步）──────────────────────────────────────────────────────────


def _fire_stream_delta(agent, text: str) -> None:
    """把流式文本增量分发给所有注册的回调（对应原版 run_agent._fire_stream_delta）。

    精简版不做原版的 think/context scrubber 清洗；回调异常静默吞掉，
    流式热路径不能被展示层拖垮。
    """
    if not isinstance(text, str) or not text:
        return
    for cb in (
        getattr(agent, "stream_delta_callback", None),
        getattr(agent, "_stream_callback", None),
    ):
        if cb is None:
            continue
        try:
            cb(text)
        except Exception:
            pass
    # 累积已流式文本：供终态去重（_current_streamed_assistant_text）
    agent._current_streamed_assistant_text = (
        getattr(agent, "_current_streamed_assistant_text", "") + text
    )


def _assemble_stream_response(
    content_parts: list,
    tool_calls_acc: list,
    finish_reason: Optional[str],
) -> Any:
    """把流式 chunk 聚合的文本/工具调用组装成响应对象。

    返回 SimpleNamespace，形状与非流式 ``response.choices[0].message`` 一致
    （content / tool_calls / finish_reason），主循环后续逻辑无需区分来源。
    """
    from types import SimpleNamespace

    tool_calls = None
    if tool_calls_acc:
        tool_calls = []
        for slot in tool_calls_acc:
            # 跳过只有占位没有实际内容的槽（流可能给出空 index）
            if slot["id"] is None and not slot["function"]["name"]:
                continue
            tool_calls.append(
                SimpleNamespace(
                    id=slot["id"] or "",
                    type="function",
                    function=SimpleNamespace(
                        name=slot["function"]["name"] or "",
                        arguments=slot["function"]["arguments"] or "",
                    ),
                )
            )

    message = SimpleNamespace(
        content="".join(content_parts) or None,
        tool_calls=tool_calls,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def interruptible_streaming_api_call(
    agent,
    api_kwargs: dict,
    *,
    on_first_delta: Optional[callable] = None,
    stale_timeout: Optional[float] = None,
) -> Any:
    """流式 API 调用，逐增量触发回调（对应原版 :2528 的 chat_completions 分支）。

    结构：
    - 后台线程 ``create(stream=True)`` 并逐 chunk 消费；
    - 文本 delta → ``_fire_stream_delta``（触发 stream_delta_callback /
      _stream_callback）；工具调用 delta → 按 index 聚合；
    - chunk 循环内检查 ``agent._interrupt_requested`` → 关闭流立即退出；
    - 主线程轮询中断/stale（同非流式）；
    - 返回聚合后的响应对象（形状与非流式一致，主循环无需感知流式）。

    Args:
        agent: AIAgent 实例。
        api_kwargs: 传给 create 的参数（内部会补 stream=True）。
        on_first_delta: 首个文本/工具 delta 到达时回调一次（如停掉转圈动画）。
        stale_timeout: 无响应判定超时（默认 agent._api_stale_timeout）。

    Returns:
        聚合响应对象（.choices[0].message / .finish_reason）。
    """
    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted before streaming API call")

    if stale_timeout is None:
        stale_timeout = getattr(agent, "_api_stale_timeout", 180.0)

    result = {"response": None, "error": None}
    client_holder = {"client": None}
    client_lock = threading.Lock()

    def _worker() -> None:
        client = None
        stream = None
        try:
            factory = getattr(agent, "_make_worker_client", None)
            if factory is None:
                client = _make_worker_client(agent)
            else:
                client = factory()
            with client_lock:
                client_holder["client"] = client

            stream = client.chat.completions.create(**api_kwargs, stream=True)

            # 聚合缓冲
            content_parts: list = []
            tool_calls_acc: list = []  # [{id, function: {name, arguments}}]，按 index 对齐
            finish_reason = None
            first_delta_fired = {"done": False}

            def _fire_first() -> None:
                if on_first_delta is not None and not first_delta_fired["done"]:
                    first_delta_fired["done"] = True
                    try:
                        on_first_delta()
                    except Exception:
                        pass

            for chunk in stream:
                # 中断：关闭流让连接立即释放（SSE 半读连接会泄漏到 httpx 连接池）
                if agent._interrupt_requested:
                    try:
                        stream.close()
                    except Exception:
                        pass
                    break
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue

                # 文本内容 → 回调（工具调用轮次抑制文本流式，避免
                # "我先用工具..."之类的碎话打断工具执行展示）
                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    if not tool_calls_acc:
                        _fire_first()
                        _fire_stream_delta(agent, content)

                # 工具调用增量 → 按 index 累加（arguments 是分片到达的 JSON）
                tc_deltas = getattr(delta, "tool_calls", None)
                if tc_deltas:
                    _fire_first()
                    for tc in tc_deltas:
                        idx = tc.index if getattr(tc, "index", None) is not None else 0
                        while len(tool_calls_acc) <= idx:
                            tool_calls_acc.append(
                                {"id": None, "function": {"name": None, "arguments": ""}}
                            )
                        slot = tool_calls_acc[idx]
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["function"]["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["function"]["arguments"] += fn.arguments

            result["response"] = _assemble_stream_response(
                content_parts, tool_calls_acc, finish_reason
            )
        except Exception as exc:
            result["error"] = exc
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_worker, daemon=True, name="stream-worker")
    thread.start()

    # 主线程轮询：中断标志 + stale 超时（同非流式）
    start = time.monotonic()
    while thread.is_alive():
        thread.join(timeout=0.25)
        if agent._interrupt_requested:
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise InterruptedError("Agent interrupted during streaming API call")
        if time.monotonic() - start > stale_timeout:
            _abort_worker_client(client_lock, client_holder)
            thread.join(timeout=5.0)
            raise TimeoutError(
                f"Streaming API call did not respond within {stale_timeout:.0f}s"
            )

    if agent._interrupt_requested:
        raise InterruptedError("Agent interrupted during streaming API call")
    if result["error"] is not None:
        raise result["error"]
    return result["response"]


# ── 回退（第四步）──────────────────────────────────────────────────────────


def _fallback_entry_key(fb: dict) -> tuple:
    """回退项的唯一键（provider, model, base_url），用于跳过不可用项。"""
    return (
        (fb.get("provider") or "").strip().lower(),
        (fb.get("model") or "").strip(),
        (fb.get("base_url") or "").strip(),
    )


def try_activate_fallback(agent, reason=None) -> bool:
    """切到回退链的下一个 provider（对应原版 chat_completion_helpers.py:1730）。

    重试耗尽后调用：前进 _fallback_index，把 provider/model/base_url/api_key
    换成回退项的值——worker 工厂 _make_worker_client 每次按这些属性新建
    客户端，所以切换对后续请求立即生效，无需动共享 client。

    - 首次回退时记录主运行时快照（_primary_runtime），供 restore_primary_runtime 恢复；
    - 限流类原因（rate_limit/upstream_rate_limit）启动 60s 冷却：冷却期内
      恢复主 provider 会被跳过，防止刚切走又切回去立刻再被 429；
    - 链耗尽或所有回退项无效 → 返回 False。

    Args:
        agent: AIAgent 实例。
        reason: FailoverReason（可选，用于限流冷却）。

    Returns:
        True=已切到下一个回退 provider；False=链耗尽/无可用回退。
    """
    chain = getattr(agent, "_fallback_chain", None) or []
    index = getattr(agent, "_fallback_index", 0)
    if index >= len(chain):
        return False

    # 限流冷却：第一次 429 起 60s 内不恢复主 provider
    if reason is not None and reason.value in ("rate_limit", "upstream_rate_limit"):
        agent._rate_limited_until = time.monotonic() + 60.0

    # 首次回退：记录主 provider 快照（restore_primary_runtime 据此恢复）
    if not getattr(agent, "_fallback_activated", False):
        agent._primary_runtime = {
            "provider": getattr(agent, "provider", "") or "",
            "model": getattr(agent, "model", "") or "",
            "base_url": getattr(agent, "base_url", "") or "",
            "api_key": getattr(agent, "api_key", "") or "",
        }
        agent._fallback_activated = True

    unavailable = getattr(agent, "_unavailable_fallback_keys", None) or set()
    while index < len(chain):
        fb = chain[index]
        agent._fallback_index = index + 1
        # 跳过无效项（缺 model）与已标记不可用的项
        if not (fb.get("model") or "").strip():
            index = agent._fallback_index
            continue
        if _fallback_entry_key(fb) in unavailable:
            index = agent._fallback_index
            continue

        # 应用回退配置：worker 工厂读这些属性建 client
        agent.model = fb["model"].strip()
        if fb.get("provider"):
            agent.provider = fb["provider"].strip().lower()
        if fb.get("base_url"):
            agent.base_url = fb["base_url"].strip()
        if fb.get("api_key"):
            agent.api_key = fb["api_key"].strip()
        return True
    return False


def restore_primary_runtime(agent) -> bool:
    """恢复主 provider 运行时（每轮开始调用，对应原版 :6451）。

    上一轮若发生过回退（_fallback_activated=True），把 provider/model/
    base_url/api_key 恢复为 _primary_runtime 快照，重置回退游标，下一轮
    从主 provider 重新开始。

    限流冷却期内（_rate_limited_until 未到）跳过恢复——保持当前回退
    provider，避免刚切走又切回主 provider 立刻再 429。
    """
    if not getattr(agent, "_fallback_activated", False):
        return True
    if time.monotonic() < getattr(agent, "_rate_limited_until", 0.0):
        return False
    primary = getattr(agent, "_primary_runtime", None) or {}
    if primary.get("provider") is not None:
        agent.provider = primary["provider"]
    if primary.get("model"):
        agent.model = primary["model"]
    if primary.get("base_url"):
        agent.base_url = primary["base_url"]
    if primary.get("api_key"):
        agent.api_key = primary["api_key"]
    agent._fallback_index = 0
    agent._fallback_activated = False
    return True
