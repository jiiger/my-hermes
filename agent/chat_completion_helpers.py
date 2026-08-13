"""聊天补全辅助：可中断的 API 调用（对应原版 agent/chat_completion_helpers.py）。

当前实现：interruptible_api_call（非流式 + 中断 + stale 超时）与
interruptible_streaming_api_call（流式 + delta 回调 + 工具调用聚合）；
回退（try_activate_fallback）精简版：沿 _fallback_chain 切换 provider
（对应原版 chat_completion_helpers.py:1923）。
"""

import logging
import threading
import time
from typing import Any, Dict, Optional

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)


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

    # Responses API：事件流消费由 codex_runtime 管理（自带 worker + 中断轮询）
    if getattr(agent, "api_mode", "") == "codex_responses":
        from agent.codex_runtime import run_codex_stream

        return run_codex_stream(agent, api_kwargs, stale_timeout=stale_timeout)

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


# ── Fallback（回退）──────────────────────────────────────────────────────

# 限流冷却指数退避上限（原版 60s → 2m → 4m → … → 4h cap）
_RATE_LIMIT_BACKOFF_CAP_S = 14400
_FALLBACK_EXHAUSTED_COOLDOWN_S = 60


def try_activate_fallback(
    agent, reason: "FailoverReason | None" = None
) -> bool:
    """切换到回退链上的下一个 provider（对应原版 chat_completion_helpers.py:1923 精简版）。

    当前 provider 重试耗尽后调用。就地换掉 agent 的 OpenAI client、模型
    名与 provider，让外层重试循环用新后端继续。每次调用沿
    ``agent._fallback_chain`` 前进一格；链耗尽返回 False。

    精简版差异（相对原版）：
    - 只支持 OpenAI 兼容 chat_completions（无 Anthropic/Bedrock/MoA 分支、
      无 resolve_provider_client / credential_pool / BackendIdentity）；
    - 回退 client 用 my-hermes 的 create_openai_client 工厂构建；
    - 同后端跳过简化为 provider+model+base_url 字符串比较；
    - 无 _unavailable_fallback_keys 本会话禁用集合。
    """
    # 限流/计费类失败：给主 provider 记冷却（指数退避），供
    # restore_primary_runtime 判断"何时允许恢复主 provider"。
    if reason in {
        FailoverReason.rate_limit,
        FailoverReason.billing,
        FailoverReason.upstream_rate_limit,
    }:
        backoff_count = getattr(agent, "_rate_limit_backoff_count", 0)
        agent._rate_limit_backoff_count = backoff_count + 1
        backoff_seconds = min(
            60 * (2 ** backoff_count), _RATE_LIMIT_BACKOFF_CAP_S
        )
        agent._rate_limited_until = time.monotonic() + backoff_seconds
        logger.info(
            "Rate-limit backoff level %d: cooldown %d s",
            backoff_count, backoff_seconds,
        )

    chain = getattr(agent, "_fallback_chain", None) or []
    index = getattr(agent, "_fallback_index", 0)
    if index >= len(chain):
        # 链耗尽：非限流/计费类失败时也记一个短冷却，避免下一轮
        # restore_primary_runtime 立刻清零游标导致跨轮重放风暴（原版 #24996）。
        if len(chain) > 0 and reason not in {
            FailoverReason.rate_limit,
            FailoverReason.billing,
            FailoverReason.upstream_rate_limit,
        }:
            existing = getattr(agent, "_rate_limited_until", 0) or 0
            agent._rate_limited_until = max(
                existing, time.monotonic() + _FALLBACK_EXHAUSTED_COOLDOWN_S
            )
        return False

    fb = chain[index]
    agent._fallback_index = index + 1

    fb_provider = (fb.get("provider") or "").strip()
    fb_model = (fb.get("model") or "").strip()
    if not fb_provider or not fb_model:
        return try_activate_fallback(agent, reason)  # 跳过无效条目

    # 同后端跳过：链上条目与当前 provider/model/base_url 一致时回退等于
    # 原地失败循环，跳过继续（精简版字符串比较，原版用 BackendIdentity）。
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    current_base_url = (getattr(agent, "base_url", "") or "").strip().rstrip("/").lower()
    fb_base_url = (fb.get("base_url") or "").strip().rstrip("/")
    if (
        fb_provider.lower() == current_provider
        and fb_model == getattr(agent, "model", "")
        and fb_base_url.lower() == current_base_url
    ):
        return try_activate_fallback(agent, reason)

    # 首次切换：保存主运行时快照，供 restore_primary_runtime 恢复
    if not getattr(agent, "_primary_runtime", None):
        agent._primary_runtime = {
            "model": getattr(agent, "model", ""),
            "provider": getattr(agent, "provider", ""),
            "base_url": getattr(agent, "base_url", "") or "",
            "api_key": getattr(agent, "api_key", "") or "",
            "client_kwargs": dict(
                getattr(agent, "_client_kwargs", None) or {}
            ),
        }

    # 回退条目的 base_url / api_key 缺失时沿用主 provider 的（同原版
    # resolve_entry_api_key 的宽松语义）。
    fb_api_key = (fb.get("api_key") or "").strip() or (
        getattr(agent, "api_key", "") or ""
    )
    fb_kwargs = {"api_key": fb_api_key, "base_url": fb_base_url}

    try:
        from agent.agent_runtime_helpers import create_openai_client

        fb_client = create_openai_client(
            agent, dict(fb_kwargs), reason="fallback", shared=True
        )
    except Exception as e:  # pragma: no cover - 建 client 失败跳过该条目
        logger.warning(
            "Fallback client creation failed for %s/%s: %s",
            fb_provider, fb_model, e,
        )
        return try_activate_fallback(agent, reason)

    agent.model = fb_model
    agent.provider = fb_provider
    agent.base_url = fb_base_url
    agent.api_key = fb_api_key
    agent._client_kwargs = dict(fb_kwargs)
    agent.client = fb_client
    agent._fallback_activated = True
    logger.info("Fallback activated: %s/%s", fb_provider, fb_model)
    return True


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
    usage=None,
) -> Any:
    """把流式 chunk 聚合的文本/工具调用组装成响应对象。

    返回 SimpleNamespace，形状与非流式 ``response.choices[0].message`` 一致
    （content / tool_calls / finish_reason / usage），主循环后续逻辑无需区分来源。
    usage 来自流式末段 chunk（``stream_options={"include_usage": true}`` 时
    provider 会附带）；拿不到时为 None（调用方按无 usage 处理）。
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
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
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

    # Responses API：同样走 run_codex_stream（本身即流式事件消费）
    if getattr(agent, "api_mode", "") == "codex_responses":
        from agent.codex_runtime import run_codex_stream

        return run_codex_stream(
            agent,
            api_kwargs,
            on_first_delta=on_first_delta,
            stale_timeout=stale_timeout,
        )

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

            # 请求流式 usage（OpenAI 协议：stream_options.include_usage 让末段
            # chunk 携带 usage，否则流式响应永远没有用量，状态栏 ctx 恒为 0）。
            # 部分 provider 拒绝该参数（400）→ 回退不带，此时拿不到用量，
            # 上层按无 usage 处理（状态栏显示 "ctx --"）。
            def _open_stream():
                kwargs = dict(api_kwargs)
                wants_usage = "stream_options" not in kwargs
                if wants_usage:
                    kwargs["stream_options"] = {"include_usage": True}
                try:
                    # 仅主动请求 include_usage 时才收集 chunk.usage；调用方
                    # 显式传入 stream_options 时原样保留、不干预其语义
                    return client.chat.completions.create(**kwargs, stream=True), wants_usage
                except Exception as exc:
                    if wants_usage and getattr(exc, "status_code", None) == 400:
                        return (
                            client.chat.completions.create(**api_kwargs, stream=True),
                            False,
                        )
                    raise

            stream, collect_usage = _open_stream()

            # 聚合缓冲
            content_parts: list = []
            tool_calls_acc: list = []  # [{id, function: {name, arguments}}]，按 index 对齐
            finish_reason = None
            usage = None
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
                if collect_usage:
                    # include_usage 时末段 chunk 携带 usage；取最后一个非空值
                    _chunk_usage = getattr(chunk, "usage", None)
                    if _chunk_usage is not None:
                        usage = _chunk_usage
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
                content_parts, tool_calls_acc, finish_reason, usage=usage
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
