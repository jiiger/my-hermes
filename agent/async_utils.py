"""异步/同步桥接辅助函数（精简移植版）。

对应原版 hermes-agent 的 agent/async_utils.py（84 行）。代码库里有约
30 处需要从工作线程通过 :func:`asyncio.run_coroutine_threadsafe` 把协程
调度到事件循环上。该函数可能抛 :class:`RuntimeError`（例如关闭竞态期间
循环已被关闭），一旦抛出，协程对象就永远不会被 await、也永远不会被
close——触发 ``"coroutine '<name>' was never awaited"`` 的
RuntimeWarning，并泄漏协程帧直到 GC。

:func:`safe_schedule_threadsafe` 包装了该调用：调度失败时关闭协程，
返回 ``None``（而不是半成形的 future），调用方可以干净分支：

    fut = safe_schedule_threadsafe(coro, loop)
    if fut is None:
        return  # 或回退行为
    fut.result(timeout=5)

该辅助函数刻意**不**处理 ``future.result()`` 失败——那是另一回事。
一旦循环接受了协程，它的生命周期就属于循环，不再属于调度线程。
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Any, Coroutine, Optional


_DEFAULT_LOGGER = logging.getLogger(__name__)


def safe_schedule_threadsafe(
    coro: Coroutine[Any, Any, Any],
    loop: Optional[asyncio.AbstractEventLoop],
    *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop",
    log_level: int = logging.DEBUG,
) -> Optional[Future]:
    """从同步上下文把 ``coro`` 调度到 ``loop`` 上，泄漏安全。

    成功时返回 :class:`concurrent.futures.Future`；循环缺失或
    :func:`asyncio.run_coroutine_threadsafe` 抛异常（例如关闭竞态期间
    循环已关闭）时返回 ``None``。所有失败路径都会 :meth:`close` 协程，
    因此不会触发 ``"coroutine was never awaited"`` 警告，也不会泄漏
    协程帧。

    调用方保留对返回 future 的完全控制权（``.result(timeout=...)``、
    挂 ``add_done_callback``、或 fire-and-forget 忽略它）。
    """
    log = logger if logger is not None else _DEFAULT_LOGGER

    if loop is None:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: loop is None", log_message)
        return None

    try:
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None


def consume_detached_task_result(task: "asyncio.Future[Any]") -> None:
    """取走已脱离任务的返回值，而不暴露取消。

    用作被取消并脱离（detached）任务的 ``add_done_callback``（例如
    超过拆除期限后吞掉 ``CancelledError`` 的适配器关闭路径）。观察
    ``task.exception()`` 可防止事件循环上出现 "exception was never
    retrieved" 噪音；取消与任何终止性错误都被刻意吞掉——任务的所有者
    早已放弃它。
    """
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
