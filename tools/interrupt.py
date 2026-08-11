"""线程级中断信号（所有工具共享）。

提供按线程隔离的中断追踪，中断一个 agent 会话不会杀死其他会话里正在
运行的工具——这在同一进程内并发运行多个 agent 的网关场景下至关重要。

agent 在 run_conversation() 开始时记录自己的执行线程 ID，并通过
``set_interrupt()`` / ``clear_interrupt()`` 设置/清除信号；工具调用
``is_interrupted()`` 检查——它检查的是「当前线程」，无需传参。

工具内用法：
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# 可选调试追踪（配合 tools/environments/base.py 的 HERMES_DEBUG_INTERRUPT）：
# 记录每次 set/check 的调用方线程、目标线程与当前状态，便于排查
# "中断已发出但工具没感知"类问题。
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent 的 quiet_mode 路径会在 CLI 启动时把 tools 日志级别压到 ERROR；
    # 这里把本模块日志强制拉回 INFO，让追踪可见于 agent.log。
    logger.setLevel(logging.INFO)

# 已收到中断信号的线程 id 集合。
_interrupted_threads: set[int] = set()
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None) -> None:
    """为指定线程设置或清除中断信号。

    Args:
        active: True 表示发出中断信号，False 表示清除。
        thread_id: 目标线程 ident；为 None 时作用于当前线程
                   （CLI/测试的向后兼容）。
    """
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        if active:
            _interrupted_threads.add(tid)
        else:
            _interrupted_threads.discard(tid)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot,
        )


def is_interrupted() -> bool:
    """检查当前线程是否收到了中断请求。

    任何线程都可安全调用——每个线程只会看到自己的中断状态。
    """
    tid = threading.current_thread().ident
    with _lock:
        return tid in _interrupted_threads


def clear_current_thread_interrupt() -> None:
    """清除当前线程上的中断位。

    让一个刚获批准的命令在 spawn 子进程前拿到干净的中断状态，避免阻塞式
    审批等待期间落到本线程的陈旧中断位 SIGINT 掉刚批准的执行
    （exit 130 + "[Command interrupted]"）。单线程在该 tid 上的有序性
    保证了 DO-NOT-BREAK 不变量：本调用之后到达的「真实」中断会重新置位
    本线程的位，仍能被执行器的轮询循环观察到。直接调用本函数，不要经由
    _interrupt_event 代理（其 .clear() 绑定到实际执行它的线程）。
    """
    set_interrupt(False)  # thread_id=None -> 当前线程（见 set_interrupt）


# ---------------------------------------------------------------------------
# 向后兼容的 _interrupt_event 代理
# ---------------------------------------------------------------------------
# 一些历史调用点（code_execution_tool、process_registry、tests）直接导入
# _interrupt_event 并调用 .is_set() / .set() / .clear()。此 shim 把这些
# 调用映射到上面的按线程函数，让既有代码继续工作，同时底层机制已按线程隔离。

class _ThreadAwareEventProxy:
    """把 threading.Event 方法映射到按线程状态的即插即用代理。"""

    def is_set(self) -> bool:
        return is_interrupted()

    def set(self) -> None:  # noqa: A003
        set_interrupt(True)

    def clear(self) -> None:
        set_interrupt(False)

    def wait(self, timeout: float | None = None) -> bool:
        """并非真正支持——立即返回当前状态。"""
        return self.is_set()


_interrupt_event = _ThreadAwareEventProxy()
