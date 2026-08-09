"""把 agent 回合上下文传播到派发 Hermes 工具的工作线程（精简移植版）。

对应原版 hermes-agent 的 tools/thread_context.py（120 行）。

一个裸的 ``threading.Thread`` / ``ThreadPoolExecutor`` 工作线程以空的
``contextvars.Context`` 启动。线程内派发工具因此会静默丢失父线程的
ContextVars（例如会话/任务上下文）。

本文件保留原版的核心传播机制：copy_context + ctx.run。
砍掉了原版对 tools.terminal_tool 的 approval/sudo 回调捕获与安装
（原版 tools/terminal_tool.py:263-280 的 _get_approval_callback /
_get_sudo_password_callback / set_approval_callback /
set_sudo_password_callback）。my-hermes 精简版 terminal_tool 没有这
四个函数，直接 import 会 ImportError；且精简版没有审批交互，无需传播。

用法 —— 在**父线程**调用 :func:`propagate_context_to_thread`（调用时
快照父线程的 ContextVars），把返回的可调用对象作为工作线程的目标::

    t = threading.Thread(target=propagate_context_to_thread(loop_fn), args=(...))
    # 或
    executor.submit(propagate_context_to_thread(worker_fn), *args)
"""

from __future__ import annotations

import contextvars
from typing import Callable


def propagate_context_to_thread(target: Callable) -> Callable:
    """把 *target* 包装为在工作线程执行，并传播**当前**线程的 ContextVars。

    在父线程调用本函数；把返回的可调用对象作为线程/executor 目标。
    返回的可调用对象把位置参数与关键字参数转发给 *target* 并返回其结果。
    """
    ctx = contextvars.copy_context()

    def _runner(*args, **kwargs):
        def _inner():
            return target(*args, **kwargs)

        return ctx.run(_inner)

    return _runner
