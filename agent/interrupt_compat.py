"""显式停止 agent 的兼容辅助函数。"""

from __future__ import annotations

import inspect
from typing import Any


def request_hard_interrupt(agent: Any, message: str | None = None) -> bool:
    """请求显式停止，回退到旧版 interrupt ABI。

    新式 agent 暴露 ``hard_interrupt(message=None)``；第三方 agent 和旧
    测试替身可能只暴露 ``interrupt(message=None)``——保留它们的可用性，
    不要发送它们不认识的新 ``hard_cancel=`` 关键字。仅在两种可调用对象
    都不存在时返回 ``False``。
    """
    # 避免把动态 ``__getattr__`` 代理（尤其是未 specced 的 ``MagicMock``
    # 或第三方 RPC 门面）误当成真正实现了新 ABI。静态查找能证明属性确实
    # 存在于实例或其类型上，之后再做正常的描述符绑定取回可调用对象。
    try:
        inspect.getattr_static(agent, "hard_interrupt")
    except AttributeError:
        interrupt = None
    else:
        interrupt = getattr(agent, "hard_interrupt", None)
    if not callable(interrupt):
        interrupt = getattr(agent, "interrupt", None)
    if not callable(interrupt):
        return False
    if message is None:
        interrupt()
    else:
        interrupt(message)
    return True
