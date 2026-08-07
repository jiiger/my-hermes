"""每个 agent 的迭代预算：线程安全的消耗/退还计数器。

对应原版 agent/iteration_budget.py（run_agent 会重新导出 IterationBudget，
保持 ``from run_agent import IterationBudget`` 的写法可用）。
"""

from __future__ import annotations

import threading


class IterationBudget:
    """线程安全的迭代计数器。

    主循环条件之一：``agent.iteration_budget.remaining > 0``。
    - consume() : 尝试消耗一次迭代（预算耗尽返回 False）；
    - refund()  : 退还一次迭代；
    - remaining : 剩余可用迭代次数（永不小于 0）。
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """尝试消耗一次迭代。预算已用完返回 False。"""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """退还一次迭代（例如程序化工具调用不占预算的场景）。"""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


__all__ = ["IterationBudget"]
