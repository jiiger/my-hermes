"""Skill 写来源溯源（精简移植自原版 tools/skill_provenance.py）。

ContextVar 记录当前线程的 skill 写来源：默认 ``user``（前台/用户要求），
background_review fork 线程内设为 ``background_review``（agent 自主沉淀）。
curator 只整理后者——用户写的 skill 绝不被自动归档。
"""

import contextvars
from typing import Optional

from tools.skill_usage import CURATOR_ORIGIN, USER_ORIGIN

_write_origin: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "skill_write_origin", default=USER_ORIGIN
)


def set_write_origin(origin: Optional[str]) -> None:
    """设置当前线程的 skill 写来源（fork 线程用）。"""
    _write_origin.set(origin or USER_ORIGIN)


def get_write_origin() -> str:
    return _write_origin.get() or USER_ORIGIN


def is_curator_managed() -> bool:
    """当前线程的写是否属于 curator 可管理（agent 自主沉淀）。"""
    return get_write_origin() == CURATOR_ORIGIN
