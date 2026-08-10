"""子进程兼容性辅助（Windows 专用；WSL/Linux 上恒返回 0）。

原版 hermes_cli/_subprocess_compat.py 有大量 Windows 平台兼容逻辑，my-hermes
只保留被 hermes_constants 引用的 ``windows_hide_flags``。WSL 上该分支不触发，
按原版"非 Windows 返回 0"的简单实现即可。
"""

import sys

IS_WINDOWS = sys.platform == "win32"
_CREATE_NO_WINDOW = 0x08000000


def windows_hide_flags() -> int:
    """返回隐藏子进程控制台窗口的 Win32 creationflags；非 Windows 返回 0。

    对应原版 hermes_cli/_subprocess_compat.py:windows_hide_flags——短命控制台
    程序（taskkill、where、版本探测）派生时隐藏窗口但不脱离父进程的进程组。
    """
    if not IS_WINDOWS:
        return 0
    return _CREATE_NO_WINDOW
