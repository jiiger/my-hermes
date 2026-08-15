"""防符号链接的安全写盘工具（精简移植版）。

对应原版 hermes-agent 的 tools/spill_safety.py。my-hermes 只移植
:func:`write_text_exclusive` 及其直接依赖（``open_exclusive`` /
``ensure_spill_dir``），供 ``web_extract_tool._store_full_text`` 落盘使用。
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO

__all__ = [
    "ensure_spill_dir",
    "open_exclusive",
    "write_text_exclusive",
]


# O_NOFOLLOW 仅 POSIX 有；Windows 上省略无害——O_EXCL 本身已拒绝一切
# 已存在路径。
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def ensure_spill_dir(path: Path, *, private: bool = True) -> Path:
    """把 *path*（含父目录）建为目录，拒绝符号链接。

    ``private=True`` 时叶子目录以 0o700 创建，已存在的叶子收紧到 0o700。
    叶子存在但不是真实目录（如被种下符号链接）时抛 OSError。
    """
    path = Path(path)
    if private:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise OSError(f"spill dir is not a directory (symlink?): {path}")
    if private and stat.S_IMODE(st.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def open_exclusive(
    path: Path,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> IO[str]:
    """以独占创建方式打开 *path* 写；绝不跟随链接。

    ``overwrite=True`` 先 unlink 已存在路径（经 lstat 校验，只移除链接
    本身，真实目录会被拒绝），再独占创建——连覆盖路径也不可能被符号
    链接重定向。
    """
    path = Path(path)
    if overwrite:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(st.st_mode):
                raise OSError(f"refusing to overwrite a directory: {path}")
            os.unlink(path)
    mode = 0o600 if private else 0o666  # 非 private 遵从 umask
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, mode)
    try:
        return os.fdopen(fd, "w", encoding=encoding, errors=errors)
    except Exception:
        os.close(fd)
        raise


def write_text_exclusive(
    path: Path,
    text: str,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> None:
    """``Path.write_text`` 的等价物，但拒绝跟随符号链接。"""
    with open_exclusive(
        path, private=private, overwrite=overwrite,
        encoding=encoding, errors=errors,
    ) as fh:
        fh.write(text)
