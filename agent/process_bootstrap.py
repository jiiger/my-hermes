"""进程级引导：OpenAI 懒加载代理 + 安全 stdio + 代理 URL 解析"""

import os
import sys
from typing import Any, Optional
from urllib.request import proxy_bypass_environment

from utils import base_url_hostname, normalize_proxy_url

# 模块级缓存：整个进程只付出一次 openai SDK import 成本（首次懒加载后）
_OPENAI_CLS_CACHE = None


def _load_openai_cls() -> type:
    """导入并缓存 ``openai.OpenAI``。"""
    global _OPENAI_CLS_CACHE
    if _OPENAI_CLS_CACHE is None:
        from openai import OpenAI as _cls

        _OPENAI_CLS_CACHE = _cls
    return _OPENAI_CLS_CACHE


class _OpenAIProxy:
    """长得像 ``openai.OpenAI`` 的模块级代理，但按需懒加载。

    保住 isinstance(client, OpenAI) 判断和 patch("run_agent.OpenAI") 测试。
    """

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return _load_openai_cls()(*args, **kwargs)

    def __instancecheck__(self, obj):
        return isinstance(obj, _load_openai_cls())

    def __repr__(self):
        return "<lazy openai.OpenAI proxy>"


class _SafeWriter:
    """透明 stdio 包装：捕获断管导致的 OSError/ValueError，防止崩溃。

    在 systemd 服务 / Docker / 无头守护进程场景下，stdout/stderr 管道
    可能不可用（空闲超时、缓冲区耗尽、socket 重置），任何 print() 都会
    抛 OSError 直接炸掉 agent 初始化。包装后静默吞掉这类错误。
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _get_proxy_from_env() -> Optional[str]:
    """从环境变量读取代理 URL（HTTPS_PROXY → HTTP_PROXY → ALL_PROXY）。"""
    for key in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            return normalize_proxy_url(value)
    return None


def _get_proxy_for_base_url(base_url: Optional[str]) -> Optional[str]:
    """返回环境配置的代理；若 NO_PROXY 排除了该 base_url 则返回 None。"""
    proxy = _get_proxy_from_env()
    if not proxy or not base_url:
        return proxy

    host = base_url_hostname(base_url)
    if not host:
        return proxy

    try:
        if proxy_bypass_environment(host):
            return None
    except Exception:
        pass

    return proxy


def _install_safe_stdio() -> None:
    """用 _SafeWriter 包裹 stdout/stderr，让控制台输出不会炸掉 agent。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


# 模块级代理实例 —— 顶替 ``openai.OpenAI``。run_agent 再导出它。
OpenAI = _OpenAIProxy()
