import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def atomic_write_text(
    path,
    content: str,
    *,
    encoding: str = "utf-8",
    tmp_prefix: str = ".tmp_",
) -> None:
    """通过临时文件 + fsync + 原子重命名写入文本。

    对应原版 utils.py:139 atomic_write_text 的精简版：保留核心语义
    （进程崩溃/中断时目标文件绝不处于半写入状态），砍掉原版的
    preserve_mode / create_mode / symlink 处理等外围。

    供 memory 工具等"整文件重写"场景使用：读者要么看到旧的完整文件，
    要么看到新的完整文件，永远不会看到空文件或截断文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=tmp_prefix, suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Proxy Helpers ────────────────────────────────────────────────────────────


_PROXY_ENV_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """规范化代理 URL，以兼容 httpx/aiohttp。

    WSL/Clash 风格的环境通常会导出 SOCKS 代理，具体形式为
    ``socks://127.0.0.1:PORT``。httpx 会拒绝该别名，并期望
    请显式使用 ``socks5://`` 方案。
    """
    candidate = str(proxy_url or "").strip()
    if not candidate:
        return None
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://') :]}"
    return candidate


def normalize_proxy_env_vars() -> None:
    """将受支持的代理环境变量原地重写为标准URL格式."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized


# ─── URL Parsing Helpers ──────────────────────────────────────────────────────


def base_url_hostname(base_url: str) -> str:
    """返回基于url提供的小写主机名"""
    raw = (base_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    return (parsed.hostname or "").lower().rstrip(".")


def base_url_host_matches(base_url: str, domain: str) -> bool:
    """Return True when the base URL's hostname is ``domain`` or a subdomain.

    Safer counterpart to ``domain in base_url``, which is the substring
    false-positive class documented on ``base_url_hostname``. Accepts bare
    hosts, full URLs, and URLs with paths.

        base_url_host_matches("https://api.moonshot.ai/v1", "moonshot.ai") == True
        base_url_host_matches("https://moonshot.ai", "moonshot.ai")        == True
        base_url_host_matches("https://evil.com/moonshot.ai/v1", "moonshot.ai") == False
        base_url_host_matches("https://moonshot.ai.evil/v1", "moonshot.ai")     == False
    """
    hostname = base_url_hostname(base_url)
    if not hostname:
        return False
    domain = (domain or "").strip().lower().rstrip(".")
    if not domain:
        return False
    return hostname == domain or hostname.endswith("." + domain)
