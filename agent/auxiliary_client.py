from typing import Any, Dict, Optional, Tuple

from utils import normalize_proxy_env_vars
import os



def resolve_provider_client(provider: str,
    model: str = None,
    async_mode: bool = False,
    raw_codex: bool = False,
    explicit_base_url: str = None,
    explicit_api_key: str = None,
    api_mode: str = None,
    main_runtime: Optional[Dict[str, Any]] = None,
    is_vision: bool = False,
    task: Optional[str] = None,)->Tuple[Optional[Any], Optional[str]]:
    
    _validate_proxy_env_urls()

def _validate_base_url(base_url: str) -> None:
    """验证base_url"""
    from urllib.parse import urlparse

    candidate = str(base_url or "").strip()
    if not candidate or candidate.startswith("acp://"):
        return
    try:
        parsed = urlparse(candidate)
        if parsed.scheme in {"http", "https"}:
            _ = parsed.port              # raises ValueError for malformed ports
    except ValueError as exc:
        raise RuntimeError(
            f"Malformed custom endpoint URL: {candidate!r}. "
            "Run `hermes setup` or `hermes model` and enter a valid http(s) base URL."
        ) from exc
        
        
def _validate_proxy_env_urls() -> None:
    """当代理环境变量包含格式错误的 URL 时，快速抛出明确错误并终止运行。

    常见原因：shell 配置文件（例如 .zshrc）中存在类似以下内容的拼写错误
    ``export HTTP_PROXY=http://127.0.0.1:6153export NEXT_VAR=...``
    该操作会将“export”字符串拼接到端口号中。如果没有这一设置
    检查 OpenAI/httpx 客户端是否会抛出晦涩的“Invalid port”异常
    未指明具体问题环境变量的错误。
    """
    from urllib.parse import urlparse

    normalize_proxy_env_vars()

    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        value = str(os.environ.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = urlparse(value)
            if parsed.scheme:
                _ = parsed.port          # raises ValueError for e.g. '6153export'
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed proxy environment variable {key}={value!r}. "
                "Fix or unset your proxy settings and try again."
            ) from exc