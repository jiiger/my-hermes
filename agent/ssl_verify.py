"""TLS verify resolution for httpx/OpenAI provider clients.
在默认情况下，Python 的网络库（如 httpx、requests）会使用内置的公共 CA 证书库（certifi）来验证服务器的身份。但在以下场景中，默认验证会直接失败并阻断连接：
企业内网/安全代理：公司使用了流量拦截代理（如 Zscaler、Palo Alto），或者内部部署的 LLM 服务使用了企业私有 CA 签发的证书。公共证书库不信任这些证书。
本地开发环境：在本地运行 Ollama、vLLM 或 llama.cpp 时，使用了自签名证书，或者根本没有配置有效的证书。
环境变量混乱：用户已经在系统里配置了 SSL_CERT_FILE 或 REQUESTS_CA_BUNDLE，但 OpenAI SDK 底层的 httpx 没有正确读取这些变量。
这个函数通过统一的逻辑，把配置、环境变量和默认行为整合在一起，确保网络请求能正确建立。"""

from __future__ import annotations

import logging
import os
import ssl
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _coerce_insecure(ssl_verify: Any) -> bool:
    if ssl_verify is False:
        return True
    if isinstance(ssl_verify, str) and ssl_verify.strip().lower() in {"false", "0", "no", "off"}:
        return True
    return False


def resolve_httpx_verify(
    *,
    ca_bundle: Optional[str] = None,
    ssl_verify: Any = None,
    base_url: str = "",
) -> bool | ssl.SSLContext:
    """Resolve httpx ``verify`` for provider HTTP clients.

    Priority:
    1. ``ssl_verify: false`` — disable verification (local dev only)
    2. explicit ``ca_bundle`` (per-provider ``ssl_ca_cert`` config field)
    3. ``HERMES_CA_BUNDLE``, ``SSL_CERT_FILE``, ``REQUESTS_CA_BUNDLE``,
       ``CURL_CA_BUNDLE`` env vars
    4. ``True`` (httpx/certifi default)

    ``base_url`` is used only for the insecure-mode warning message.
    """
    if _coerce_insecure(ssl_verify):
        logger.warning(
            "TLS certificate verification DISABLED (ssl_verify: false) for %s — "
            "this is intended for local development only and is unsafe on any "
            "network you do not fully control.",
            base_url or "a custom provider endpoint",
        )
        return False

    effective_ca = (
        (ca_bundle or "").strip()
        or os.getenv("HERMES_CA_BUNDLE", "").strip()
        or os.getenv("SSL_CERT_FILE", "").strip()
        or os.getenv("REQUESTS_CA_BUNDLE", "").strip()
        or os.getenv("CURL_CA_BUNDLE", "").strip()
    )
    if effective_ca:
        ca_path = str(Path(effective_ca).expanduser())
        if os.path.isfile(ca_path):
            return ssl.create_default_context(cafile=ca_path)
        logger.warning(
            "CA bundle path does not exist: %s — falling back to default certificates",
            effective_ca,
        )
    return True
