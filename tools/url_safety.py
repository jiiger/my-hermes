"""URL 安全校验（精简移植版）。

对应原版 hermes-agent 的 tools/url_safety.py（874 行）。my-hermes 只移植
``web_extract_tool`` 需要的三个函数及其直接依赖：

- :func:`normalize_url_for_request`：URL 归一化（ASCII-safe）；
- :func:`sensitive_query_param_name`：敏感 query 参数检测；
- :func:`async_is_safe_url`：SSRF 防护（DNS 解析走 asyncio.to_thread）。

原版的连接期 SSRF 客户端（create_ssrf_safe_client / _SSRFGuardedAsyncNetworkBackend）、
``is_always_blocked_url``、``_resolved_http_connect_ips`` 等不搬。
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from typing import Any, Optional
from urllib.parse import parse_qsl, quote, unquote, urlparse, urlsplit, urlunsplit

from hermes_constants import get_hermes_home_override
from utils import is_truthy_value

logger = logging.getLogger(__name__)


# ── 代理检测 ──────────────────────────────────────────────────────────
# 命中代理环境变量时，把 DNS 交给代理解析（沙箱/代理环境可能禁直连 DNS）。
_PROXY_ENV_VARS = (
    "HTTPS_PROXY", "https_proxy",
    "HTTP_PROXY", "http_proxy",
    "ALL_PROXY", "all_proxy",
)


def _proxy_is_configured() -> bool:
    """返回是否设置了至少一个 HTTP 代理环境变量。"""
    return any(os.environ.get(v) for v in _PROXY_ENV_VARS)


def normalize_url_for_request(url: str) -> str:
    """返回供 Hermes URL 工具使用的 ASCII-safe HTTP URL。

    用户/模型常给出 IRI（如 ``https://wttr.in/Köln``）。保留 URL 语法与
    已有百分号转义，同时把非 ASCII 的 host/path/query/fragment 编码为
    ASCII。仅供 URL 工具入参使用，不得用于改写任意 shell 命令。
    """
    if not isinstance(url, str):
        return url

    raw = url.strip()
    if not raw:
        return raw

    # 模型偶尔会在 scheme 分隔符与 authority 之间塞空白
    # （``https:// docs.example``）——该位置在 HTTP(S) URL 里永远无意义，
    # 先修复再解析，避免 web 工具被格式瑕疵卡住。
    raw = re.sub(r"^([A-Za-z][A-Za-z0-9+.-]*://)\s+", r"\1", raw)

    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw

    if parsed.scheme.lower() not in {"http", "https"}:
        return raw

    netloc = parsed.netloc
    hostname = parsed.hostname
    if hostname:
        try:
            ascii_host = hostname.encode("idna").decode("ascii")
        except UnicodeError:
            ascii_host = hostname
        if ascii_host != hostname:
            netloc = netloc.replace(hostname, ascii_host, 1)

    path = quote(parsed.path, safe="/%:@!$&'()*+,;=")
    query = quote(parsed.query, safe="/%:@!$&'()*+,;=?")
    fragment = quote(parsed.fragment, safe="/%:@!$&'()*+,;=?")

    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


# 明确的凭据型 query 参数名。刻意收窄：普通英文词（code/key/auth/session/sig）
# 可能同时是普通页面 facet，故意排除以免拦截正常浏览。前缀式 token 脱敏
# 仍能抓已知厂商 key 形状；本集合是兜底，抓不带厂商前缀的 opaque secret。
_SENSITIVE_QUERY_PARAM_NAMES = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "awsaccesskeyid",
    "client_secret",
    "credential",
    "credentials",
    "jwt",
    "password",
    "passwd",
    "secret",
    "session_id",
    "signature",
    "token",
    "x_amz_security_token",
    "x_amz_signature",
    "x-amz-security-token",
    "x-amz-signature",
})


def sensitive_query_param_name(url: str) -> Optional[str]:
    """返回 ``url`` 里第一个敏感 query 参数名，没有则返回 None。

    在把 URL 交给第三方抓取/浏览器后端前使用。前缀式 token 脱敏抓已知
    凭据形状；本函数抓不带厂商前缀的 opaque magic link、OAuth code、
    签名 URL 参数、自定义 ``?token=...``。
    """
    if not isinstance(url, str) or "?" not in url:
        return None
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.query:
        return None
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if value and unquote(key).lower() in _SENSITIVE_QUERY_PARAM_NAMES:
            return key
    return None


# 无论 IP 解析如何都必须拦截的 hostname——云元数据端点，攻击者可借此
# 窃取实例凭据。
_BLOCKED_HOSTNAMES = frozenset({
    "metadata.google.internal",
    "metadata.goog",
})

# 无论 allow_private_urls 开关如何都必须拦截的 IP/网段——云元数据/凭据
# 端点（SSRF 头号目标）与其所在的 link-local 段。IPv4-mapped IPv6 变体
# 也包含：DNS 可能对 IPv4-only host 返回 ``::ffff:x.x.x.x``。
_ALWAYS_BLOCKED_IPS = frozenset({
    ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure/DO/Oracle metadata
    ipaddress.ip_address("169.254.170.2"),     # AWS ECS task metadata
    ipaddress.ip_address("169.254.169.253"),   # Azure IMDS wire server
    ipaddress.ip_address("fd00:ec2::254"),     # AWS metadata (IPv6)
    ipaddress.ip_address("100.100.100.200"),   # Alibaba Cloud metadata
    ipaddress.ip_address("::ffff:169.254.169.254"),
    ipaddress.ip_address("::ffff:169.254.170.2"),
    ipaddress.ip_address("::ffff:169.254.169.253"),
    ipaddress.ip_address("::ffff:100.100.100.200"),
})
_ALWAYS_BLOCKED_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),          # 整个 link-local 段
    ipaddress.ip_network("::ffff:169.254.0.0/112"),  # IPv4-mapped link-local
)

# 允许解析到私网/benchmark IP 的 HTTPS hostname（刻意极窄）。
_TRUSTED_PRIVATE_IP_HOSTS = frozenset({
    "multimedia.nt.qq.com.cn",
})

# 100.64.0.0/10（CGNAT，RFC 6598）不在 ipaddress.is_private 覆盖内，
# 必须显式拦截。
_CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")


# ── 全局开关：允许私网/内网 IP 解析 ────────────────────────────────────
# 首次读取后缓存，避免每次 URL 检查都碰文件系统。
_allow_private_resolved = False
_cached_allow_private: bool = False


def _resolve_allow_private_urls() -> bool:
    """从当前配置作用域解析私网 URL 开关的有效值。"""
    # 1. 环境变量覆盖（最高优先级）
    env_val = os.getenv("HERMES_ALLOW_PRIVATE_URLS", "").strip().lower()
    if env_val in {"true", "1", "yes"}:
        return True
    if env_val in {"false", "0", "no"}:
        return False

    # 2. 配置文件
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config()
        sec = cfg.get("security", {})
        if isinstance(sec, dict) and is_truthy_value(
            sec.get("allow_private_urls"), default=False
        ):
            return True
        browser = cfg.get("browser", {})
        if isinstance(browser, dict) and is_truthy_value(
            browser.get("allow_private_urls"), default=False
        ):
            return True
    except Exception:
        pass
    return False


def _global_allow_private_urls() -> bool:
    """返回用户是否选择关闭私网 IP 拦截。

    优先级：``HERMES_ALLOW_PRIVATE_URLS`` 环境变量 >
    ``security.allow_private_urls`` > ``browser.allow_private_urls``（legacy）。
    单 profile 结果缓存到进程生命周期。
    """
    global _allow_private_resolved, _cached_allow_private

    if get_hermes_home_override() is not None:
        return _resolve_allow_private_urls()

    if _allow_private_resolved:
        return _cached_allow_private
    _allow_private_resolved = True
    _cached_allow_private = _resolve_allow_private_urls()
    return _cached_allow_private


def _reset_allow_private_cache() -> None:
    """重置缓存开关——仅测试用。"""
    global _allow_private_resolved, _cached_allow_private
    _allow_private_resolved = False
    _cached_allow_private = False


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """返回该 IP 是否应被 SSRF 拦截。"""
    # IPv4-mapped IPv6（``::ffff:x.x.x.x``）按其内嵌 IPv4 地址检查
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        embedded_ip = ip.ipv4_mapped
        return (
            embedded_ip.is_private or embedded_ip.is_loopback
            or embedded_ip.is_link_local or embedded_ip.is_reserved
            or embedded_ip.is_multicast or embedded_ip.is_unspecified
            or embedded_ip in _CGNAT_NETWORK
        )

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return True
    if ip.is_multicast or ip.is_unspecified:
        return True
    if ip in _CGNAT_NETWORK:
        return True
    return False


def _allows_private_ip_resolution(hostname: str, scheme: str) -> bool:
    """受信任的 HTTPS hostname 可绕过 IP 类别拦截。"""
    return scheme == "https" and hostname in _TRUSTED_PRIVATE_IP_HOSTS


def is_safe_url(url: str) -> bool:
    """返回 URL 目标是否不是私网/内网地址。

    解析 hostname 到 IP 并检查私网范围。失败即关闭（fail closed）：DNS
    错误与意外异常都拦截请求。配置 ``security.allow_private_urls`` 或
    环境变量 ``HERMES_ALLOW_PRIVATE_URLS=true`` 时跳过私网拦截；云元数据
    端点（169.254.169.254、metadata.google.internal）无论开关都拦截。
    """
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        scheme = (parsed.scheme or "").strip().lower()
        if scheme not in {"http", "https"}:
            logger.warning("Blocked request — unsupported URL scheme: %s", scheme or "<empty>")
            return False
        if not hostname:
            return False

        if hostname in _BLOCKED_HOSTNAMES:
            logger.warning("Blocked request to internal hostname: %s", hostname)
            return False

        allow_all_private = _global_allow_private_urls()
        allow_private_ip = _allows_private_ip_resolution(hostname, scheme)

        try:
            addr_info = socket.getaddrinfo(
                hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM,
            )
        except socket.gaierror:
            # DNS 失败。沙箱/代理环境（NVIDIA OpenShell、Docker+Squid 等）
            # 可能只允许经代理走 HTTP(S)、禁直连 DNS。配置代理时把 DNS 交给
            # 代理解析而不是直接拦截。字面 IP 不需 DNS，getaddrinfo 失败说明
            # 真异常，保持 fail-closed。
            _is_literal_ip = True
            try:
                ipaddress.ip_address(hostname)
            except ValueError:
                _is_literal_ip = False
            if not _is_literal_ip and _proxy_is_configured():
                logger.debug(
                    "DNS resolution failed for %s — proxy configured, "
                    "allowing through for proxy-side resolution",
                    hostname,
                )
                return True
            logger.warning("Blocked request — DNS resolution failed for: %s", hostname)
            return False

        for _family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if "%" in ip_str:
                ip_str = ip_str.split("%")[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                logger.warning(
                    "Blocked request — unparseable IP address %r for hostname %s",
                    sockaddr[0], hostname,
                )
                return False

            if ip in _ALWAYS_BLOCKED_IPS or any(
                ip in net for net in _ALWAYS_BLOCKED_NETWORKS
            ):
                logger.warning(
                    "Blocked request to cloud metadata address: %s -> %s",
                    hostname, ip_str,
                )
                return False

            if not allow_all_private and not allow_private_ip and _is_blocked_ip(ip):
                logger.warning(
                    "Blocked request to private/internal address: %s -> %s",
                    hostname, ip_str,
                )
                return False

        return True
    except Exception as exc:
        logger.warning("Blocked request — URL safety check error for %s: %s", url, exc)
        return False


async def async_is_safe_url(url: str) -> bool:
    """与 :func:`is_safe_url` 同规则，但 DNS 工作丢到事件循环外执行。

    ``socket.getaddrinfo`` 可能阻塞——异步路径（gateway、
    ``web_extract_tool``、vision 下载钩子）用本函数而非 ``is_safe_url``。
    """
    return await asyncio.to_thread(is_safe_url, url)
