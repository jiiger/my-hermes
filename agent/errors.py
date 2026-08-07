"""精简版异常定义（对应原版 agent/errors.py）。

原版还有 EmptyStreamError、MoAPresetNotFoundError 等，my-hermes
目前只有 ssl_guard 用到 SSLConfigurationError，其余按需再补。
"""


class SSLConfigurationError(Exception):
    """SSL/TLS 配置错误（证书、CA bundle、校验开关等）。"""
