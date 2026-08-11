"""API 错误分类：把异常归类为恢复策略（回退/重试/压缩）。

对应原版 agent/error_classifier.py（1841 行）的 FailoverReason + classify_api_error；
精简版只保留 my-hermes 实际会用到的分类，去掉原版数百行的消息模式匹配，
只按 状态码 → 异常类型 → 关键词 三级分类。
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class FailoverReason(enum.Enum):
    """API 调用失败的原因 —— 决定恢复策略（回退/重试/压缩）。

    对应原版 agent/error_classifier.py 的 FailoverReason，精简版只保留
    my-hermes 实际会用到的分类。
    """

    # 认证失败（401/403）—— 需要轮换/刷新凭据
    auth = "auth"
    # 认证彻底失败（刷新后仍失败）—— 放弃本轮
    auth_permanent = "auth_permanent"
    # 计费/额度问题（402 或确认余额不足）—— 立即切换 provider
    billing = "billing"
    # 限流（429）—— 退避后重试，重试耗尽再切换
    rate_limit = "rate_limit"
    # 上游（聚合商）限流 —— 与 rate_limit 语义相同但来源不同
    upstream_rate_limit = "upstream_rate_limit"
    # 服务过载（503/529）—— 退避重试
    overloaded = "overloaded"
    # 服务器内部错误（500/502）—— 重试
    server_error = "server_error"
    # 连接/读超时 —— 重建客户端后重试
    timeout = "timeout"
    # SSL 证书校验失败
    ssl_cert_verification = "ssl_cert_verification"
    # 上下文超长 —— 压缩上下文而不是回退
    context_overflow = "context_overflow"
    # 载荷过大（413）—— 压缩载荷
    payload_too_large = "payload_too_large"
    # 模型不存在（404 或无效模型名）—— 换模型
    model_not_found = "model_not_found"
    # 内容策略拦截（安全过滤）—— 不重试，需要改提示词
    content_policy_blocked = "content_policy_blocked"
    # 请求格式错误（400）—— 中止或裁剪后重试
    format_error = "format_error"
    # 无法分类 —— 带退避重试
    unknown = "unknown"


@dataclass
class ClassifiedError:
    """一次 API 错误的分类结果（字段对齐原版 agent/error_classifier.py:77）。

    - reason: 失败原因（FailoverReason），上层据此决定 重试/回退/压缩；
    - status_code: 原始 HTTP 状态码（可能没有）；
    - provider / model: 出错时的 provider 与模型（诊断/恢复用）；
    - message: 原始错误消息（保留用于日志/展示）；
    - error_context: 附加错误上下文（响应 body、错误码等）；
    - retryable: 该错误是否值得原样重试（auth/计费/策略拦截等重试无意义）；
    - should_compress / should_rotate_credential / should_fallback: 恢复动作提示，
      重试循环直接按此执行，不必再自行分类。
    """

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        """是否认证类失败（含刷新后仍失败的彻底失败）。"""
        return self.reason in {FailoverReason.auth, FailoverReason.auth_permanent}


# ── 关键词模式（状态码缺失时的兜底判断）──────────────────────────────────

_TIMEOUT_PATTERNS = ("timeout", "timed out", "timedout", "read timed")
_CONNECT_PATTERNS = (
    "connection error",
    "connection refused",
    "connection reset",
    "connection closed",
    "network error",
    "failed to connect",
    "cannot connect",
)
_RATE_LIMIT_PATTERNS = ("rate limit", "too many requests", "quota exceeded", "throttl")
_SSL_PATTERNS = (
    "ssl",
    "certificate verify",
    "cert verify",
    "self-signed",
    "tls handshake",
)
_AUTH_PATTERNS = (
    "api key",
    "invalid key",
    "authentication",
    "unauthorized",
    "incorrect api",
    "invalid api",
)
_BILLING_PATTERNS = ("billing", "insufficient", "balance", "credit", "payment")
_CONTEXT_PATTERNS = (
    "context length",
    "maximum context",
    "context window",
    "context overflow",
    "token limit",
    "too many tokens",
    "max context",
)
_POLICY_PATTERNS = ("policy", "safety", "content filter", "blocked by", "harmful", "violat")


def _contains_any(text: str, patterns) -> bool:
    """text 是否命中任意一个关键词（text 需已小写）。"""
    return any(p in text for p in patterns)


def _extract_status_code(error: Exception) -> Optional[int]:
    """从异常对象提取 HTTP 状态码。

    openai SDK 的 APIStatusError（BadRequestError/NotFoundError/RateLimitError 等）
    都带 status_code 属性；其他异常（超时/连接错误）没有。
    """
    sc = getattr(error, "status_code", None)
    if sc is not None:
        try:
            return int(sc)
        except (TypeError, ValueError):
            return None
    return None


def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
) -> ClassifiedError:
    """把一次 API 异常分类成恢复建议（对应原版 error_classifier.py:623）。

    优先级：
      1. HTTP 状态码（最可靠，openai SDK 异常自带）；
      2. 异常类型（超时/连接错误）；
      3. 错误消息关键词（无状态码时的兜底）。

    精简版不做原版那套数百行的 provider 特化匹配；参数签名保留
    provider/model/approx_tokens 等，方便以后按 provider 细化。

    Args:
        error: API 调用抛出的异常。
        provider: 当前 provider 名（精简版暂不使用）。
        model: 当前模型名（精简版暂不使用）。
        approx_tokens: 上下文大致 token 数（精简版暂不使用）。
        context_length: 当前模型上下文长度（精简版暂不使用）。
        num_messages: 消息条数（精简版暂不使用）。

    Returns:
        ClassifiedError：含 reason（恢复策略依据）与 retryable（是否值得重试）。
    """
    # 精简版尚未使用这些上下文参数；保留在签名里对齐原版接口
    del provider, model, approx_tokens, context_length, num_messages

    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    msg = str(error).lower()

    # ── 1) 按 HTTP 状态码分类 ──
    if status_code is not None:
        if status_code in (401, 403):
            # 认证失败：换凭据/回退，原样重试无意义
            return ClassifiedError(reason=FailoverReason.auth, status_code=status_code, message=str(error), retryable=False)
        if status_code == 402:
            # 余额不足：立即切换 provider
            return ClassifiedError(reason=FailoverReason.billing, status_code=status_code, message=str(error), retryable=False)
        if status_code == 429:
            return ClassifiedError(reason=FailoverReason.rate_limit, status_code=status_code, message=str(error))
        if status_code == 404:
            # 模型不存在：换模型/回退
            return ClassifiedError(reason=FailoverReason.model_not_found, status_code=status_code, message=str(error), retryable=False)
        if status_code == 413:
            return ClassifiedError(
                reason=FailoverReason.payload_too_large,
                status_code=status_code,
                message=str(error),
                should_compress=True,
            )
        if status_code in (500, 502):
            return ClassifiedError(reason=FailoverReason.server_error, status_code=status_code, message=str(error))
        if status_code in (503, 529):
            return ClassifiedError(reason=FailoverReason.overloaded, status_code=status_code, message=str(error))
        if status_code == 400:
            # 400 最模糊：按消息细分
            if _contains_any(msg, _CONTEXT_PATTERNS):
                return ClassifiedError(
                    reason=FailoverReason.context_overflow,
                    status_code=status_code,
                    message=str(error),
                    should_compress=True,
                )
            if _contains_any(msg, _POLICY_PATTERNS):
                return ClassifiedError(
                    reason=FailoverReason.content_policy_blocked, status_code=status_code, message=str(error), retryable=False
                )
            return ClassifiedError(reason=FailoverReason.format_error, status_code=status_code, message=str(error), retryable=False)
        if 400 <= status_code < 500:
            return ClassifiedError(reason=FailoverReason.format_error, status_code=status_code, message=str(error), retryable=False)
        if status_code >= 500:
            return ClassifiedError(reason=FailoverReason.server_error, status_code=status_code, message=str(error))

    # ── 2) 无状态码：按异常类型 / 消息关键词 ──
    if error_type in ("APITimeoutError", "TimeoutError") or _contains_any(msg, _TIMEOUT_PATTERNS):
        return ClassifiedError(reason=FailoverReason.timeout, status_code=None, message=str(error))
    if error_type in ("APIConnectionError", "ConnectionError") or _contains_any(msg, _CONNECT_PATTERNS):
        return ClassifiedError(reason=FailoverReason.timeout, status_code=None, message=str(error))
    if error_type == "RateLimitError" or _contains_any(msg, _RATE_LIMIT_PATTERNS):
        return ClassifiedError(reason=FailoverReason.rate_limit, status_code=None, message=str(error))
    if _contains_any(msg, _SSL_PATTERNS):
        return ClassifiedError(reason=FailoverReason.ssl_cert_verification, status_code=None, message=str(error))
    if _contains_any(msg, _AUTH_PATTERNS):
        return ClassifiedError(reason=FailoverReason.auth, status_code=None, message=str(error), retryable=False)
    if _contains_any(msg, _BILLING_PATTERNS):
        return ClassifiedError(reason=FailoverReason.billing, status_code=None, message=str(error), retryable=False)
    if _contains_any(msg, _CONTEXT_PATTERNS):
        return ClassifiedError(
            reason=FailoverReason.context_overflow,
            status_code=None,
            message=str(error),
            should_compress=True,
        )
    if _contains_any(msg, _POLICY_PATTERNS):
        return ClassifiedError(
            reason=FailoverReason.content_policy_blocked, status_code=None, message=str(error), retryable=False
        )

    # ── 3) 兜底：无法分类，带退避重试 ──
    return ClassifiedError(reason=FailoverReason.unknown, status_code=status_code, message=str(error))
