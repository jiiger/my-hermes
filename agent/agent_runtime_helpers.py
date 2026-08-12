"""运行时辅助（精简移植版）。

client 创建/关闭 + strip_think_blocks（interim 旁白交付用）。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from utils import base_url_host_matches


def _ra():
    """Lazy ``run_agent`` reference for test-patch routing."""
    import run_agent

    return run_agent


def create_openai_client(
    agent, client_kwargs: dict, *, reason: str, shared: bool
) -> Any:

    from agent.auxiliary_client import _validate_base_url, _validate_proxy_env_urls
    from agent.ssl_verify import resolve_httpx_verify

    client_kwargs = dict(client_kwargs)
    # 验证网络
    ssl_ca_cert = client_kwargs.pop("ssl_ca_cert", None)
    ssl_verify_cfg = client_kwargs.pop("ssl_verify", None)

    httpx_verify = resolve_httpx_verify(
        ca_bundle=ssl_ca_cert, ssl_verify=ssl_verify_cfg
    )
    _validate_proxy_env_urls()
    _validate_base_url(client_kwargs.get("base_url"))

    # TODO 判断provider 是不是copilot或者gemini,创建专属client

    # 注入 TCP 保活机制，防止 Agent 永久挂起
    if "http_client" not in client_kwargs:
        keepalive_http = agent._build_keepalive_http_client(
            client_kwargs.get("base_url", ""),
            verify=httpx_verify,
        )
        if keepalive_http is not None:
            client_kwargs["http_client"] = keepalive_http

    # 禁用 SDK 默认重试，通过 client_kwargs.setdefault("max_retries", 0) 强制关闭 OpenAI SDK 的内置重试。
    client_kwargs.setdefault("max_retries", 0)

    # TODO 自动补全 GitHub Copilot 必需的请求头，防止路由错误 (Header Injection for Copilot)

    client = _ra().OpenAI(**client_kwargs)

    _ra().logger.info(
        "OpenAI client created (%s, shared=%s) %s",
        reason,
        shared,
        agent._client_log_context(),
    )
    return client


def copy_reasoning_content_for_api(agent, source_msg: dict, api_msg: dict) -> None:

    from agent.message_sanitization import apply_reasoning_content_policy

    apply_reasoning_content_policy(
        source_msg, api_msg, agent._needs_thinking_reasoning_pad()
    )


# ─── strip_think_blocks（对应原版 agent/agent_runtime_helpers.py:809 精简版）───

_REASONING_TAG_NAMES = (
    "think",
    "thinking",
    "reasoning",
    "REASONING_SCRATCHPAD",
    "thought",
)
_TOOL_CALL_TAG_NAMES = (
    "tool_call",
    "tool_calls",
    "tool_result",
    "function_call",
    "function_calls",
)

# 闭合推理块：<think>…</think>（含全部变体，大小写不敏感）
_REASONING_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _REASONING_TAG_NAMES
)
# 闭合工具调用 XML 块：<tool_call …>…</tool_call> 等
_TOOL_CALL_BLOCK_PATTERNS = tuple(
    re.compile(rf"<{name}\b[^>]*>.*?</{name}>", re.DOTALL | re.IGNORECASE)
    for name in _TOOL_CALL_TAG_NAMES
)
# Gemma 风格 <function name="…">…</function>（仅在行首或标点后，避免误伤正文）
_NAMED_FUNCTION_BLOCK_PATTERN = re.compile(
    r"(?:(?<=^)|(?<=[\n\r.!?:]))[ \t]*"
    r"<function\b[^>]*\bname\s*=[^>]*>"
    r"(?:(?:(?!</function>).)*)</function>",
    re.DOTALL | re.IGNORECASE,
)
# 未闭合推理块（某些端点丢弃闭合标签）：从开标签到串尾整体剥掉
_UNTERMINATED_REASONING_BLOCK_PATTERN = re.compile(
    rf"(?:^|\n)[ \t]*<(?:{'|'.join(_REASONING_TAG_NAMES)})\b[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)
# 游离的孤儿开/闭标签
_ORPHAN_REASONING_TAG_PATTERN = re.compile(
    rf"</?(?:{'|'.join(_REASONING_TAG_NAMES)})>\s*",
    re.IGNORECASE,
)
# 漏网的闭合标签
_STRAY_TOOL_CALL_CLOSER_PATTERN = re.compile(
    rf"</(?:{'|'.join(_TOOL_CALL_TAG_NAMES)}|function)>\s*",
    re.IGNORECASE,
)


def strip_think_blocks(agent, content: str) -> str:
    """从 content 中移除推理/思考块，只返回可见文本。

    对应原版 agent/agent_runtime_helpers.py:809 的精简版，处理四类：
      1. 闭合标签对（<think>…</think> 及 thinking/reasoning/
         REASONING_SCRATCHPAD/thought 变体，大小写不敏感）；
      2. 未闭合的开标签（从开标签到串尾剥掉）；
      3. 游离的孤儿开/闭标签；
      4. 开放模型在 content 里直接输出的工具调用 XML 块
         （tool_call / function_call / Gemma 风格 <function name=…>）。

    content 可能是 str、结构块列表（Anthropic 风格）或 dict。
    """
    if not content:
        return ""
    # 非字符串内容：展平为可见文本（跳过 thinking/reasoning 类型块）
    if not isinstance(content, str):
        if isinstance(content, list):
            _parts: list[str] = []
            for _part in content:
                if isinstance(_part, str):
                    _parts.append(_part)
                elif isinstance(_part, dict):
                    _ptype = str(_part.get("type") or "").strip().lower()
                    if _ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    _text = _part.get("text")
                    if isinstance(_text, str) and _text:
                        _parts.append(_text)
            content = "".join(_parts)
        elif isinstance(content, dict):
            content = str(content.get("text") or content.get("content") or "")
        else:
            content = str(content)
        if not content:
            return ""
    for _pattern in _REASONING_BLOCK_PATTERNS:
        content = _pattern.sub("", content)
    for _pattern in _TOOL_CALL_BLOCK_PATTERNS:
        content = _pattern.sub("", content)
    content = _NAMED_FUNCTION_BLOCK_PATTERN.sub("", content)
    content = _UNTERMINATED_REASONING_BLOCK_PATTERN.sub("", content)
    content = _ORPHAN_REASONING_TAG_PATTERN.sub("", content)
    content = _STRAY_TOOL_CALL_CLOSER_PATTERN.sub("", content)
    return content
