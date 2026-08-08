"""_summary_
运行时辅助client的创建/关闭
"""

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
