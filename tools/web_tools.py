"""Web 搜索/提取工具（精简移植版）。

对应原版 hermes-agent 的 tools/web_tools.py。my-hermes 版把后端分发
完全交给 ``agent.web_search_registry``：toolsets.py 已声明
``web_search`` / ``web_extract`` 两个工具名，本文件在模块级自注册把它们
接进 registry（AST 工具发现自动 import 本文件触发注册，无需改 model_tools）。

提供工具：
- ``web_search_tool``：同步，搜索网络；
- ``web_extract_tool``：async，提取 URL 内容（ddgs 等 search-only 后端
  返回原版同款错误）。

裁剪项（相对原版 tools/web_tools.py）：
- 七后端硬编码分发与 ``_LEGACY_WEB_BACKENDS``：后端可用性一律从注册表
  动态判定（"注册表里有啥算啥"）；
- ``_debug`` 日志（WEB_TOOLS_DEBUG）；
- ``convert_base64_images_to_links`` / ``_store_full_text``：extract 结果
  直接返回，不做 base64 内联转占位与全文落盘；
- SSRF / 内嵌密钥防护（依赖 agent/redact 与网络安全检查，my-hermes 未移植）；
- ``_disabled_web_plugin_for`` 分支（my-hermes 无插件禁用机制）。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ── 配置读取 ─────────────────────────────────────────────────────────────


def _load_web_config() -> dict:
    """读取 ~/.hermes/config.yaml 的 ``web:`` 段。

    ``or {}``：``web:`` 段存在但为 null（YAML ``web:`` 无正文）时
    ``.get("web", {})`` 返回 None，用 ``or {}`` 保证返回 dict 契约。
    """
    try:
        from hermes_cli.config import load_config

        return load_config().get("web") or {}
    except Exception:  # noqa: BLE001 — 配置读取 best-effort
        return {}


# ── 注册表查询（后端可用性从注册表动态判定）──────────────────────────────


def _registered_web_provider(backend: str):
    """按名字查插件注册的 web provider，未注册返回 None。"""
    if not backend:
        return None
    try:
        from agent.web_search_registry import get_provider

        return get_provider(backend)
    except Exception as exc:  # noqa: BLE001 — 注册表可选，绝不致命
        logger.debug("web provider registry lookup failed for %r: %s", backend, exc)
        return None


def _registered_web_provider_available(backend: str):
    """已注册 provider 的可用性；未注册返回 None（让调用方继续回退）。"""
    provider = _registered_web_provider(backend)
    if provider is None:
        return None
    try:
        return bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001 — 坏 provider 视为不可用
        logger.debug("web provider %r.is_available() raised: %s", backend, exc)
        return False


def _list_registered_web_providers():
    """返回全部已注册 web provider（失败返回空列表）。"""
    try:
        from agent.web_search_registry import list_providers

        return list_providers()
    except Exception as exc:  # noqa: BLE001
        logger.debug("web provider registry list failed: %s", exc)
        return []


# ── 后端选择 ─────────────────────────────────────────────────────────────


def _get_backend() -> Optional[str]:
    """确定共享 web 后端（读 ``web.backend``，回退到注册表可用 provider）。

    相对原版：不再硬编码 ``_LEGACY_WEB_BACKENDS`` 候选序——注册表里有啥
    算啥，按 ``is_available()`` 过滤。
    """
    configured = (_load_web_config().get("backend") or "").strip().lower()
    if configured and _registered_web_provider_available(configured):
        return configured

    # 回退：注册表里第一个可用 provider。
    for provider in _list_registered_web_providers():
        try:
            if provider.is_available():
                return provider.name
        except Exception as exc:  # noqa: BLE001 — 坏 provider 跳过
            logger.debug("web provider %r.is_available() raised: %s", provider.name, exc)
    return None


def _get_capability_backend(capability: str) -> Optional[str]:
    """按能力选后端：``web.{capability}_backend`` 优先，否则共享 ``_get_backend``。"""
    cfg = _load_web_config()
    specific = (cfg.get(f"{capability}_backend") or "").strip().lower()
    if specific and _registered_web_provider_available(specific):
        return specific
    return _get_backend()


def _get_search_backend() -> Optional[str]:
    """web_search 专用后端（``web.search_backend`` → ``web.backend`` → 自动）。"""
    return _get_capability_backend("search")


def _get_extract_backend() -> Optional[str]:
    """web_extract 专用后端（``web.extract_backend`` → ``web.backend`` → 自动）。"""
    return _get_capability_backend("extract")


# ── 插件懒发现 ───────────────────────────────────────────────────────────


def _ensure_web_plugins_loaded() -> None:
    """幂等触发插件发现，让 web 注册表被填充。

    内置 web provider（当前只有 ddgs）在插件发现期经
    ``plugins/web/<vendor>/__init__.py`` 自注册。工具派发可能从尚未触发
    发现的上下文进入（独立脚本、测试路径），没有它注册表就是空的。
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        # warning 而非 debug：插件导入真坏时，用户会撞上误导性的
        # "No web search provider configured"，warning 留下真实线索。
        logger.warning("Web plugin discovery failed (non-fatal): %s", exc)


# ── 工具实现 ─────────────────────────────────────────────────────────────


def web_search_tool(query: str, limit: int = 5) -> str:
    """搜索网络并返回 JSON 结果字符串。

    只返回搜索元数据（URL、标题、描述）；要完整内容请用 web_extract_tool。

    返回结构::

        {
            "success": bool,
            "data": {"web": [{"title", "url", "description", "position"}, ...]}
        }

    失败时: ``{"success": False, "error": str}``。
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 100)

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # 经注册表派发：同步——每个 provider 的 search() 都是同步。
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_search_provider,
            get_provider as _wsp_get_provider,
        )

        backend = _get_search_backend()
        provider = _wsp_get_provider(backend) if backend else None
        if provider is None or not provider.supports_search():
            # 配置的后端不是已注册搜索 provider（笔误/插件未装/能力不匹配）
            # 时，回退到可用性走查的活动 provider。
            provider = get_active_search_provider()

        if provider is None:
            response_data = {
                "success": False,
                "error": (
                    "No web search provider configured. "
                    "Run `hermes tools` to set one up."
                ),
            }
        else:
            logger.info(
                "Web search via %s: '%s' (limit: %d)",
                provider.name, query, limit,
            )
            response_data = provider.search(query, limit)

        return json.dumps(response_data, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Web search error: %s", exc)
        return tool_error(f"Error searching web: {exc}")


async def web_extract_tool(
    urls: List[Any],
    format: str = None,
    char_limit: Optional[int] = None,
) -> str:
    """从指定 URL 提取内容并返回 JSON 结果字符串。

    extract 后端（Firecrawl、Tavily、Exa、Parallel）返回干净、去样板的内容，
    本工具直接返回不摘要。search-only 后端（ddgs / brave-free / searxng）
    返回原版同款 "search-only backend" 错误。

    Args:
        urls: URL 字符串列表，或含 ``url`` / ``href`` 字符串字段的对象。
        format: 期望输出格式（"markdown" / "html"，可选）。
        char_limit: 每页字符预算（可选，my-hermes 直接透传给 provider）。
    """
    del char_limit  # my-hermes 裁剪截断/落盘，保留签名兼容

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted", success=False)

        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_extract_provider,
            get_provider as _wsp_get_provider,
        )

        backend = _get_extract_backend()
        provider = _wsp_get_provider(backend) if backend else None
        if provider is None or not provider.supports_extract():
            # 配置名已注册但不支持 extract（search-only 后端）→ 给出类型化
            # 错误，而不是静默换后端；名字未注册则回退活动 provider 走查。
            if provider is not None and not provider.supports_extract():
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"{provider.display_name} is a search-only "
                            "backend and cannot extract URL content. "
                            "Set web.extract_backend to firecrawl, "
                            "tavily, exa, or parallel."
                        ),
                    },
                    ensure_ascii=False,
                )
            provider = get_active_extract_provider()
            if provider is None:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            "No web extract provider configured. "
                            "Set web.extract_backend to firecrawl, "
                            "tavily, exa, or parallel."
                        ),
                    },
                    ensure_ascii=False,
                )

        logger.info("Web extract via %s: %d URL(s)", provider.name, len(urls))

        # 同步/异步分派：async extract() 直接 await；同步 extract() 丢线程，
        # 避免阻塞事件循环的网络 I/O。
        if inspect.iscoroutinefunction(provider.extract):
            results = await provider.extract(urls, format=format)
        else:
            results = await asyncio.to_thread(provider.extract, urls, format=format)

        return json.dumps({"results": results}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Web extract error: %s", exc)
        return tool_error(f"Error extracting web content: {exc}")


# ── 工具 schema ──────────────────────────────────────────────────────────


WEB_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "description": (
        "Search the web for up-to-date information. Returns a JSON list of "
        "results with title, URL, and description. Use for recent events, "
        "docs, prices, or anything that might have changed since the "
        "knowledge cutoff."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
}


WEB_EXTRACT_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "description": (
        "Extract clean page content (markdown/text) from one or more URLs. "
        "Returns title and content for each page. Use when you need the "
        "full text of a page rather than just search snippets."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5,
            },
        },
        "required": ["urls"],
    },
}


# ── 模块级自注册（照抄 tools/session_search_tool.py 尾部的注册写法）──────


registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda **kw: web_search_tool(
        kw.get("query", ""),
        limit=kw.get("limit", 5),
    ),
    check_fn=None,  # 工具始终可见；调用时才懒发现插件并解析 provider
    emoji="🔍",
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda **kw: asyncio.run(web_extract_tool(
        kw.get("urls", [])[:5] if isinstance(kw.get("urls"), list) else [],
        "markdown",
    )),
    check_fn=None,
    is_async=True,
    emoji="📄",
)
