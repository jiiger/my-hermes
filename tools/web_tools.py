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
import re
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from agent.redact import _PREFIX_RE
from tools.registry import registry, tool_error
from tools.url_safety import (
    async_is_safe_url,
    normalize_url_for_request,
    sensitive_query_param_name,
)

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


# ── web_extract 辅助（URL 归一化 / 安全 / 截断落盘）────────────────────────


def _web_extract_url(value: Any) -> Optional[str]:
    """从模型提供的提取项里取可用 URL。

    模型有时会转发整个 web-search 结果而非其 URL。接受常见的两个 URL 键，
    但拒绝缺失/非字符串值——不要为误导性的抓取目标 stringify 任意对象。
    """
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


DEFAULT_EXTRACT_CHAR_LIMIT = 15000

# 写入 cache/web 全文文件的上限。截断-落盘路径若不做上限，多 MB 页面会
# 无界写盘。上限 2MB 远超任何单次 read_file 分页需要；模型只看到
# char_limit 窗口，所以落盘副本被上限化不损失可用性。
MAX_STORED_TEXT_CHARS = 2_000_000


def _get_extract_char_limit() -> int:
    """从配置解析每页字符预算，并钳制到合理区间。"""
    try:
        configured = _load_web_config().get("extract_char_limit")
        if configured is not None:
            value = int(configured)
            # 下限 2k（再低 footer 会占大头），上限给宽松护栏防 typo 撑爆上下文。
            return max(2000, min(value, 500_000))
    except (TypeError, ValueError):
        pass
    return DEFAULT_EXTRACT_CHAR_LIMIT


def convert_base64_images_to_links(text: str) -> str:
    """把内联 base64 图片 blob 替换成带标签的 markdown 链接。

    base64 图片载荷是 token 炸弹（单个内联 PNG 可达数万字符），绝不把原始
    字节发给模型。但保留"这里曾有图"的事实与 alt 文本，作为可检查的占位符。
    真实（http/https）markdown 图片链接保持不动，agent 可以继续
    ``web_extract`` / ``vision_analyze`` 它们。

    变换：
      ``![alt](data:image/png;base64,AAAA...)``  -> ``[IMAGE: alt]``
      ``(data:image/png;base64,AAAA...)``        -> ``[IMAGE]``
      裸 ``data:image/...;base64,AAAA...``       -> ``[IMAGE]``
    """
    def _md_repl(m: "re.Match[str]") -> str:
        alt = (m.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    md_b64 = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
    )
    out = md_b64.sub(_md_repl, text)

    # 圆括号 base64（非 markdown）与裸 base64 → [IMAGE]
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    out = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)
    return out


def _store_full_text(url: str, content: str) -> Optional[str]:
    """把提取到的完整页面写入 cache/web，返回其绝对路径。

    落盘是 best-effort：失败只记 debug 并返回 None，截断后的内容照常返回。
    """
    try:
        import hashlib
        from urllib.parse import urlparse

        from hermes_constants import get_hermes_dir

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        host = (urlparse(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"{slug}-{digest}.md"
        # 给病态大页面设上限，避免无界写盘。若被截断，追加标记让读文件者
        # 知道这不是字面的完整页面。
        if len(content) > MAX_STORED_TEXT_CHARS:
            content = (
                content[:MAX_STORED_TEXT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_TEXT_CHARS:,} chars "
                f"of {len(content):,}; re-extract a more specific URL for the rest ...]"
            )
        from tools.spill_safety import write_text_exclusive

        # 已知目录里的确定性文件名：经 lstat-unlink + 独占创建拒绝符号链接。
        # 同一 URL 的再提取合法覆盖（同名 slug-digest）。非 private：
        # cache/web 会被 bind-mount 进远端后端，容器 UID 必须能读。
        write_text_exclusive(path, content, private=False, overwrite=True)
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full web_extract text for %s: %s", url, exc)
        return None


def _truncate_with_footer(
    content: str,
    url: str,
    char_limit: int,
) -> tuple[str, bool]:
    """返回单页干净内容的 (model_text, was_truncated)。

    不超 ``char_limit`` 的页面原样返回。更大的页面取 head+tail 窗口
    （约 75% head / 25% tail），尽量在 markdown 行边界切割，并附上明确
    footer：告诉模型它看到多少、全文存在哪、哪个 read_file 调用翻页取
    被省略的中间段。确定性——无模型参与。
    """
    if len(content) <= char_limit:
        return content, False

    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget

    head = content[:head_budget]
    tail = content[-tail_budget:]
    # head 切割回退到最后一个换行，避免断在行中间。
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # tail 切割前进到下一个换行，同样避免断行。
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    total = len(content)
    stored_path = _store_full_text(url, content)

    footer_lines = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {total:,} total clean characters.",
    ]
    if stored_path:
        # 被省略的中间段从我们展示的 head 之后开始。给模型一个具体起始行
        # （head 行数 + 1），它的第一次 read_file 就能落在缺口而不是瞎猜
        # <line>。read_file 从 1 计数；+1 越过已展示的 head 最后一行。
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full text saved to: {stored_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete page; "
            f"raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL or use browser_navigate for the complete page."
        )
    footer_lines.append("─" * 29)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    model_text += "\n" + "\n".join(footer_lines)
    return model_text, True


async def web_extract_tool(
    urls: List[Any],
    format: str = None,
    char_limit: Optional[int] = None,
) -> str:
    """从指定 URL 提取内容并返回 JSON 结果字符串。

    流程：URL 归一化 → 内嵌密钥/敏感参数拦截 → SSRF 过滤 → 后端提取 →
    顺序重建 → base64 图片转占位 + 超长截断 + 全文落盘 → 精简输出字段。
    extract 后端（Tavily、Firecrawl、Exa、Parallel）返回干净、去样板的
    内容，本工具直接返回不摘要。search-only 后端（ddgs / brave-free /
    searxng）返回原版同款 "search-only backend" 错误。

    Args:
        urls: URL 字符串列表，或含 ``url`` / ``href`` 字符串字段的对象。
        format: 期望输出格式（"markdown" / "html"，可选）。
        char_limit: 每页字符预算（默认 web.extract_char_limit 或 15000）。
            更大的页面 head+tail 截断，全文落盘到 cache/web。
    """
    # ── 1. URL 归一化（非法项记 invalid_urls，不中断整体）───────────────
    normalized_urls: List[str] = []
    normalized_indices: List[int] = []
    invalid_urls: Dict[int, Dict[str, Any]] = {}
    for index, item in enumerate(urls):
        _url = _web_extract_url(item)
        if _url is None:
            invalid_urls[index] = {
                "url": "",
                "title": "",
                "content": "",
                "error": (
                    f"Invalid URL item at index {index}: expected a URL string "
                    "or an object with a string 'url' or 'href' field"
                ),
            }
            continue
        normalized_url = normalize_url_for_request(_url)
        # 内嵌密钥拦截（先 URL-decode，防百分号编码绕过 %73k- = sk-）。
        if (
            _PREFIX_RE.search(_url)
            or _PREFIX_RE.search(unquote(_url))
            or _PREFIX_RE.search(normalized_url)
            or _PREFIX_RE.search(unquote(normalized_url))
        ):
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: URL contains what appears to be an API key or token. "
                    "Secrets must not be sent in URLs."
                ),
            })
        sensitive_query_key = sensitive_query_param_name(normalized_url)
        if sensitive_query_key:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: URL contains a credential-like query parameter "
                    f"({sensitive_query_key}). Web extract backends are third-party "
                    "readers; remove the sensitive query parameter or use a local "
                    "browser session when this access is explicitly required."
                ),
            })
        normalized_urls.append(normalized_url)
        normalized_indices.append(index)

    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted", success=False)

        logger.info("Extracting content from %d URL(s)", len(normalized_urls))

        # ── 2. SSRF 防护：任何后端之前过滤私网/内网 URL ──────────────────
        safe_urls: List[str] = []
        safe_indices: List[int] = []
        ssrf_blocked: Dict[int, Dict[str, Any]] = {}
        for index, url in zip(normalized_indices, normalized_urls):
            if not await async_is_safe_url(url):
                ssrf_blocked[index] = {
                    "url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                }
            else:
                safe_urls.append(url)
                safe_indices.append(index)

        # ── 3. 只把安全 URL 派发给配置后端 ───────────────────────────────
        if not safe_urls:
            results: List[Dict[str, Any]] = []
        else:
            _ensure_web_plugins_loaded()
            from agent.web_search_registry import (
                get_active_extract_provider,
                get_provider as _wsp_get_provider,
            )

            backend = _get_extract_backend()
            provider = _wsp_get_provider(backend) if backend else None
            if provider is None or not provider.supports_extract():
                # 配置名已注册但不支持 extract（search-only 后端）→ 类型化
                # 错误而非静默换后端；名字未注册则回退活动 provider 走查。
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

            logger.info("Web extract via %s: %d URL(s)", provider.name, len(safe_urls))

            # 同步/异步双派发：async extract() 直接 await；同步 extract() 丢
            # 线程，避免阻塞事件循环的网络 I/O。
            if inspect.iscoroutinefunction(provider.extract):
                results = await provider.extract(safe_urls, format=format)
            else:
                results = await asyncio.to_thread(
                    provider.extract, safe_urls, format=format,
                )

        # ── 4. 顺序重建：invalid / ssrf_blocked / 正常结果按原始 index 拼回 ──
        if invalid_urls or ssrf_blocked:
            safe_results = {
                index: (
                    results[position]
                    if position < len(results)
                    else {
                        "url": safe_urls[position],
                        "title": "",
                        "content": "",
                        "error": "Extract backend returned no result for this URL",
                    }
                )
                for position, index in enumerate(safe_indices)
            }
            by_index = {**safe_results, **ssrf_blocked, **invalid_urls}
            results = [by_index[index] for index in range(len(urls))]

        response = {"results": results}

        # ── 5. 后处理：base64 转占位 + 超长截断 + 全文落盘 ───────────────
        effective_char_limit = (
            char_limit if char_limit is not None else _get_extract_char_limit()
        )
        try:
            effective_char_limit = max(
                2000, min(int(effective_char_limit), 500_000),
            )
        except (TypeError, ValueError):
            effective_char_limit = DEFAULT_EXTRACT_CHAR_LIMIT

        for result in response.get("results", []):
            if result.get("error"):
                continue
            url = result.get("url", "")
            raw_content = result.get("raw_content", "") or result.get("content", "")
            if not raw_content:
                continue
            clean = convert_base64_images_to_links(raw_content)
            model_text, truncated = _truncate_with_footer(
                clean, url, effective_char_limit,
            )
            result["content"] = model_text
            if truncated:
                logger.info("%s (truncated %d -> %d chars)", url, len(clean), len(model_text))
            else:
                logger.info("%s (%d chars, whole)", url, len(clean))

        # 精简输出字段：每条只留 url / title / content / error。
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

        if trimmed_response.get("results") == []:
            result_json = tool_error("Content was inaccessible or not found")
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)

        # base64 占位已在单条后处理完成；这里对序列化 JSON 再扫一遍兜底，
        # 防 provider 把 blob 塞进意外位置（如 metadata）。
        cleaned_result = convert_base64_images_to_links(result_json)
        return cleaned_result
    except Exception as exc:  # noqa: BLE001
        logger.debug("Web extract error: %s", exc)
        return tool_error(f"Error extracting content: {exc}")


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
