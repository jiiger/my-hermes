"""Tavily web 搜索 + 内容提取插件 provider（精简移植版）。

对应原版 hermes-agent 的 plugins/web/tavily/provider.py。子类化
:class:`agent.web_search_provider.WebSearchProvider`，宣告两种能力：

- ``supports_search()``  -> True（Tavily ``/search``）
- ``supports_extract()`` -> True（Tavily ``/extract``）

两者都是同步——底层调用是 ``httpx.post(...)``（httpx 是 my-hermes 已有
依赖，不新增第三方包）。

配置键::

    web:
      search_backend: "tavily"     # 显式按能力覆盖
      extract_backend: "tavily"    # 显式按能力覆盖
      backend: "tavily"            # 共享回退

环境变量::

    TAVILY_API_KEY=...           # https://app.tavily.com/home（必需）
    TAVILY_BASE_URL=...          # 可选覆盖 https://api.tavily.com

裁剪项（相对原版）：``get_setup_schema``（hermes tools 配置向导用的，
my-hermes 没有，抽象基类也无该方法）。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)


def _tavily_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST 到 Tavily API 并返回解析后的 JSON 响应。

    ``TAVILY_API_KEY`` 未设置时抛 ``ValueError``；调用方捕获并把它呈现为
    类型化错误响应。httpx 默认 trust_env=True，会读取 HTTPS_PROXY 环境变量
    ——国内环境靠 Clash 代理，无需代码特判。
    """
    import httpx

    api_key = get_provider_env("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )

    base_url = get_provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    payload = dict(payload)  # 不修改调用方的 dict
    payload["api_key"] = api_key
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)

    response = httpx.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """把 Tavily ``/search`` 响应映射成 ``{success, data: {web: [...]}}``。"""
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("content", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(
    response: Dict[str, Any], fallback_url: str = ""
) -> List[Dict[str, Any]]:
    """把 Tavily ``/extract`` 响应映射成标准文档列表。

    文档遵循 legacy LLM 后处理形状::

        {"url", "title", "content", "raw_content", "metadata"}

    失败（``failed_results`` / ``failed_urls``）变成带 ``error`` 字段的
    结果条目，而不是抛异常。
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append(
            {
                "url": url,
                "title": result.get("title", ""),
                "content": raw,
                "raw_content": raw,
                "metadata": {"sourceURL": url, "title": result.get("title", "")},
            }
        )
    for fail in response.get("failed_results", []):
        documents.append(
            {
                "url": fail.get("url", fallback_url),
                "title": "",
                "content": "",
                "raw_content": "",
                "error": fail.get("error", "extraction failed"),
                "metadata": {"sourceURL": fail.get("url", fallback_url)},
            }
        )
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append(
            {
                "url": url_str,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "extraction failed",
                "metadata": {"sourceURL": url_str},
            }
        )
    return documents


class TavilyWebSearchProvider(WebSearchProvider):
    """Tavily 搜索 + 提取 provider。"""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def display_name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """``TAVILY_API_KEY`` 非空即可用。"""
        return bool(get_provider_env("TAVILY_API_KEY"))

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """执行一次 Tavily 搜索。"""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Tavily search: '%s' (limit=%d)", query, limit)
            raw = _tavily_request(
                "search",
                {
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
            return _normalize_tavily_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — 含 httpx 异常
            logger.warning("Tavily search error: %s", exc)
            return {"success": False, "error": f"Tavily search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """经 Tavily 从一个或多个 URL 提取内容。

        同步——底层是 httpx.post(...)。返回 legacy 结果列表形状；单 URL
        失败变成带 ``error`` 的条目。
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Tavily extract: %d URL(s)", len(urls))
            raw = _tavily_request(
                "extract",
                {
                    "urls": urls,
                    "include_images": False,
                },
            )
            return _normalize_tavily_documents(
                raw, fallback_url=urls[0] if urls else ""
            )
        except ValueError as exc:
            return [
                {"url": u, "title": "", "content": "", "error": str(exc)}
                for u in urls
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily extract error: %s", exc)
            return [
                {
                    "url": u, "title": "", "content": "",
                    "error": f"Tavily extract failed: {exc}",
                }
                for u in urls
            ]
