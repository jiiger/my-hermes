"""DuckDuckGo 搜索插件 provider（精简移植版）。

对应原版 hermes-agent 的 plugins/web/ddgs/provider.py。my-hermes 学习版
裁剪了子进程隔离机制（原版约 490 行含 _search_worker.py 与
_run_ddgs_search_bounded：防 ddgs/primp 原生代码持 GIL 卡死整个进程）；
本版直接调用 ``ddgs.DDGS().text()``，单次请求 timeout=10，以后需要再补
子进程隔离。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# 单次 HTTP 请求超时（秒）。DDGS 构造器的 timeout 只约束单个请求，
# ddgs 的多引擎重试循环没有总上限——my-hermes 直接调用，接受该风险。
_SEARCH_TIMEOUT_SECS = 10


def _run_ddgs_search(query: str, safe_limit: int) -> list[dict[str, Any]]:
    """执行阻塞式 ddgs 查询并返回归一化命中。

    模块级函数（非闭包），便于测试 monkeypatch。
    """
    from ddgs import DDGS  # 可选依赖，调用期导入

    results: list[dict[str, Any]] = []
    with DDGS(timeout=_SEARCH_TIMEOUT_SECS) as client:
        for i, hit in enumerate(client.text(query, max_results=safe_limit)):
            if i >= safe_limit:
                break
            url = str(hit.get("href") or hit.get("url") or "")
            results.append({
                "title": str(hit.get("title", "")),
                "url": url,
                "description": str(hit.get("body", "")),
                "position": i + 1,
            })
    return results


class DDGSWebSearchProvider(WebSearchProvider):
    """DuckDuckGo HTML 抓取搜索 provider（无需 API key）。

    rate limit 由 DuckDuckGo 服务端强制；provider 把
    DuckDuckGoSearchException 等 ddgs 错误归一化成
    ``{"success": False, "error": ...}`` 而非抛出。
    """

    @property
    def name(self) -> str:
        return "ddgs"

    @property
    def display_name(self) -> str:
        return "DuckDuckGo (ddgs)"

    def is_available(self) -> bool:
        """ddgs 包可导入即可用。廉价探针，禁止网络 I/O。"""
        try:
            import ddgs  # noqa: F401

            return True
        except ImportError:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """执行 DuckDuckGo 搜索并返回归一化结果。"""
        try:
            import ddgs  # noqa: F401 — 可用性探针
        except ImportError:
            return {
                "success": False,
                "error": "ddgs package is not installed — run `pip install ddgs`",
            }

        safe_limit = max(1, int(limit))

        try:
            web_results = _run_ddgs_search(query, safe_limit)
        except Exception as exc:  # noqa: BLE001 — ddgs 抛自己的异常族
            logger.warning("DDGS search error: %s", exc)
            return {"success": False, "error": f"DuckDuckGo search failed: {exc}"}

        logger.info(
            "DDGS search '%s': %d results (limit %d)",
            query, len(web_results), limit,
        )
        return {"success": True, "data": {"web": web_results}}
