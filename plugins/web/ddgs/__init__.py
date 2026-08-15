"""DuckDuckGo 搜索插件——内置、自动加载。

基于社区 ``ddgs`` Python 包抓取 DuckDuckGo HTML 结果页。无需 API key，
但包本身必须安装（可选依赖——经 :meth:`is_available` 门控）。
"""

from __future__ import annotations

from plugins.web.ddgs.provider import DDGSWebSearchProvider


def register(ctx) -> None:
    """向插件上下文注册 DDGS provider。"""
    ctx.register_web_search_provider(DDGSWebSearchProvider())
