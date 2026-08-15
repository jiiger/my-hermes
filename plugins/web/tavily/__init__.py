"""Tavily web 搜索 + 提取插件——内置、自动加载。"""

from __future__ import annotations

from .provider import TavilyWebSearchProvider


def register(ctx) -> None:
    """向插件上下文注册 Tavily provider。"""
    ctx.register_web_search_provider(TavilyWebSearchProvider())
