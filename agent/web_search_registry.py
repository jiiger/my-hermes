"""Web 搜索 Provider 注册表（精简移植版）。

对应原版 hermes-agent 的 agent/web_search_registry.py（343 行）。
插件在导入期经 :meth:`PluginContext.register_web_search_provider` 把
provider 填进这张中央表；``tools.web_tools`` 里的 ``web_search`` /
``web_extract`` 工具包装层查表把每次调用派发给活动后端。

活动 provider 选择优先级（对应原版模块 docstring）：

1. ``web.search_backend`` / ``web.extract_backend``（按能力覆盖）；
2. ``web.backend``（共享回退）；
3. 恰好一个能力合格且可用的已注册 provider 时用它；
4. legacy 偏好序（firecrawl → parallel → tavily → exa → searxng →
   brave-free → ddgs）按可用性过滤；
5. 否则 None——工具层给出"去配置 provider"的错误。

每一步都套 :meth:`supports_search` / :meth:`supports_extract` 能力过滤，
搜索专用后端（如 ddgs）被配成 ``web.extract_backend`` 时能正确落到
可提取的后端上。

裁剪项（相对原版）：``scope`` 参数（多 profile 隔离，my-hermes 单进程）、
``snapshot_registration`` / ``restore_registration``（插件热重载替换，
my-hermes 的 ``_track`` 记账 + 本表 ``_unregister_provider`` 已覆盖）、
``_disabled_web_plugin_for``（my-hermes 无插件禁用机制）。
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


_providers: Dict[str, WebSearchProvider] = {}
_lock = threading.Lock()


def register_provider(provider: WebSearchProvider) -> None:
    """注册一个 web 搜索/提取 provider。

    同名重新注册覆盖旧条目并记 debug 日志——让热重载场景（测试、开发
    循环）行为可预期。
    """
    if not isinstance(provider, WebSearchProvider):
        raise TypeError(
            f"register_provider() expects a WebSearchProvider instance, "
            f"got {type(provider).__name__}"
        )
    raw_name = provider.name
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise ValueError("Web provider .name must be a non-empty string")
    name = raw_name.strip()
    with _lock:
        existing = _providers.get(name)
        _providers[name] = provider
    if existing is not None:
        logger.debug(
            "Web provider '%s' re-registered (was %r)",
            name, type(existing).__name__,
        )
    else:
        logger.debug(
            "Registered web provider '%s' (%s)",
            name, type(provider).__name__,
        )


def _unregister_provider(name: str) -> None:
    """移除指定名字的 provider（内部，供插件卸载 release 回调使用）。"""
    key = name.strip() if isinstance(name, str) else ""
    if not key:
        return
    with _lock:
        _providers.pop(key, None)


def list_providers() -> List[WebSearchProvider]:
    """返回全部已注册 provider，按 name 排序。"""
    with _lock:
        items = list(_providers.values())
    return sorted(items, key=lambda p: p.name)


def get_provider(name: str) -> Optional[WebSearchProvider]:
    """返回注册在 *name* 下的 provider，未注册返回 None。"""
    if not isinstance(name, str):
        return None
    with _lock:
        return _providers.get(name.strip())


# ---------------------------------------------------------------------------
# 活动 provider 解析
# ---------------------------------------------------------------------------


def _read_config_key(*path: str) -> Optional[str]:
    """从 config.yaml 解析点分配置键，miss 返回 None。"""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        cur = cfg
        for segment in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(segment)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()
    except Exception as exc:  # noqa: BLE001 — 配置读取 best-effort
        logger.debug("Could not read config %s: %s", ".".join(path), exc)
    return None


# legacy 偏好序：未设置任何 web.backend 键时保持历史候选顺序
# （付费 provider 优先，已有付费配置的安装不会在升级时降级到免费档）。
_LEGACY_PREFERENCE = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "ddgs",
)


def _resolve(configured: Optional[str], *, capability: str) -> Optional[WebSearchProvider]:
    """为某个能力（"search" | "extract"）解析活动 provider。

    解析规则（依次）：

    1. **显式配置优先，无视可用性**。``web.{capability}_backend`` 或
       ``web.backend`` 指名了一个支持该能力的已注册 provider 时直接返回，
       即使其 :meth:`is_available` 返回 False——派发方会给用户精确的
       "X_API_KEY 未设置"错误，而不是静默换后端。
    2. **单 provider 捷径**。只有一个已注册 provider 支持该能力且
       ``is_available()`` 为 True 时返回它。
    3. **legacy 偏好序走查**，按可用性过滤（firecrawl → ... → ddgs）。
    4. 都没有 → None。
    """
    with _lock:
        snapshot = dict(_providers)

    def _capable(p: WebSearchProvider) -> bool:
        if capability == "search":
            return bool(p.supports_search())
        if capability == "extract":
            return bool(p.supports_extract())
        return False

    def _is_available_safe(p: WebSearchProvider) -> bool:
        """包一层 is_available()，坏 provider 不能杀死解析。"""
        try:
            return bool(p.is_available())
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider %s.is_available() raised %s", p.name, exc)
            return False

    # 1. 显式配置优先——无视 is_available()，让用户拿到精确的下游错误。
    if configured:
        provider = snapshot.get(configured)
        if provider is not None and _capable(provider):
            return provider
        if provider is None:
            logger.debug(
                "web backend '%s' configured but not registered; falling back",
                configured,
            )
        else:
            logger.debug(
                "web backend '%s' configured but does not support '%s'; falling back",
                configured, capability,
            )

    # 2. + 3. 回退路径——按可用性过滤，避免裸注册未配置的 provider
    #    在全新安装（无任何 API key）时被当成"活动"。
    eligible = [
        p for p in snapshot.values()
        if _capable(p) and _is_available_safe(p)
    ]
    if len(eligible) == 1:
        return eligible[0]

    for legacy in _LEGACY_PREFERENCE:
        provider = snapshot.get(legacy)
        if (
            provider is not None
            and _capable(provider)
            and _is_available_safe(provider)
        ):
            return provider

    return None


def get_active_search_provider() -> Optional[WebSearchProvider]:
    """解析当前活动的 web 搜索 provider。

    读 ``web.search_backend``（优先）或 ``web.backend``（共享回退）；
    按模块 docstring 回退。
    """
    explicit = _read_config_key("web", "search_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="search")


def get_active_extract_provider() -> Optional[WebSearchProvider]:
    """解析当前活动的 web 提取 provider。

    读 ``web.extract_backend``（优先）或 ``web.backend``（共享回退）。
    """
    explicit = _read_config_key("web", "extract_backend") or _read_config_key("web", "backend")
    return _resolve(explicit, capability="extract")


def _reset_for_tests() -> None:
    """清空注册表。**仅测试用**。"""
    with _lock:
        _providers.clear()
