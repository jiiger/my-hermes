"""Memory provider 插件加载器（精简移植版）。

对应原版 hermes-agent 的 plugins/memory/__init__.py（461 行）。扫描两个
目录寻找 memory provider 插件：

1. 内置 provider：``plugins/memory/<name>/``（随 my-hermes 分发）
2. 用户安装 provider：``$HERMES_HOME/plugins/memory/<name>/``

每个子目录必须包含 ``__init__.py``，其中有一个实现 MemoryProvider ABC
的类（或 register(ctx) 注册函数）。名字冲突时内置优先。

同时只能激活一个 provider，由 config.yaml 的 ``memory.provider`` 选择。

用法：:

    from plugins.memory import discover_memory_providers, load_memory_provider

    available = discover_memory_providers()   # [(name, desc, available), ...]
    provider = load_memory_provider("honcho")  # MemoryProvider 实例

精简版改动（相对原版）：
- 砍掉 discover_plugin_cli_commands（provider 的 CLI 命令注册，my-hermes
  无插件 CLI 命令系统，留注释扩展点）；
- 新增模块级 register_memory_provider 注册表 —— 为未来的完整插件系统
  （原版 hermes_cli/plugins.py 的 PluginContext.register_memory_provider）
  预留注册位置：插件系统接入时把 provider 注册进这里，即可与目录发现
  走同一套加载流程。
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# 内置 provider 目录 = 本文件所在目录（plugins/memory/）
_MEMORY_PLUGINS_DIR = Path(__file__).parent

# 用户安装 provider 的合成父包命名空间，避免与内置 provider 在
# sys.modules 里冲突。
_USER_NAMESPACE = "_hermes_user_memory"


# ══════════════════════════════════════════════════════════════════
# 插件系统注册点（预留）
#
# 原版 hermes-agent 的完整插件系统（hermes_cli/plugins.py）通过
# PluginContext.register_memory_provider 让插件注册 provider（路径 B）。
# my-hermes 尚无插件系统，这里提供同名的模块级注册表：
# 未来插件系统接入时，PluginContext.register_memory_provider 可直接委托
# 到本函数，注册的 provider 会与目录发现走同一套加载/查找流程。
# ══════════════════════════════════════════════════════════════════

_plugin_registered_providers: dict[str, Callable[[], "object"]] = {}


def register_memory_provider(
    name: str, provider_factory: Callable[[], "object"]
) -> None:
    """注册一个 memory provider 工厂（插件系统扩展点）。

    provider_factory 是无参可调用对象，返回 MemoryProvider 实例。
    注册的 provider 会被 load_memory_provider 优先于目录发现命中。
    """
    _plugin_registered_providers[name] = provider_factory


# ─── 目录辅助 ──────────────────────────────────────────────────────────


def _get_user_plugins_dir() -> Optional[Path]:
    """返回 ``$HERMES_HOME/plugins/memory/``（不存在返回 None）。"""
    try:
        from hermes_constants import get_hermes_home

        d = get_hermes_home() / "plugins" / "memory"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_memory_provider_dir(path: Path) -> bool:
    """启发式：*path* 是否像一个 memory provider 插件目录？

    检查 ``__init__.py`` 源码里是否含 ``register_memory_provider`` 或
    ``MemoryProvider``。廉价文本扫描，不 import。
    """
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


def _register_synthetic_package(name: str, search_locations: List[str]) -> None:
    """在 sys.modules 里注册一个空的包壳。

    用户安装的 provider 以 ``_hermes_user_memory.<name>`` 导入，其父包在
    磁盘上不存在。若父包不在 sys.modules，插件内的相对导入
    （``from . import config``）会因 ModuleNotFoundError 失败——与内置
    provider 注册 ``plugins`` / ``plugins.memory`` 父包同理。
    """
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = search_locations
    sys.modules[name] = importlib.util.module_from_spec(spec)


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """产出所有发现的 ``(name, path)`` provider 目录。

    先扫内置，再扫用户安装。名字冲突时内置优先（seen 集合首见者胜）。
    """
    seen: set = set()
    dirs: List[Tuple[str, Path]] = []

    # 1. 内置 provider（plugins/memory/<name>/）
    if _MEMORY_PLUGINS_DIR.is_dir():
        for child in sorted(_MEMORY_PLUGINS_DIR.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if not (child / "__init__.py").exists():
                continue
            seen.add(child.name)
            dirs.append((child.name, child))

    # 2. 用户安装 provider（$HERMES_HOME/plugins/memory/<name>/）
    user_dir = _get_user_plugins_dir()
    if user_dir:
        for child in sorted(user_dir.iterdir()):
            if not child.is_dir() or child.name.startswith(("_", ".")):
                continue
            if child.name in seen:
                continue  # 内置优先
            if not _is_memory_provider_dir(child):
                continue  # 跳过非 memory 插件
            dirs.append((child.name, child))

    return dirs


def find_provider_dir(name: str) -> Optional[Path]:
    """把 provider 名解析到目录（先内置后用户）。"""
    bundled = _MEMORY_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    user_dir = _get_user_plugins_dir()
    if user_dir:
        user = user_dir / name
        if user.is_dir() and _is_memory_provider_dir(user):
            return user
    return None


# ─── 公开 API ──────────────────────────────────────────────────────────


def list_memory_provider_names() -> List[str]:
    """廉价的名字列表：纯目录扫描，不 import 也不跑可用性检查。"""
    return sorted({name for name, _ in _iter_provider_dirs()})


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """扫描内置 + 用户目录，返回 ``[(name, description, is_available)]``。"""
    results = []

    for name, child in _iter_provider_dirs():
        # 从 plugin.yaml 读描述（如有）
        desc = ""
        yaml_file = child / "plugin.yaml"
        if yaml_file.exists():
            try:
                import yaml

                with open(yaml_file, encoding="utf-8-sig") as f:
                    meta = yaml.safe_load(f) or {}
                desc = meta.get("description", "")
            except Exception:
                pass

        # 快速可用性检查：加载并调 is_available()
        available = True
        try:
            provider = _load_provider_from_dir(child)
            if provider:
                available = provider.is_available()
            else:
                available = False
        except Exception:
            available = False

        results.append((name, desc, available))

    return results


def load_memory_provider(name: str) -> Optional["object"]:
    """按名字加载并返回 MemoryProvider 实例。

    先查插件系统注册表（register_memory_provider），再查内置/用户目录。
    未找到或加载失败返回 None。
    """
    factory = _plugin_registered_providers.get(name)
    if factory is not None:
        try:
            provider = factory()
            if provider is not None:
                return provider
            logger.warning(
                "Memory provider '%s' factory returned None; "
                "falling back to directory discovery",
                name,
            )
        except Exception as e:
            logger.warning(
                "Memory provider '%s' factory failed (%s); "
                "falling back to directory discovery",
                name,
                e,
            )
        # 注册表失败 → 回退到目录发现（best-effort，同原版加载容错精神）

    provider_dir = find_provider_dir(name)
    if not provider_dir:
        logger.debug(
            "Memory provider '%s' not found in bundled or user plugins", name
        )
        return None

    try:
        provider = _load_provider_from_dir(provider_dir)
        if provider:
            return provider
        logger.warning(
            "Memory provider '%s' loaded but no provider instance found", name
        )
        return None
    except Exception as e:
        logger.warning("Failed to load memory provider '%s': %s", name, e)
        return None


def _load_provider_from_dir(provider_dir: Path) -> Optional["object"]:
    """import provider 模块并提取 MemoryProvider 实例。

    模块必须满足其一：
    - register(ctx) 函数（插件风格）——用模拟 ctx 收集 provider；
    - 顶层继承 MemoryProvider 的类——直接实例化。

    TODO 插件系统扩展点：未来 my-hermes 接入完整插件系统（原版
    hermes_cli/plugins.py）后，这里传给 register() 的 ctx 可替换为真实
    PluginContext 实例（其 register_memory_provider 委托到本模块的
    register_memory_provider 注册表），两种加载路径即可统一。
    """
    name = provider_dir.name
    # 用户安装插件用独立命名空间，避免与内置在 sys.modules 冲突
    is_bundled = (
        _MEMORY_PLUGINS_DIR in provider_dir.parents
        or provider_dir.parent == _MEMORY_PLUGINS_DIR
    )
    module_name = f"plugins.memory.{name}" if is_bundled else f"{_USER_NAMESPACE}.{name}"
    init_file = provider_dir / "__init__.py"

    if not init_file.exists():
        return None

    cached = sys.modules.get(module_name)
    if cached is not None and getattr(cached, "__file__", None):
        mod = cached
    else:
        # 处理插件内相对导入：先注册父包
        for parent in ("plugins", "plugins.memory"):
            if parent not in sys.modules:
                parent_path = Path(__file__).parent
                if parent == "plugins":
                    parent_path = parent_path.parent
                parent_init = parent_path / "__init__.py"
                if parent_init.exists():
                    spec = importlib.util.spec_from_file_location(
                        parent,
                        str(parent_init),
                        submodule_search_locations=[str(parent_path)],
                    )
                    if spec:
                        parent_mod = importlib.util.module_from_spec(spec)
                        sys.modules[parent] = parent_mod
                        try:
                            spec.loader.exec_module(parent_mod)
                        except Exception:
                            pass

        # 用户安装插件需要合成父包
        if not is_bundled:
            _register_synthetic_package(_USER_NAMESPACE, [])

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(init_file),
            submodule_search_locations=[str(provider_dir)],
        )
        if not spec:
            return None

        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod

        # 注册子模块，让相对导入（如 "from .store import MemoryStore"）可用
        for sub_file in provider_dir.glob("*.py"):
            if sub_file.name == "__init__.py":
                continue
            sub_name = sub_file.stem
            full_sub_name = f"{module_name}.{sub_name}"
            if full_sub_name not in sys.modules:
                sub_spec = importlib.util.spec_from_file_location(
                    full_sub_name, str(sub_file)
                )
                if sub_spec:
                    sub_mod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[full_sub_name] = sub_mod
                    try:
                        sub_spec.loader.exec_module(sub_mod)
                    except Exception as e:
                        logger.debug(
                            "Failed to load submodule %s: %s", full_sub_name, e
                        )

        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            logger.debug("Failed to exec_module %s: %s", module_name, e)
            sys.modules.pop(module_name, None)
            return None

    # 先试 register(ctx) 模式（原版插件写法）
    if hasattr(mod, "register"):
        collector = _ProviderCollector()
        try:
            mod.register(collector)
            if collector.provider:
                return collector.provider
        except Exception as e:
            logger.debug("register() failed for %s: %s", name, e)

    # 兜底：找 MemoryProvider 子类并实例化
    try:
        from agent.memory_provider import MemoryProvider
    except Exception:
        return None
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name, None)
        if (
            isinstance(attr, type)
            and issubclass(attr, MemoryProvider)
            and attr is not MemoryProvider
        ):
            try:
                return attr()
            except Exception:
                pass

    return None


class _ProviderCollector:
    """模拟插件上下文的收集器，捕获 register_memory_provider 调用。

    TODO 插件系统扩展点：未来接入完整插件系统后，本类可替换/委托为真实
    PluginContext（register_memory_provider 落到本模块注册表）。
    """

    def __init__(self):
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider

    # 其他注册方法 no-op（my-hermes 尚无对应系统）
    def register_tool(self, *args, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass


def _get_active_memory_provider() -> Optional[str]:
    """从 config.yaml 读激活的 memory provider 名（轻量，不加载插件）。"""
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        return (config.get("memory", {}) or {}).get("provider") or None
    except Exception:
        return None
