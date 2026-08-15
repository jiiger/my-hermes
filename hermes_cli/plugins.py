"""插件系统（精简移植版，对应原版 hermes_cli/plugins.py，6318 行）。

my-hermes 的插件体系：四层架构 ``PluginManifest → PluginContext →
PluginManager → 模块级入口``，形成"发现 → 加载 → 注册 → 消费"完整链。

相对原版砍掉的内容：
- entry_points 打包发现（只做目录发现）；
- 全部 register_*_provider（image_gen/video_gen/web_search/browser/tts/
  transcription/dashboard_auth/secret_source/context_engine/platform/
  slack/approval_transport）；
- inject_message / platform_actions / 预工具调用指令（resolve_pre_tool_block 等）；
- 插件错误分类 / 调试 handler（_install_plugin_debug_handler）/ 技能命名空间 /
  便携 MCP 注册 / CLI 斜杠命令注册（register_command / register_cli_command）；
- on_session_finalize / on_session_reset 钩子。

保留的内容：
- 事件总线（emit/subscribe，同步派发精简版，事件名强制加 ``<plugin>:`` 前缀）；
- 系统提示词段渲染（render_system_prompt_sections + PluginSystemPromptSection
  + format_system_prompt_section / format_system_prompt_sections）；
- 中间件接口（invoke_middleware / has_middleware / register_middleware）；
- 钩子集（只保留 5 个）：on_session_start / pre_api_request / post_api_request /
  pre_tool_call / on_session_end。
"""

import copy
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

import yaml

from hermes_cli.config import load_config_readonly
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════

# 钩子集：只保留 5 个（原版 plugins.py:156 VALID_HOOKS 有数十个，其余不接）。
VALID_HOOKS: Set[str] = {
    "on_session_start",
    "pre_api_request",
    "post_api_request",
    "pre_tool_call",
    "on_session_end",
}

# 中间件种类（对齐原版 hermes_cli/middleware.py:29 VALID_MIDDLEWARE）。
VALID_MIDDLEWARE: Set[str] = {
    "tool_request",
    "tool_execution",
    "llm_request",
    "llm_execution",
}
OBSERVER_SCHEMA_VERSION = "hermes.observer.v1"

# 系统提示词段（原版 plugins.py:495-503）。
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"

# manifest 版本（原版 plugins.py:665 SUPPORTED_MANIFEST_VERSION = 2）。
SUPPORTED_MANIFEST_VERSION = 2

# 合法插件 kind（原版 plugins.py:620）。
_VALID_PLUGIN_KINDS: Set[str] = {
    "standalone",
    "backend",
    "exclusive",
    "platform",
    "model-provider",
}

# 事件总线：互发事件的递归深度上限，防止两个插件互相 emit 死循环。
_EVENT_EMIT_DEPTH_CAP = 8
# 单插件状态配额（原版 plugins.py:1241 _PLUGIN_STATE_QUOTA_BYTES）。
_PLUGIN_STATE_QUOTA_BYTES = 10 * 1024 * 1024

# 目录插件导入用的父包命名空间（原版 plugins.py:513 _NS_PARENT）。
_NS_PARENT = "hermes_plugins"
_MODULE_NAMESPACE_LOCK = threading.RLock()
_BARE_MODULE_SCOPE: Dict[str, str] = {}


# ══════════════════════════════════════════════════════════════════
# 模块级辅助
# ══════════════════════════════════════════════════════════════════


def get_bundled_plugins_dir() -> Path:
    """定位内置 plugins/ 目录（对应原版 plugins.py:76）。

    支持 ``HERMES_BUNDLED_PLUGINS`` 环境变量覆盖（打包安装场景）；
    缺省回落到仓库内 ``<项目根>/plugins``（开发场景）。
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """*value* 是否是稳定、可作标题的段 id（对应原版 plugins.py:506）。"""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """渲染一段可审计、长度有标注的提示词块（对应原版 plugins.py:511）。"""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n"
        f"{content}"
    )


def format_system_prompt_sections(sections: list) -> str:
    """把若干段拼成带起止标记的容器（对应原版 plugins.py:520）。"""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


def _get_disabled_plugins() -> set:
    """读 config.yaml 的 ``plugins.disabled`` 黑名单（对应原版 plugins.py:570）。

    黑名单中的插件永远不加载，即使同时出现在 enabled 白名单里。
    """
    try:
        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, Mapping) else None
        disabled = plugins.get("disabled") if isinstance(plugins, Mapping) else None
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """读 config.yaml 的 ``plugins.enabled`` 白名单（对应原版 plugins.py:586）。

    返回 None 表示配置未声明——my-hermes 语义为"全开"（无白名单限制）；
    返回空集合表示显式禁用全部；返回集合则是具体白名单。
    """
    try:
        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, Mapping) else None
        if not isinstance(plugins, Mapping):
            return None
        if "enabled" not in plugins:
            return None
        enabled = plugins.get("enabled")
        return set(enabled) if isinstance(enabled, list) else None
    except Exception:
        return None


def _parse_manifest_v2_fields(data: Mapping, key: str) -> Dict[str, Any]:
    """校验并归一化 manifest v2 字段（对应原版 plugins.py:683 精简版）。

    只保留 hooks 列表与基本元数据；未知字段一律忽略并记 warning，
    绝不因 v2 附加字段导致加载失败（v2 元数据是建议性、附加性的）。
    """
    out: Dict[str, Any] = {}

    # manifest_version——缺省按 v1（永久支持）。
    raw_mv = data.get("manifest_version", 1)
    try:
        mv = int(raw_mv)
    except (TypeError, ValueError):
        logger.warning("Plugin %s: manifest_version %r 非整数，按 1 处理", key, raw_mv)
        mv = 1
    if mv > SUPPORTED_MANIFEST_VERSION:
        logger.warning(
            "Plugin %s: manifest_version %d 高于本版本支持上限 %d，未知字段将被忽略",
            key,
            mv,
            SUPPORTED_MANIFEST_VERSION,
        )
    out["manifest_version"] = mv

    # hooks：manifest 声明插件会注册的钩子（展示用，非强制约束）。
    raw_hooks = data.get("hooks")
    if raw_hooks is not None and isinstance(raw_hooks, list):
        out["hooks"] = [str(h) for h in raw_hooks if isinstance(h, str)]
    else:
        if raw_hooks is not None:
            logger.warning("Plugin %s: hooks 必须是列表；已忽略", key)
        out["hooks"] = []

    # requires_env：插件要求的环境变量（原版仅记录，不强制校验）。
    raw_req_env = data.get("requires_env")
    if raw_req_env is not None and not isinstance(raw_req_env, list):
        logger.warning("Plugin %s: requires_env 必须是列表；已忽略", key)
        raw_req_env = None
    out["requires_env"] = list(raw_req_env or [])

    # 标准元数据（宽松读取，非字符串强转字符串）。
    out["license"] = str(data.get("license") or "")
    out["homepage"] = str(data.get("homepage") or "")
    raw_tags = data.get("tags")
    out["tags"] = [str(t) for t in raw_tags] if isinstance(raw_tags, list) else []

    return out


# ══════════════════════════════════════════════════════════════════
# 数据类
# ══════════════════════════════════════════════════════════════════


@dataclass
class PluginManifest:
    """plugin.yaml 的解析结果（对应原版 plugins.py:1031 精简版）。"""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    # 插件 kind：standalone（默认）/ backend / exclusive / platform / model-provider。
    # memory provider 家族走 plugins/memory 的专属发现，通用加载器跳过 exclusive。
    kind: str = "standalone"
    # 注册键——路径派生（如 image_gen/openai），空则回落到 name。
    key: str = ""
    # 来源："bundled"（内置）/ "user"（$HERMES_HOME/plugins）。
    source: str = ""
    path: Optional[str] = None
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    # manifest 声明的 hooks 列表（展示用，非强制）。
    hooks: List[str] = field(default_factory=list)
    # 加载顺序权重：数字小者先加载（同权重按 key 字母序）。
    load_order: int = 0
    manifest_version: int = 1
    license: str = ""
    homepage: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """插件注册的系统提示词段（对应原版 plugins.py:1110）。"""

    id: str
    content: Union[str, Callable[[Mapping[str, Any]], str]]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """渲染后冻结的提示词段（对应原版 plugins.py:1121）。"""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """事件订阅登记项（对应原版 plugins.py:1131）。"""

    owner: str
    callback: Callable


@dataclass
class LoadedPlugin:
    """单个已加载插件的运行态（对应原版 plugins.py:1150 精简版）。"""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    middleware_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None


@dataclass
class PluginRegistration:
    """宿主持有的一次注册（对应原版 plugins.py:1187 精简版）。

    插件只拿到上下文注册 API；管理器持有对应的反操作（release）。
    卸载时按反序 dispose 每个 handle；重复 dispose 无害。
    """

    kind: str
    key: str
    release: Callable[[], None]
    plugin_key: str = ""
    _disposed: bool = field(default=False, init=False, repr=False)
    _on_dispose: Optional[Callable[["PluginRegistration"], None]] = field(
        default=None, init=False, repr=False
    )

    @property
    def active(self) -> bool:
        """该 handle 是否仍持有一份活跃注册。"""
        return not self._disposed

    def dispose(self) -> None:
        """释放本次注册（仅一次；重复释放无害）。"""
        if self._disposed:
            return
        self._disposed = True
        try:
            self.release()
        finally:
            if self._on_dispose is not None:
                self._on_dispose(self)


# ══════════════════════════════════════════════════════════════════
# PluginState——原子、配额受限的 JSON 键值状态
# ══════════════════════════════════════════════════════════════════

_PLUGIN_STATE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PLUGIN_STATE_LOCKS: Dict[str, threading.RLock] = {}
_PLUGIN_STATE_LOCKS_GUARD = threading.Lock()


def _locked_plugin_state(path: Path) -> threading.RLock:
    """按文件路径复用的进程内重入锁（对应原版 plugins.py:1281）。"""
    with _PLUGIN_STATE_LOCKS_GUARD:
        lock = _PLUGIN_STATE_LOCKS.get(str(path))
        if lock is None:
            lock = threading.RLock()
            _PLUGIN_STATE_LOCKS[str(path)] = lock
        return lock


class PluginState:
    """原子、配额受限的 JSON 键值状态（对应原版 plugins.py:1315 精简版）。

    每个插件一份 profile 作用域的 ``state.json``；读写持文件级重入锁，
    写入超过配额抛 ValueError。写入走 utils.atomic_write_text（临时文件
    + 原子重命名），进程崩溃也不会留下半写入文件。
    """

    def __init__(self, plugin_id: str) -> None:
        self._plugin_id = plugin_id

    @property
    def data_dir(self) -> Path:
        """profile 作用域的数据目录（对应原版 plugins.py:1322）。"""
        return get_hermes_home() / "plugin-data" / self._plugin_id

    @property
    def path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def quota_bytes(self) -> int:
        return _PLUGIN_STATE_QUOTA_BYTES

    @staticmethod
    def _validate_key(key: str) -> None:
        if (
            not isinstance(key, str)
            or not _PLUGIN_STATE_KEY_RE.fullmatch(key)
            or ".." in key
        ):
            raise ValueError(
                "Plugin state keys must be 1-128 characters using letters, "
                "numbers, '_', '-', '.', or ':' (without '..')"
            )

    def _read_unlocked(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Cannot parse plugin state {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Cannot parse plugin state {self.path}: root must be an object"
            )
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """读一个 JSON 值，键缺失时返回 *default*。"""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            return self._read_unlocked().get(key, default)

    def set(self, key: str, value: Any) -> None:
        """原子地写入一个 JSON 值，不丢失并发更新。"""
        self._validate_key(key)
        with _locked_plugin_state(self.path):
            data = self._read_unlocked()
            data[key] = value
            try:
                encoded = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Plugin state value for {key!r} is not JSON-serializable"
                ) from exc
            if len(encoded) > self.quota_bytes:
                raise ValueError(
                    f"Plugin state quota exceeded: {len(encoded)} bytes is greater "
                    f"than the {self.quota_bytes}-byte per-plugin quota"
                )
            from utils import atomic_write_text

            atomic_write_text(self.path, encoded.decode("utf-8"), encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# PluginContext——交给每个插件 register() 的门面
# ══════════════════════════════════════════════════════════════════


class PluginContext:
    """插件上下文门面（对应原版 plugins.py:1388 精简版）。

    只保留 11 个注册/查询方法：
    get_config / set_config / state / on_unload / register_tool /
    register_memory_provider / register_hook / register_system_prompt_section /
    emit / subscribe / register_middleware，
    外加 plugin_id / has_plugin 两个只读查询。
    """

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        self._state: Optional[PluginState] = None

    @property
    def plugin_id(self) -> str:
        """插件的有效注册键（key 优先，空则用 name）。"""
        return self.manifest.key or self.manifest.name

    def has_plugin(self, plugin_id: str) -> bool:
        """另一个同名插件是否已加载并启用（对应原版 plugins.py:1406）。"""
        for key, loaded in self._manager._plugins.items():
            if not loaded.enabled:
                continue
            if key == plugin_id or loaded.manifest.name == plugin_id:
                return True
        return False

    # -- 命名空间配置与持久化状态 ----------------------------------------

    def get_config(self, key: str, default: Any = None) -> Any:
        """读 ``plugins.entries.<plugin_id>.settings.<key>``（对应原版 :1422）。

        优先返回进程内 override（set_config 写入）；其次读 config.yaml 的
        settings 子树；再回退到旧式 config 子树。key 相对插件命名空间。
        """
        overrides = self._manager._plugin_config_overrides.get(self.plugin_id)
        if overrides is not None:
            value = overrides
            for segment in key.split("."):
                if not isinstance(value, dict) or segment not in value:
                    break
                value = value[segment]
            else:
                return value

        config = load_config_readonly() or {}
        entries = config.get("plugins") or {}
        entries = entries.get("entries") if isinstance(entries, dict) else None
        entry = entries.get(self.plugin_id) if isinstance(entries, dict) else None
        if not isinstance(entry, dict):
            return default
        settings = entry.get("settings")
        if isinstance(settings, dict):
            value = settings
            for segment in key.split("."):
                if not isinstance(value, dict) or segment not in value:
                    break
                value = value[segment]
            else:
                return value
        legacy = entry.get("config")
        if isinstance(legacy, dict):
            return legacy.get(key, default)
        return default

    def set_config(self, key: str, value: Any) -> None:
        """写入 ``settings`` 子树的一个值（对应原版 :1450 精简版）。

        my-hermes 的 config 层是纯读取（无 save_config），因此本方法只更新
        进程内 override（get_config 优先读取），重启后失效；不写回磁盘。
        原版会持久化到 config.yaml，这是精简版差异。
        """
        overrides = self._manager._plugin_config_overrides.setdefault(
            self.plugin_id, {}
        )
        segments = key.split(".")
        cursor = overrides
        for segment in segments[:-1]:
            child = cursor.get(segment)
            if not isinstance(child, dict):
                child = {}
                cursor[segment] = child
            cursor = child
        cursor[segments[-1]] = value

    @property
    def state(self) -> PluginState:
        """本插件的 profile 作用域持久 JSON 状态门面（对应原版 :1507）。"""
        if self._state is None:
            self._state = PluginState(self.plugin_id)
        return self._state

    # -- 生命周期：卸载回调 ----------------------------------------------

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """注册一个插件卸载时执行的清理回调（对应原版 :1623）。

        回调按反序与注册清理交错执行，每个回调异常被隔离（只记 warning，
        不中断其他清理）。
        """
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track(
            "on_unload", getattr(callback, "__name__", "callback"), callback
        )
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    # -- 工具注册 --------------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
    ) -> Optional[PluginRegistration]:
        """注册一个工具到全局注册表并跟踪为插件提供（对应原版 :1700 精简版）。

        委托 tools/registry.py 的 registry.register（函数内 import 避免循环
        依赖）；卸载时 deregister。my-hermes 精简版无 override 授权门禁，
        同名直接覆盖（与 registry 精简版语义一致）。
        """
        from tools.registry import registry

        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env,
            is_async=is_async,
            description=description,
            emoji=emoji,
        )

        def _release() -> None:
            registry.deregister(name)

        return self._track("tool", name, _release)

    # -- memory provider 注册 --------------------------------------------

    def register_memory_provider(self, name: str, provider_factory: Callable) -> None:
        """注册一个 memory provider（委托 plugins/memory 注册表）。

        对齐 my-hermes ``plugins/memory/__init__.py:60`` 预留的模块级注册表
        （name → 无参工厂）。注册的 provider 会被 load_memory_provider 优先
        于目录发现命中。与 my-hermes 注册表签名一致；原版接收 provider 实例，
        这里接收工厂，是本移植版的精简差异。
        """
        from plugins.memory import register_memory_provider as _register

        _register(name, provider_factory)
        logger.debug(
            "Plugin %s registered memory provider: %s",
            self.manifest.name,
            name,
        )

    # -- 钩子 / 系统提示词段 / 事件总线 / 中间件 --------------------------

    def register_hook(self, hook_name: str, callback: Callable) -> PluginRegistration:
        """注册一个生命周期钩子回调（对应原版 :3109 精简版）。

        未知钩子名记 warning 但仍存储，避免破坏前向兼容插件。
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' (valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        callbacks = self._manager._hooks.setdefault(hook_name, [])
        callbacks.append(callback)
        handle = self._track(
            "hook",
            hook_name,
            lambda: self._manager._remove_callback(
                self._manager._hooks, hook_name, callback
            ),
        )
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)
        return handle

    def register_system_prompt_section(
        self,
        id: str,
        content: Union[str, Callable[[Mapping[str, Any]], str]],
        *,
        position: str = "after_memory",
        max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """注册一段冻结进每次新会话系统提示词的受限上下文（对应原版 :3134）。

        可调用对象接收只读的 session-info 映射。段 id 全局唯一，重复注册抛
        ValueError。
        """
        if not is_valid_system_prompt_section_id(id):
            raise ValueError(
                "system prompt section id must be 1-128 lowercase characters "
                "using letters, numbers, '.', '_', or '-'"
            )
        if not isinstance(content, str) and not callable(content):
            raise TypeError(
                "system prompt section content must be a string or callable"
            )
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError(
                "system prompt section position must be one of: "
                + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS))
            )
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS
        ):
            raise ValueError(
                "system prompt section max_chars must be between 1 and "
                f"{MAX_SYSTEM_PROMPT_SECTION_CHARS}"
            )
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(
                f"system prompt section {id!r} is already registered by "
                f"plugin {existing.plugin!r}"
            )
        plugin_id = self.plugin_id
        section = PluginSystemPromptSection(
            id=id,
            content=content,
            position=position,
            max_chars=max_chars,
            plugin=plugin_id,
        )
        self._manager._system_prompt_sections[id] = section

        def _release() -> None:
            if self._manager._system_prompt_sections.get(id) is section:
                self._manager._system_prompt_sections.pop(id, None)

        handle = self._track("system_prompt_section", id, _release)
        logger.debug(
            "Plugin %s registered system prompt section: %s",
            self.manifest.name,
            id,
        )
        return handle

    def emit(self, event: str, payload: Optional[dict] = None) -> int:
        """发布 *event* 给所有订阅者，返回被调度的订阅者数（对应原版 :3213）。

        事件以 ``<plugin_key>:<event>`` 派发——plugin_key 强制为插件自身注册
        键。只接受裸事件名：含 ``':'`` 的名称（含 hermes: 保留前缀与外部
        命名空间）一律 ValueError 拒绝（fail-closed）。

        简化版为同步派发：每个订阅者收到 deepcopy 后的 payload，单回调异常
        隔离（只记 warning）；递归深度超过 _EVENT_EMIT_DEPTH_CAP 时丢弃本次
        emit（记 warning），保证互发插件不会死循环。
        """
        plugin_key = self.plugin_id
        if not event or not isinstance(event, str):
            logger.warning(
                "Plugin '%s' tried to emit an invalid event name %r",
                plugin_key,
                event,
            )
            raise ValueError(
                f"Plugin '{plugin_key}' emit() requires a non-empty event name"
            )
        if ":" in event:
            logger.warning(
                "Plugin '%s' tried to emit namespaced/reserved event '%s' — "
                "a plugin may only emit bare event names under its own '%s:' namespace",
                plugin_key,
                event,
                plugin_key,
            )
            raise ValueError(
                f"Plugin '{plugin_key}' may not emit '{event}': emit only the "
                f"bare event name; the namespace is forced to '{plugin_key}:'"
            )
        if payload is not None and not isinstance(payload, dict):
            raise TypeError(
                f"Plugin '{plugin_key}' emit() payload must be a dict or None"
            )
        full_event = f"{plugin_key}:{event}"
        return self._manager._dispatch_event(full_event, payload or {})

    def subscribe(self, event: str, callback: Callable) -> None:
        """订阅一个全限定事件名（对应原版 :3265）。

        *event* 是完整的 ``<plugin_key>:<event>`` 名称。订阅不限命名空间——
        任何插件可监听任意已发布事件；只有"发布"受命名空间门禁。
        订阅以 owner 标记登记，插件卸载/重载时统一移除，杜绝僵尸回调。
        """
        if not event or not isinstance(event, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' subscribe() requires a "
                f"non-empty event name"
            )
        plugin_key = self.plugin_id
        self._manager._subscribe_event(plugin_key, event, callback)
        logger.debug(
            "Plugin %s subscribed to event: %s",
            self.manifest.name,
            event,
        )

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """注册一个行为改变型中间件回调（对应原版 :3289 精简版）。

        中间件与观察型钩子分离：请求中间件可改写有效 payload，执行中间件可
        包裹真实回调。未知 kind 记 warning 但仍存储（前向兼容）。
        """
        if kind not in VALID_MIDDLEWARE:
            logger.warning(
                "Plugin '%s' registered unknown middleware '%s' (valid: %s)",
                self.manifest.name,
                kind,
                ", ".join(sorted(VALID_MIDDLEWARE)),
            )
        callbacks = self._manager._middleware.setdefault(kind, [])
        callbacks.append(callback)
        handle = self._track(
            "middleware",
            kind,
            lambda: self._manager._remove_callback(
                self._manager._middleware, kind, callback
            ),
        )
        logger.debug(
            "Plugin %s registered middleware: %s",
            self.manifest.name,
            kind,
        )
        return handle

    # -- 注册跟踪（内部） -------------------------------------------------

    def _track(
        self,
        kind: str,
        key: str,
        release: Callable[[], None],
    ) -> PluginRegistration:
        """记录一次宿主持有的注册（卸载时反序清理）。"""
        return self._manager._track_registration(self.manifest, kind, key, release)


# ══════════════════════════════════════════════════════════════════
# PluginManager——发现、加载、注册、消费的中心
# ══════════════════════════════════════════════════════════════════


class PluginManager:
    """发现、加载、注册、调用插件的中心管理器（对应原版 plugins.py:3388 精简版）。

    持有四类表：已加载插件 ``_plugins``、钩子 ``_hooks``、中间件
    ``_middleware``、系统提示词段 ``_system_prompt_sections``，以及事件订阅
    表 ``_subscriptions`` 与注册账本 ``_ownership_ledger``。首次 invoke_hook /
    has_hook / render_system_prompt_sections 等入口会经模块级
    ``_delivery_manager`` 懒触发 discover_and_load。
    """

    def __init__(self, scope_key: Optional[str] = None) -> None:
        # 捕获 home 为不可变作用域：卸载可能在别的 profile 上下文执行，
        # 每个反操作必须命中注册时的原始作用域。
        self.scope_key = scope_key or str(_plugin_home_key())
        self.home_path = Path(self.scope_key)
        self._discovery_lock = threading.RLock()
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._system_prompt_sections: Dict[str, PluginSystemPromptSection] = {}
        # 事件订阅表：{事件名: [_EventSubscription, ...]}，owner 标记便于卸载。
        self._subscriptions: Dict[str, List[_EventSubscription]] = {}
        self._event_lock = threading.RLock()
        # 互发事件递归深度（thread-local，见 emit 说明）。
        self._emit_depth = threading.local()
        self._discovered: bool = False
        # 注册账本：{插件键: [PluginRegistration, ...]} + 全局反序表。
        self._ownership_ledger: Dict[str, List[PluginRegistration]] = {}
        self._registration_order: List[PluginRegistration] = []
        # set_config 的进程内 override（my-hermes config 层只读，见 set_config）。
        self._plugin_config_overrides: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 注册账本内部方法
    # ------------------------------------------------------------------

    def _track_registration(
        self,
        manifest: PluginManifest,
        kind: str,
        key: str,
        release: Callable[[], None],
    ) -> PluginRegistration:
        """记录一次成功的注册到账本（对应原版 plugins.py:3462）。"""
        plugin_key = manifest.key or manifest.name
        registration = PluginRegistration(
            kind=kind,
            key=key,
            release=release,
            plugin_key=plugin_key,
        )
        registration._on_dispose = lambda disposed: self._forget_registrations(
            [disposed]
        )
        self._ownership_ledger.setdefault(plugin_key, []).append(registration)
        self._registration_order.append(registration)
        return registration

    @staticmethod
    def _remove_identity(values: list, target: Any) -> bool:
        """从列表中移除最后一个与 *target* 相同的对象（对应原版 :3485）。"""
        for index in range(len(values) - 1, -1, -1):
            if values[index] is target:
                del values[index]
                return True
        return False

    def _remove_callback(
        self,
        mapping: Dict[str, List[Callable]],
        key: str,
        callback: Callable,
    ) -> None:
        """从钩子/中间件映射中按对象身份移除一个回调（对应原版 :3493）。"""
        callbacks = mapping.get(key)
        if callbacks is None:
            return
        self._remove_identity(callbacks, callback)
        if not callbacks:
            mapping.pop(key, None)

    def _forget_registrations(
        self,
        registrations: List[PluginRegistration],
    ) -> None:
        """从账本中抹掉这些注册（不触发 release；对应原版 :3552）。"""
        if not registrations:
            return
        registration_ids = {id(registration) for registration in registrations}
        self._registration_order = [
            registration
            for registration in self._registration_order
            if id(registration) not in registration_ids
        ]
        for plugin_key, owned in list(self._ownership_ledger.items()):
            remaining = [
                registration
                for registration in owned
                if id(registration) not in registration_ids
            ]
            if remaining:
                self._ownership_ledger[plugin_key] = remaining
            else:
                self._ownership_ledger.pop(plugin_key, None)

    def _dispose_registrations(
        self,
        registrations: List[PluginRegistration],
    ) -> None:
        """按获取反序释放注册，逐个隔离异常（对应原版 :3575）。"""
        for registration in reversed(registrations):
            try:
                registration.dispose()
            except Exception as exc:  # pragma: no cover - 防御性清理
                logger.warning(
                    "Failed to unload plugin registration %s/%s: %s",
                    registration.plugin_key,
                    registration.key,
                    exc,
                )

    # ------------------------------------------------------------------
    # 发现与加载
    # ------------------------------------------------------------------

    def discover_and_load(self, force: bool = False) -> None:
        """扫描所有插件源并加载发现的插件（对应原版 plugins.py:3758 精简版）。

        force=True 先卸载再重扫（长会话里新增/变更插件无需重启即可生效）。
        """
        with self._discovery_lock:
            if self._discovered and not force:
                return
            if force:
                self.unload()
            # 先置位作为重入守卫（插件 register() 可能再触发发现）；扫描抛
            # 异常则复位，避免失败扫描被缓存成"已发现空注册表"。
            self._discovered = True
            try:
                self._discover_and_load_inner()
            except BaseException:
                self._discovered = False
                raise

    def _discover_and_load_inner(self) -> None:
        """真正的发现扫描（对应原版 plugins.py:3865 精简版）。"""
        manifests: List[PluginManifest] = self._collect_directory_manifests()

        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = 无白名单（全开）
        # 键冲突时后者覆盖前者：用户插件优先于内置插件。
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest

        to_load: Dict[str, PluginManifest] = {}
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # 显式禁用永远优先。
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # exclusive（memory provider 家族）走 plugins/memory 专属发现，
            # 通用加载器只记录 manifest、不导入模块。
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "exclusive plugin - activate via memory.provider config"
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (exclusive, handled by memory discovery)",
                    lookup_key,
                )
                continue

            # 其余（standalone / backend / platform / model-provider）按
            # enabled 白名单决定是否加载。
            is_enabled = enabled is None or (
                lookup_key in enabled or manifest.name in enabled
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "not enabled in config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping '%s' (not in plugins.enabled)", lookup_key)
                continue
            to_load[lookup_key] = manifest

        # 按 load_order（数字小者先）+ key 字母序加载，保证确定性。
        for lookup_key in sorted(
            to_load, key=lambda k: (to_load[k].load_order, k)
        ):
            self._load_plugin(to_load[lookup_key])

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    def _collect_directory_manifests(self) -> List[PluginManifest]:
        """收集目录 manifest（对应原版 plugins.py:4038 精简版）。

        扫描内置 ``<项目根>/plugins``（跳过 memory 子目录——memory provider
        家族走 load_memory_provider）与 ``$HERMES_HOME/plugins``。
        """
        manifests: List[PluginManifest] = []

        bundled_dir = get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", bundled_dir)
        bundled = self._scan_directory(
            bundled_dir, source="bundled", skip_names={"memory"}
        )
        manifests.extend(bundled)

        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = self._scan_directory(user_dir, source="user")
        manifests.extend(user_manifests)
        return manifests

    def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """读 *path* 子目录里的 plugin.yaml（对应原版 plugins.py:4149 精简版）。

        支持两种布局混用：
        - Flat：``<root>/<plugin-name>/plugin.yaml``，键为 ``<plugin-name>``；
        - Category：``<root>/<category>/<plugin-name>/plugin.yaml``，键为
          ``<category>/<plugin-name>``，深度最多两级。
        """
        return self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: Optional[Set[str]],
        prefix: str,
        depth: int,
    ) -> List[PluginManifest]:
        """递归扫描实现（对应原版 plugins.py:4174）。"""
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"

            if manifest_file.exists():
                manifest = self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            # 本层无 manifest：若仍在深度上限内，把该目录当作分类命名空间
            # 再往下找一层带 manifest 的子目录。
            if depth >= 1:
                logger.debug(
                    "Skipping %s (no plugin.yaml, depth cap reached)", child
                )
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )
        return manifests

    def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> Optional[PluginManifest]:
        """解析单个 plugin.yaml 为 PluginManifest（对应原版 plugins.py:4260 精简版）。

        解析失败返回 None 并记 warning。kind 缺省 standalone；检测到
        memory provider 源码标记时自动归类 exclusive（路由到专属发现）。
        """
        try:
            data = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                data = {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            kind = str(raw_kind).strip().lower() if isinstance(raw_kind, str) else "standalone"
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key,
                    raw_kind,
                    ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # 自动归类：用户安装的 memory provider → exclusive，走
            # plugins/memory 专属发现而非通用加载器（镜像原版 _detect_kind_from_source）。
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        detected = _detect_kind_from_source(
                            init_file.read_text(
                                errors="replace", encoding="utf-8"
                            )[:8192]
                        )
                        if detected:
                            kind = detected
                    except Exception:
                        pass

            v2_fields = _parse_manifest_v2_fields(data, key)
            raw_load_order = data.get("load_order", 0)
            try:
                load_order = int(raw_load_order)
            except (TypeError, ValueError):
                load_order = 0
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                kind=kind,
                key=key,
                source=source,
                path=str(plugin_dir),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                load_order=load_order,
                **v2_fields,
            )
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", manifest_file, exc)
            return None

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """导入插件模块并调用其 ``register(ctx)``（对应原版 plugins.py:4590 精简版）。

        失败时清理该插件已产生的所有注册与订阅，插件以 enabled=False +
        error 记录在 _plugins，绝不向外抛（发现/加载对主流程失败开放）。
        """
        loaded = LoadedPlugin(manifest=manifest)
        plugin_key = manifest.key or manifest.name
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            plugin_key,
            manifest.source,
            manifest.kind,
            manifest.path,
        )
        try:
            module = self._load_directory_module(manifest)
            loaded.module = module

            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                register_fn(ctx)
                loaded.enabled = True
                owned = [
                    registration
                    for registration in self._registration_order
                    if registration.plugin_key == plugin_key and registration.active
                ]
                loaded.tools_registered = [
                    r.key for r in owned if r.kind == "tool"
                ]
                loaded.hooks_registered = [
                    r.key for r in owned if r.kind == "hook"
                ]
                loaded.middleware_registered = [
                    r.key for r in owned if r.kind == "middleware"
                ]
        except Exception as exc:
            owned = [
                registration
                for registration in self._registration_order
                if registration.plugin_key == plugin_key
            ]
            self._dispose_registrations(owned)
            self._forget_registrations(owned)
            self._remove_plugin_subscriptions(plugin_key)
            loaded.error = str(exc)
            logger.warning("Failed to load plugin '%s': %s", manifest.name, exc)
        self._plugins[plugin_key] = loaded

    def _load_directory_module(
        self,
        manifest: PluginManifest,
    ) -> types.ModuleType:
        """把目录插件作为 ``hermes_plugins.<slug>`` 导入（对应原版 :4793 精简版）。

        slug 由 manifest.key 派生（``image_gen/openai`` → ``image_gen__openai``），
        避免同名分类插件冲突。导入前清理 sys.modules 里同名旧模块及子模块，
        失败时也不留半初始化模块。
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        key = manifest.key or manifest.name
        slug = key.replace("/", "__").replace("-", "_")
        module_name = f"{_NS_PARENT}.{slug}"

        # 清掉同名旧模块及其子模块（force 重载 / 跨 profile 复用 slug 场景）。
        stale_prefix = f"{module_name}."
        for name in [
            n
            for n in sys.modules
            if n == module_name or n.startswith(stale_prefix)
        ]:
            del sys.modules[name]

        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in [
                n
                for n in sys.modules
                if n == module_name or n.startswith(stale_prefix)
            ]:
                del sys.modules[name]
            raise
        return module

    # ------------------------------------------------------------------
    # 钩子调用
    # ------------------------------------------------------------------

    @staticmethod
    def _invoke_hook_callback(callback: Callable, payload: Dict[str, Any]) -> Any:
        """调用一个钩子回调，只传入它声明接受的键（对应原版 plugins.py:4883）。

        钩子 payload 是附加演进的：接收 **kwargs 的回调拿到完整 payload；
        窄签名回调只拿到它声明的位置/关键字参数。
        """
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            # 扩展/内建可调用对象不暴露签名：保持历史行为，全量传入。
            return callback(**payload)

        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return callback(**payload)

        accepted_payload = {
            name: value
            for name, value in payload.items()
            if name in parameters
            and parameters[name].kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        return callback(**accepted_payload)

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """调用 *hook_name* 的全部已注册回调（对应原版 plugins.py:4910）。

        每个回调独立 try/except 隔离——一个插件异常只记 warning，绝不打断
        核心 agent 循环。返回所有非 None 返回值的列表。
        """
        kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        callbacks = self._hooks.get(hook_name, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = self._invoke_hook_callback(cb, kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # ------------------------------------------------------------------
    # 事件总线（同步派发精简版）
    # ------------------------------------------------------------------

    def _subscribe_event(
        self,
        owner: str,
        event: str,
        callback: Callable,
    ) -> None:
        """按注册顺序追加一条 owner 标记的事件订阅（对应原版 plugins.py:4955）。"""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        entry = _EventSubscription(owner=owner, callback=callback)
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(entry)

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """移除 *owner* 的所有订阅并返回数量（对应原版 plugins.py:4968）。"""
        removed = 0
        with self._event_lock:
            for event in list(self._subscriptions):
                entries = self._subscriptions[event]
                retained = [entry for entry in entries if entry.owner != owner]
                removed += len(entries) - len(retained)
                if retained:
                    self._subscriptions[event] = retained
                else:
                    del self._subscriptions[event]
        return removed

    def _dispatch_event(self, event: str, payload: Dict[str, Any]) -> int:
        """同步派发一个事件，返回被调用的订阅者数（对应原版 :5101 同步版）。

        每个订阅者收到 deepcopy 后的 payload，单回调异常隔离；递归深度超过
        _EVENT_EMIT_DEPTH_CAP 时丢弃本次 emit（防互发死循环）。
        """
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit",
                _EVENT_EMIT_DEPTH_CAP,
                event,
            )
            return 0

        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, ()))
        if not subscriptions:
            return 0

        self._emit_depth.value = depth + 1
        try:
            for subscription in subscriptions:
                try:
                    subscription.callback(**copy.deepcopy(payload))
                except Exception as exc:
                    logger.warning(
                        "Event '%s' subscriber %s raised: %s",
                        event,
                        getattr(subscription.callback, "__name__", repr(subscription.callback)),
                        exc,
                    )
        finally:
            self._emit_depth.value = depth
        return len(subscriptions)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def has_hook(self, hook_name: str) -> bool:
        """*hook_name* 是否至少注册了一个回调（对应原版 plugins.py:5152）。"""
        return bool(self._hooks.get(hook_name))

    def iter_hook_callbacks(self, hook_name: str) -> tuple:
        """返回某钩子已注册回调的稳定快照（对应原版 plugins.py:5156）。"""
        return tuple(self._hooks.get(hook_name, ()))

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> List[RenderedPluginSystemPromptSection]:
        """渲染全部已注册段，确定性排序 + 失败开放（对应原版 plugins.py:5160 精简版）。

        逐段做预算检查（数量 / 单段 max_chars / 聚合字符），超限跳过并记
        warning；段渲染异常只记 warning 不阻断其余段。
        """
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: List[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for section_id in sorted(self._system_prompt_sections):
            section = self._system_prompt_sections[section_id]
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped",
                    section.id,
                    MAX_SYSTEM_PROMPT_SECTIONS,
                )
                continue
            try:
                value = (
                    section.content(frozen_info)
                    if callable(section.content)
                    else section.content
                )
            except Exception as exc:
                logger.warning(
                    "Plugin system prompt section %s (%s) raised and was skipped: %s",
                    section.id,
                    section.plugin,
                    exc,
                )
                continue
            if not isinstance(value, str):
                logger.warning(
                    "Plugin system prompt section %s (%s) returned %s, not str; skipped",
                    section.id,
                    section.plugin,
                    type(value).__name__,
                )
                continue
            text = value.strip()
            if not text:
                continue
            if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
                logger.warning(
                    "Plugin system prompt section %s (%s) contained a reserved "
                    "persistence marker and was skipped",
                    section.id,
                    section.plugin,
                )
                continue
            if len(text) > section.max_chars:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded max_chars "
                    "(%d > %d) and was skipped",
                    section.id,
                    section.plugin,
                    len(text),
                    section.max_chars,
                )
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # 段间规范分隔符 ``\n\n``
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped",
                    section.id,
                    section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
                )
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id,
                    content=text,
                    position=section.position,
                    plugin=section.plugin,
                )
            )
            total_chars += rendered_chars
        return rendered

    def has_middleware(self, kind: str) -> bool:
        """*kind* 是否至少注册了一个中间件回调（对应原版 plugins.py:5250）。"""
        return bool(self._middleware.get(kind))

    def invoke_middleware(self, kind: str, **kwargs: Any) -> List[Any]:
        """调用 *kind* 的全部中间件回调（对应原版 plugins.py:5254）。

        每个回调独立隔离；要改变行为的中继按调用方约定返回特定形状。
        """
        callbacks = self._middleware.get(kind, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Middleware '%s' callback %s raised: %s",
                    kind,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    def list_plugins(self) -> List[Dict[str, Any]]:
        """返回已发现插件的可读快照（供测试与诊断，对应原版 :5297 精简版）。"""
        out: List[Dict[str, Any]] = []
        for key in sorted(self._plugins):
            loaded = self._plugins[key]
            out.append(
                {
                    "key": key,
                    "name": loaded.manifest.name,
                    "version": loaded.manifest.version,
                    "kind": loaded.manifest.kind,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "error": loaded.error,
                    "tools": list(loaded.tools_registered),
                    "hooks": list(loaded.hooks_registered),
                    "middleware": list(loaded.middleware_registered),
                }
            )
        return out

    # ------------------------------------------------------------------
    # 卸载
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_plugin_key(
        plugin: Union[str, PluginManifest, LoadedPlugin],
    ) -> str:
        if isinstance(plugin, LoadedPlugin):
            return plugin.manifest.key or plugin.manifest.name
        if isinstance(plugin, PluginManifest):
            return plugin.key or plugin.name
        return str(plugin)

    def unload(
        self,
        plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
    ) -> bool:
        """卸载一个或全部插件（对应原版 plugins.py:3602 精简版）。

        plugin=None 为全量卸载（force 重扫的生命周期操作）：账本驱动反序
        释放全部注册，清空各表并复位 _discovered。返回是否卸载了任何内容。
        """
        with self._discovery_lock:
            return self._unload_scoped(plugin)

    def _unload_scoped(
        self,
        plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
    ) -> bool:
        """unload 的无锁实现（见 unload 文档）。"""
        unload_all = plugin is None
        if unload_all:
            target_keys = set(self._ownership_ledger) | set(self._plugins)
            registrations = list(self._registration_order)
        else:
            requested = self._resolve_plugin_key(plugin)
            target_keys = {
                key
                for key in self._plugins
                if key == requested
                or self._plugins[key].manifest.name == requested
            }
            target_keys.update(
                key for key in self._ownership_ledger if key == requested
            )
            registrations = [
                registration
                for registration in self._registration_order
                if registration.plugin_key in target_keys
            ]

        found = bool(target_keys or registrations)
        self._dispose_registrations(registrations)
        self._forget_registrations(registrations)

        if unload_all:
            # handle 是全局注册的权威来源；同时清空管理器本地表，抹掉账本
            # 出现前遗留的手动/旧状态。
            self._ownership_ledger.clear()
            self._plugins.clear()
            self._hooks.clear()
            self._middleware.clear()
            self._system_prompt_sections.clear()
            self._subscriptions.clear()
            self._plugin_config_overrides.clear()
            self._discovered = False
        else:
            for key in target_keys:
                self._plugins.pop(key, None)

        return found


def _detect_kind_from_source(source_text: str) -> Optional[str]:
    """从源码标记推断插件 kind（对应原版 plugins.py:922 精简版）。

    注册 memory provider（register_memory_provider / MemoryProvider）的模块
    属于 exclusive——走 plugins/memory 专属发现，通用 PluginManager 不导入它。
    """
    if "register_memory_provider" in source_text or "MemoryProvider" in source_text:
        return "exclusive"
    return None


# ══════════════════════════════════════════════════════════════════
# 模块级入口（对应原版 plugins.py:5390-5736 精简版）
# ══════════════════════════════════════════════════════════════════


def _plugin_home_key() -> Path:
    """返回插件状态用的 profile/home 键（对应原版 plugins.py:5390）。"""
    try:
        return get_hermes_home().expanduser().resolve()
    except Exception:
        return get_hermes_home().expanduser()


_plugin_manager: Optional[PluginManager] = None
_plugin_managers_by_home: Dict[str, PluginManager] = {}
_plugin_managers_lock = threading.RLock()


def _clear_plugin_submodules(manager: Optional[PluginManager]) -> None:
    """清掉目录插件在 sys.modules 里的模块及其子模块（对应原版 :5407 精简版）。

    插件 __init__.py 的相对导入（``from . import foo``）会缓存成
    ``hermes_plugins.<slug>.<submodule>``；丢弃 manager 前必须一并驱逐，
    否则同 slug 插件重载会静默复用上一份代码/状态。
    """
    if manager is None:
        return
    for loaded in getattr(manager, "_plugins", {}).values():
        module = getattr(loaded, "module", None)
        module_name = getattr(module, "__name__", None)
        if not module_name or not module_name.startswith(f"{_NS_PARENT}."):
            continue
        prefix = f"{module_name}."
        for name in [
            n for n in sys.modules if n == module_name or n.startswith(prefix)
        ]:
            del sys.modules[name]


def get_plugin_manager() -> PluginManager:
    """返回当前 profile/home 的插件管理器（对应原版 plugins.py:5440 精简版）。

    按解析后的 hermes home 缓存 manager；测试可通过 monkeypatch
    ``plugins_mod._plugin_manager`` 注入假 manager。
    """
    global _plugin_manager
    current_home = _plugin_home_key()

    with _plugin_managers_lock:
        if (
            _plugin_manager is not None
            and _plugin_manager not in _plugin_managers_by_home.values()
        ):
            _plugin_managers_by_home[current_home] = _plugin_manager
            return _plugin_manager

        manager = _plugin_managers_by_home.get(current_home)
        if manager is None:
            manager = PluginManager(scope_key=str(current_home))
            _plugin_managers_by_home[current_home] = manager

        _plugin_manager = manager
        return manager


def _reset_plugin_managers_for_tests() -> None:
    """测试专用：丢弃全部缓存的 manager 及其子模块（对应原版 :5474 精简版）。"""
    global _plugin_manager
    with _plugin_managers_lock:
        managers = list(dict.fromkeys(_plugin_managers_by_home.values()))
        if _plugin_manager is not None and _plugin_manager not in managers:
            managers.append(_plugin_manager)
        for manager in managers:
            _clear_plugin_submodules(manager)
            try:
                manager.unload()
            except Exception:
                logger.debug("test plugin-manager unload failed", exc_info=True)
        _plugin_managers_by_home.clear()
        _plugin_manager = None


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """返回全局 manager，并确保发现已执行（对应原版 plugins.py:6166）。"""
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def discover_plugins(force: bool = False) -> None:
    """显式触发插件发现（对应原版 plugins.py:5505 精简版）。"""
    _ensure_plugins_discovered(force=force)


def unload_plugins(
    plugin: Union[str, PluginManifest, LoadedPlugin, None] = None,
) -> bool:
    """从进程全局 manager 卸载一个或全部插件（对应原版 plugins.py:5648）。"""
    return get_plugin_manager().unload(plugin)


def _delivery_manager() -> PluginManager:
    """返回活跃 manager，若从未发现则懒触发（对应原版 plugins.py:5660）。

    钩子/中间件派发不依赖调用方是否显式 discover——首次 invoke_hook 自动
    完成发现，保证任何入口（CLI / 测试 / 库调用）都能触发用户插件回调。
    """
    manager = get_plugin_manager()
    if not getattr(manager, "_discovered", True):
        manager.discover_and_load()
    return manager


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """在已加载插件上调用生命周期钩子（对应原版 plugins.py:5680）。

    首次调用触发懒发现。返回插件回调的非 None 返回值列表。
    """
    return _delivery_manager().invoke_hook(hook_name, **kwargs)


def render_system_prompt_sections(
    session_info: Mapping[str, Any],
) -> List[RenderedPluginSystemPromptSection]:
    """渲染插件系统提示词段（对应原版 plugins.py:5693）。"""
    return _ensure_plugins_discovered().render_system_prompt_sections(session_info)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """调用已注册中间件回调（对应原版 plugins.py:5700）。"""
    return _delivery_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """*kind* 是否有中间件回调（对应原版 plugins.py:5711）。"""
    return _delivery_manager().has_middleware(kind)


def has_hook(hook_name: str) -> bool:
    """*hook_name* 是否有已加载插件处理（对应原版 plugins.py:5727）。"""
    return _delivery_manager().has_hook(hook_name)


def iter_hook_callbacks(hook_name: str) -> tuple:
    """返回某钩子回调的稳定快照（对应原版 plugins.py:5736）。"""
    return get_plugin_manager().iter_hook_callbacks(hook_name)
