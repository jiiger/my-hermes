"""工具注册中心（精简移植版）。

对应原版 hermes-agent 的 tools/registry.py（956 行）。本文件只保留
Phase 1 需要的核心：
- ToolEntry / ToolRegistry：register、get_definitions、dispatch、查询接口；
- registry 单例 + tool_error / tool_result 序列化助手。

精简版砍掉（见各函数注释）：
- discover_builtin_tools 的 AST 扫描 + mtime/size 磁盘缓存已移植（见
  文件末尾），由 model_tools 模块导入时调用触发工具自注册；
- 插件授权 / override / deregister 门禁（hermes_plugins 相关）；
- check_fn 的 TTL 缓存与 last-good 抖动抑制（Phase 1 无 gated 工具，
  保留 check_fn 参数、在 get_definitions 里直接调用）。

handler 契约：与 my-hermes run_agent._execute_tool_calls 一致，
handler 是「可被 **args 直接调用」的普通函数，返回 str 或可 str() 的对象。
"""

import ast
import importlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ToolEntry:
    """单个已注册工具的元数据。

    对应原版 registry.py:184 ToolEntry；精简版砍掉了
    dynamic_schema_overrides（运行时动态 schema 覆盖）。
    """

    __slots__ = (
        "name", "toolset", "schema", "handler", "check_fn",
        "requires_env", "is_async", "description", "emoji",
        "max_result_size_chars",
    )

    def __init__(
        self,
        name,
        toolset,
        schema,
        handler,
        check_fn,
        requires_env,
        is_async,
        description,
        emoji,
        max_result_size_chars=None,
    ):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.requires_env = requires_env
        self.is_async = is_async
        self.description = description
        self.emoji = emoji
        self.max_result_size_chars = max_result_size_chars


class ToolRegistry:
    """收集工具 schema + handler 的单例注册表。

    对应原版 registry.py:373 ToolRegistry；精简版砍掉了插件 override
    策略、toolset 别名、MCP 动态刷新等 Phase 1 用不到的成员。
    """

    def __init__(self):
        # 注册表：{工具名: ToolEntry}。Python dict 保持插入顺序，
        # get_definitions 按注册顺序返回（冒烟测试期望的顺序）。
        self._tools: Dict[str, ToolEntry] = {}
        # toolset 别名映射：{别名: 真实 toolset}（对应原版 registry.py:487
        # register_toolset_alias；MCP 用它把 server 名注册为 mcp-{server}
        # toolset 的别名，供按 server 名查询工具）。
        self._toolset_aliases: Dict[str, str] = {}
        # 读写锁：工具模块在 import 时注册、运行期只读查询，
        # 用 RLock 让并发读取拿到稳定快照。
        self._lock = threading.RLock()
        # 注册表代次（对应原版 registry.py:436）：register / deregister /
        # register_toolset_alias 都会 +1。model_tools 的 get_tool_definitions
        # 缓存键用它做失效触发器——MCP 工具后注册时缓存自动作废。
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable = None,
        requires_env: list = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
        max_result_size_chars: int | float | None = None,
    ):
        """注册一个工具（工具文件在模块级调用）。

        对应原版 registry.py:521 register()；精简版去掉跨 toolset
        shadow 拒绝与插件 override 授权，同名直接覆盖。
        """
        with self._lock:
            self._tools[name] = ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                handler=handler,
                check_fn=check_fn,
                requires_env=requires_env or [],
                is_async=is_async,
                description=description or schema.get("description", ""),
                emoji=emoji,
                max_result_size_chars=max_result_size_chars,
            )
            self._generation += 1

    def deregister(self, name: str) -> None:
        """移除一个工具（按名删除；不存在则静默忽略）。

        对应原版 registry.py:646 deregister()；精简版没有 MCP 动态
        发现，此方法仅作预留接口。
        """
        with self._lock:
            self._tools.pop(name, None)
            self._generation += 1

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_entry(self, name: str) -> Optional[ToolEntry]:
        """返回已注册工具条目，或 None（对应原版 registry.py:461）。"""
        with self._lock:
            return self._tools.get(name)

    def get_all_tool_names(self) -> List[str]:
        """返回全部已注册工具名（排序）（对应原版 registry.py:824）。"""
        with self._lock:
            return sorted(entry.name for entry in self._tools.values())

    def get_schema(self, name: str) -> Optional[dict]:
        """返回工具的原始 schema（不做 check_fn 过滤）（原版 registry.py:831）。"""
        entry = self.get_entry(name)
        return entry.schema if entry else None

    def get_toolset_for_tool(self, name: str) -> Optional[str]:
        """返回工具所属 toolset，或 None（对应原版 registry.py:839）。"""
        entry = self.get_entry(name)
        return entry.toolset if entry else None

    def get_emoji(self, name: str, default: str = "⚡") -> str:
        """返回工具 emoji，未设置则用 *default*（对应原版 registry.py:845）。"""
        entry = self.get_entry(name)
        return (entry.emoji if entry and entry.emoji else default)

    def get_max_result_size(self, name: str, default: int | float | None = None) -> int | float:
        """返回工具的 max_result_size（用于结果持久化阈值）。

        对应原版 registry.py:796 get_max_result_size()；供
        tools/budget_config.py 的 resolve_threshold 使用（第 2 批移植
        预算链时补齐的方法）。优先级：工具注册的 max_result_size_chars
        → 调用方给的 default → budget_config 的全局默认。
        """
        entry = self.get_entry(name)
        if entry and entry.max_result_size_chars is not None:
            return entry.max_result_size_chars
        if default is not None:
            return default
        # 函数内 import 避免与 tools.budget_config 的循环导入
        # （budget_config.resolve_threshold 也是函数内 import registry）。
        from tools.budget_config import DEFAULT_RESULT_SIZE_CHARS
        return DEFAULT_RESULT_SIZE_CHARS

    def get_tool_to_toolset_map(self) -> Dict[str, str]:
        """返回 {工具名: toolset} 映射（对应原版 registry.py:851）。"""
        with self._lock:
            return {entry.name: entry.toolset for entry in self._tools.values()}

    def get_registered_toolset_names(self) -> List[str]:
        """返回注册表中出现过的 toolset 名（排序去重）（原版 registry.py:467）。"""
        with self._lock:
            return sorted({entry.toolset for entry in self._tools.values()})

    def iter_entries(self) -> List[ToolEntry]:
        """返回全部条目快照（按注册顺序，对应原版 _snapshot_entries 的简化）。"""
        with self._lock:
            return list(self._tools.values())

    def get_tool_names_for_toolset(self, toolset: str) -> List[str]:
        """返回某个 toolset 下的工具名（排序）（对应原版 registry.py:473）。"""
        with self._lock:
            return sorted(
                entry.name for entry in self._tools.values()
                if entry.toolset == toolset
            )

    # ------------------------------------------------------------------
    # Toolset aliases
    # ------------------------------------------------------------------

    def register_toolset_alias(self, alias: str, toolset: str) -> None:
        """注册 toolset 别名（对应原版 registry.py:487）。

        别名指向真实 toolset，供按别名查询工具（如 MCP server 名 → 其
        ``mcp-{server}`` toolset）。幂等：重复注册直接覆盖（覆盖时记
        warning，对齐原版注册纪律）。
        """
        with self._lock:
            existing = self._toolset_aliases.get(alias)
            if existing and existing != toolset:
                logger.warning(
                    "Toolset alias collision: '%s' (%s) overwritten by %s",
                    alias, existing, toolset,
                )
            self._toolset_aliases[alias] = toolset
            self._generation += 1

    def get_toolset_for_alias(self, alias: str) -> Optional[str]:
        """返回别名指向的 toolset（未注册返回 None）。"""
        with self._lock:
            return self._toolset_aliases.get(alias)

    def get_registered_toolset_aliases(self) -> Dict[str, str]:
        """返回 ``{alias: canonical_toolset}`` 映射快照（对应原版 :499）。"""
        with self._lock:
            return dict(self._toolset_aliases)

    def get_toolset_alias_target(self, alias: str) -> Optional[str]:
        """返回别名指向的 canonical toolset 名（对应原版 :510）。"""
        with self._lock:
            return self._toolset_aliases.get(alias)

    def get_tool_names_for_alias(self, alias: str) -> List[str]:
        """返回别名下所有工具名（按 toolset 查询，未注册返回空列表）。"""
        toolset = self.get_toolset_for_alias(alias)
        if not toolset:
            return []
        return self.get_tool_names_for_toolset(toolset)

    # ------------------------------------------------------------------
    # Schema retrieval
    # ------------------------------------------------------------------

    def get_definitions(self, tool_names: Set[str], quiet: bool = False) -> List[dict]:
        """按工具名集合返回 OpenAI 格式工具 schema 列表。

        对应原版 registry.py:606 get_definitions()；精简版对 check_fn
        直接调用（无 TTL 缓存、无 last-good 抖动抑制），返回 False 即
        过滤掉该工具。
        """
        result = []
        check_results: Dict[Callable, bool] = {}
        with self._lock:
            entries_by_name = {entry.name: entry for entry in self._tools.values()}
        for name in sorted(tool_names):
            entry = entries_by_name.get(name)
            if not entry:
                continue
            if entry.check_fn:
                if entry.check_fn not in check_results:
                    check_results[entry.check_fn] = bool(entry.check_fn())
                if not check_results[entry.check_fn]:
                    if not quiet:
                        logger.debug("Tool %s unavailable (check failed)", name)
                    continue
            # 确保 schema 始终有 "name" 字段（用注册名兜底）
            schema_with_name = {**entry.schema, "name": entry.name}
            result.append({"type": "function", "function": schema_with_name})
        return result

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, name: str, args: dict, **kwargs) -> str:
        """按名执行工具 handler。

        对应原版 registry.py:775 dispatch()；精简版 handler 直接以
        **args 调用（与 run_agent._execute_tool_calls 契约一致），
        is_async 暂未使用（Phase 1 无异步工具），异常统一转
        tool_error（fail-open，对话循环可继续）。
        """
        entry = self.get_entry(name)
        if not entry:
            return tool_error(f"Unknown tool: {name}")
        try:
            result = entry.handler(**args)
            return str(result)
        except Exception as exc:
            logger.exception("Tool %s dispatch error: %s", name, exc)
            return tool_error(f"Tool execution failed: {type(exc).__name__}: {exc}")


# 模块级单例（对应原版 registry.py:905）
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# 工具响应序列化助手（对应原版 registry.py:913-956）
# 每个工具 handler 返回 JSON 字符串；这两个助手消除反复 json.dumps 的样板。
# ---------------------------------------------------------------------------


def tool_error(message, **extra) -> str:
    """返回 JSON 错误字符串（对应原版 registry.py:931）。"""
    result = {"error": str(message)}
    if extra:
        result.update(extra)
    return json.dumps(result, ensure_ascii=False)


def tool_result(data=None, **kwargs) -> str:
    """返回 JSON 结果字符串（对应原版 registry.py:948）。

    接受 dict 位置参数或关键字参数（二选一）。
    """
    if data is not None:
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(kwargs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 内置工具发现（对应原版 registry.py:71-200 discover_builtin_tools）
# ---------------------------------------------------------------------------


def _is_registry_register_call(node: ast.AST) -> bool:
    """*node* 是否为 ``registry.register(...)`` 顶层调用表达式（原版 :71）。"""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "register"
        and isinstance(func.value, ast.Name)
        and func.value.id == "registry"
    )


def _module_registers_tools(module_path: Path) -> bool:
    """模块体顶层是否含 ``registry.register(...)`` 调用（原版 :90）。

    只检查模块体语句，避免把函数体内调用 registry.register 的辅助模块
    误判。先做廉价文本预过滤（必须同时出现 registry 与 register），
    减少 ast.parse 开销。
    """
    try:
        source = module_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if "registry" not in source or "register" not in source:
        return False
    try:
        tree = ast.parse(source, filename=str(module_path))
    except SyntaxError:
        return False
    return any(_is_registry_register_call(stmt) for stmt in tree.body)


def discover_builtin_tools(tools_dir: Optional[Path] = None) -> List[str]:
    """导入所有自带的自注册工具模块并返回模块名列表（原版 :108）。

    扫描 ``tools/*.py``，对含顶层 ``registry.register(...)`` 的模块执行
    importlib 触发自注册。逐文件 AST 扫描有成本，判定结果按
    ``(mtime_ns, size)`` 键控存磁盘缓存（``~/.hermes/cache/
    tool_discovery_cache.json``）；命中缓存的文件直接信任不重扫，缓存
    写入 best-effort 原子写，多进程可无害竞争。
    """
    tools_path = Path(tools_dir) if tools_dir is not None else Path(__file__).resolve().parent

    cache = _load_discovery_cache()
    fresh_cache: Dict[str, list] = {}
    cache_dirty = False

    module_names: List[str] = []
    for path in sorted(tools_path.glob("*.py")):
        if path.name in {"__init__.py", "registry.py", "mcp_tool.py"}:
            continue
        abs_path = str(path.resolve())
        try:
            st = path.stat()
            stat_key = (st.st_mtime_ns, st.st_size)
        except OSError:
            continue
        cached = cache.get(abs_path)
        if (
            isinstance(cached, (list, tuple))
            and len(cached) == 3
            and (cached[0], cached[1]) == stat_key
        ):
            registers = bool(cached[2])
        else:
            registers = _module_registers_tools(path)
            cache_dirty = True
        fresh_cache[abs_path] = [stat_key[0], stat_key[1], registers]
        if registers:
            module_names.append(f"tools.{path.stem}")

    # 清理已删除文件的缓存条目；仅在变化时重写。
    if cache_dirty or set(fresh_cache) != set(cache):
        _save_discovery_cache(fresh_cache)

    imported: List[str] = []
    for mod_name in module_names:
        try:
            importlib.import_module(mod_name)
            imported.append(mod_name)
        except Exception as e:
            logger.warning("Could not import tool module %s: %s", mod_name, e)
    return imported


def _discovery_cache_path() -> Optional[Path]:
    """工具发现判定缓存路径；解析失败返回 None（原版 :162）。"""
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()) / "cache" / "tool_discovery_cache.json"
    except Exception:
        return None


def _load_discovery_cache() -> Dict[str, list]:
    """读发现缓存；任何错误 → 空 dict（全量重扫）（原版 :179）。"""
    path = _discovery_cache_path()
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_discovery_cache(cache: Dict[str, list]) -> None:
    """best-effort 原子写发现缓存，绝不抛异常（原版 :190）。

    复用 utils.atomic_write_text（临时文件 + fsync + 原子重命名）。
    """
    path = _discovery_cache_path()
    if path is None:
        return
    try:
        from utils import atomic_write_text

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(cache, indent=0, ensure_ascii=False))
    except Exception as e:
        logger.debug("Could not write tool discovery cache %s: %s", path, e)
