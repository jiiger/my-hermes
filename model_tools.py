"""工具装配 / 派发层（对齐原版 toolsets 语义的移植版）。

对应原版 hermes-agent 的 model_tools.py（1569 行）。my-hermes 是 flat
layout，顶层模块直接 import。本版按「方案 B」对齐原版：
- 模块导入时即 discover_builtin_tools()（AST 扫描 + 磁盘缓存 + 自注册）；
- get_tool_definitions 走原版 _compute_tool_definitions 语义：
  enabled_toolsets 用 toolsets.resolve_toolset 收集 / disabled_toolsets
  做严格减法 / 默认 get_all_toolsets 全收集（含 MCP 的 mcp-* toolset）；
- quiet_mode 缓存（_tool_defs_cache + LRU），缓存键含 registry._generation
  （MCP 注册/注销自动失效）；
- 输出顺序为 registry.get_definitions 的字母序（对齐原版）。

与原版裁剪：kanban/delegated/profile-scope 缓存键项、execute_code/discord
动态 schema 重建、tool_search 装配（skip_tool_search_assembly 保留签名
但忽略）、check_toolset_requirements/check_tool_availability（无消费方）。
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error, discover_builtin_tools
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# ── 模块导入时即发现内置工具（对齐原版 model_tools.py:214）──────────────
# AST 扫描 tools/*.py 中顶层 registry.register(...) 的模块并 import 触发
# 自注册；判定结果按 (mtime, size) 存磁盘缓存，热路径近零成本。
discover_builtin_tools()


# ── 兼容旧工具集名（原版 :254 精简：只保留 my-hermes 实际有工具的 legacy 名）
_LEGACY_TOOLSET_MAP = {
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "terminal_tools": ["terminal"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
}


# ── get_tool_definitions 缓存（对齐原版 :247-297）───────────────────────
# quiet_mode=True 时按缓存键记忆结果；LRU 淘汰防止长驻进程无限累积。
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_TOOL_DEFS_CACHE_MAX = 8


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """返回 OpenAI 格式的工具 schema 列表（原版 :305 签名 + 缓存语义）。

    enabled_toolsets 为 None = 全开（静态 TOOLSETS + registry 派生工具集，
    含 MCP 的 ``mcp-{server}``）；否则只包含指定工具集解析出的工具。
    disabled_toolsets 最后做严格减法。quiet_mode=True 时结果被缓存，
    缓存键含 registry._generation——MCP 工具注册/注销后自动失效。
    """
    # 快速路径：quiet 调用无需 stdout 打印时用记忆化结果。
    # 缓存键捕获入参 + registry 代次（MCP 刷新/插件加载）+ config mtime
    # （动态 schema 依赖 config 的场景）。
    cache_key = None
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path

            cfg_stat = get_config_path().stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
        )
        cached = _tool_defs_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            # 浅拷贝：下游（run_agent 追加工具 schema）不会污染缓存。
            return list(cached)

    result = _compute_tool_definitions(
        enabled_toolsets,
        disabled_toolsets,
        quiet_mode,
        skip_tool_search_assembly=skip_tool_search_assembly,
    )
    if quiet_mode and cache_key is not None:
        # LRU 淘汰最旧条目，防长驻进程无限累积。
        if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
            _tool_defs_cache.pop(next(iter(_tool_defs_cache)))
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
    skip_tool_search_assembly: bool = False,
) -> List[Dict[str, Any]]:
    """get_tool_definitions 的无缓存实现（对齐原版 :399 结构）。

    流程：enabled/默认收集 → disabled 减法 → registry.get_definitions
    取 schema（只返回 check_fn 通过且已注册的工具；TOOLSETS 里声明的
    未移植工具名自动被忽略）。
    """
    del skip_tool_search_assembly  # my-hermes 无 tool_search，保留签名忽略

    tools_to_include: set = set()

    if enabled_toolsets is not None:
        for ts_name in enabled_toolsets:
            if validate_toolset(ts_name):
                tools_to_include.update(resolve_toolset(ts_name))
            elif ts_name in _LEGACY_TOOLSET_MAP:
                tools_to_include.update(_LEGACY_TOOLSET_MAP[ts_name])
            elif not quiet_mode:
                logger.warning("Unknown toolset: %s", ts_name)
    else:
        # 默认全开：静态 TOOLSETS + registry 派生工具集（含 MCP mcp-*）。
        from toolsets import get_all_toolsets

        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # disabled 最后做严格减法：即使复合集启用，禁用集的工具也被剥除。
    if disabled_toolsets:
        for ts_name in disabled_toolsets:
            if validate_toolset(ts_name):
                tools_to_include.difference_update(resolve_toolset(ts_name))
            elif ts_name in _LEGACY_TOOLSET_MAP:
                tools_to_include.difference_update(_LEGACY_TOOLSET_MAP[ts_name])
            elif not quiet_mode:
                logger.warning("Unknown toolset: %s", ts_name)

    # registry.get_definitions 按名字母序返回（对齐原版）。
    return registry.get_definitions(tools_to_include, quiet=quiet_mode)


def handle_function_call(
    function_name: str,
    function_args: Optional[Dict[str, Any]],
    task_id: Optional[str] = None,
    user_task: Optional[str] = None,
) -> str:
    """派发工具调用，返回 handler 的 str() 结果。

    对应原版 model_tools.py:1123 handle_function_call()；精简版砍掉了
    middleware / hooks / 类型强转 / tool_search 桥接，只做注册表派发。
    未注册的工具返回错误字符串（fail-open，对话循环可继续）。
    """
    del task_id, user_task
    entry = registry.get_entry(function_name)
    if not entry:
        return tool_error(f"Unknown tool: {function_name}")
    try:
        result = entry.handler(**(function_args or {}))
        return str(result)
    except Exception as exc:
        logger.exception("Tool %s dispatch error: %s", function_name, exc)
        return tool_error(f"Tool execution failed: {type(exc).__name__}: {exc}")


def get_all_tool_names() -> List[str]:
    """返回全部已注册工具名（对应原版 model_tools.py:1547）。"""
    return registry.get_all_tool_names()


def build_tool_impls_map() -> Dict[str, Callable]:
    """返回 {工具名: handler} 映射，供 agent._tool_impls 使用。

    run_agent._execute_tool_calls 按此契约调用 impl(**args)；
    handler 的签名参数名与 schema properties 一致。
    """
    return {entry.name: entry.handler for entry in registry.iter_entries()}


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """返回工具所属 toolset（对应原版 model_tools.py:1552）。"""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """返回 toolset 可用性信息（UI 显示用；保持 my-hermes 既有实现）。

    遍历 registry 返回 {toolset: {available, tools}}（叶子 + mcp-*）。
    原版 registry.get_available_toolsets 依赖 _snapshot_state 等较重内部
    结构，my-hermes 精简版未移植——功能等价，仅缺 description/requirements。
    """
    toolsets: Dict[str, dict] = {}
    for entry in registry.iter_entries():
        ts = entry.toolset
        if ts not in toolsets:
            toolsets[ts] = {"available": True, "tools": []}
        toolsets[ts]["tools"].append(entry.name)
    return toolsets
