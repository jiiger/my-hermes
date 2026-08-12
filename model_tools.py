"""工具装配 / 派发层（精简移植版）。

对应原版 hermes-agent 的 model_tools.py（1569 行）。my-hermes 是 flat
layout，顶层模块直接 import。精简版：
- 用 _TOOL_MODULES 显式导入工具模块触发自注册
  （替代原版 discover_builtin_tools 的 AST 扫描 + 磁盘缓存）；
- 工具集过滤内联在 get_tool_definitions（选更简单的方案 a，
  不建 toolsets.py；后续要加工具集只需扩展过滤逻辑）；
- handle_function_call 查 registry 派发，未注册返回错误字符串（fail-open）；
- build_tool_impls_map 生成 {工具名: handler} 映射，
  喂给 my-hermes run_agent.AIAgent._tool_impls 供 _execute_tool_calls 使用。
"""

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具模块清单：显式 import 触发模块级自注册。以后加工具只改这一个列表。
# 顺序即 get_tool_definitions 的返回顺序（todo → file → terminal）。
# ---------------------------------------------------------------------------
_TOOL_MODULES = [
    "tools.todo_tool",
    "tools.file_tools",
    "tools.terminal_tool",
    "tools.memory_tool",
]

# Phase 1 工具名的展示顺序（与 _TOOL_MODULES 一一对应）。
# get_tool_definitions 按此顺序输出：即使其他代码提前单独导入某个工具
# 模块（注册顺序被扰动），返回给 API 的工具列表顺序依然稳定。
_TOOL_ORDER = [
    "todo",
    "read_file",
    "write_file",
    "patch",
    "search_files",
    "terminal",
    "memory",
]

_tools_loaded = False


def _ensure_tools_loaded() -> None:
    """按 _TOOL_MODULES 导入工具模块（幂等）。

    单个模块导入失败只告警不中断：其余工具仍可用。
    """
    global _tools_loaded
    if _tools_loaded:
        return
    for mod_name in _TOOL_MODULES:
        try:
            importlib.import_module(mod_name)
        except Exception as exc:
            logger.warning("Could not import tool module %s: %s", mod_name, exc)
    _tools_loaded = True


# ---------------------------------------------------------------------------
# 工具集过滤：Phase 1 三个工具集 todo / file / terminal。
# 与注册时 entry.toolset 字符串一致；"all"/"*" 表示全开。
# ---------------------------------------------------------------------------


def _toolset_enabled(
    toolset: str,
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
) -> bool:
    """判断某个 toolset 是否应暴露（内联过滤，对应原版 toolsets.py 语义）。

    enabled_toolsets 为 None = 不限制；否则只放行列表内的 toolset。
    disabled_toolsets 为 None = 不排除；否则排除列表内的 toolset。
    """
    if enabled_toolsets is not None:
        if "all" not in enabled_toolsets and toolset not in enabled_toolsets:
            return False
    if disabled_toolsets and toolset in disabled_toolsets:
        return False
    return True


def get_tool_definitions(
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    quiet_mode: bool = False,
) -> List[Dict[str, Any]]:
    """返回 OpenAI 格式的工具 schema 列表（按注册顺序）。

    对应原版 model_tools.py:305 get_tool_definitions()；砍掉了
    记忆化缓存、delegated-child / kanban 特判、平台工具集等逻辑。

    Args:
        enabled_toolsets: 只包含这些 toolset 的工具（None = 全开）。
        disabled_toolsets: 排除这些 toolset 的工具（None = 不排除）。
        quiet_mode: 静默模式（精简版仅影响 check_fn 失败时的日志级别）。

    Returns:
        [{"type": "function", "function": {name, description, parameters}}]
    """
    _ensure_tools_loaded()
    order = {name: idx for idx, name in enumerate(_TOOL_ORDER)}
    entries = [
        entry
        for entry in registry.iter_entries()
        if _toolset_enabled(entry.toolset, enabled_toolsets, disabled_toolsets)
    ]
    # 稳定排序：_TOOL_ORDER 内的按声明顺序，其余（未来新增）排最后
    entries.sort(key=lambda e: order.get(e.name, len(order)))
    result = []
    for entry in entries:
        if entry.check_fn and not entry.check_fn():
            if not quiet_mode:
                logger.debug("Tool %s unavailable (check failed)", entry.name)
            continue
        # 确保 schema 始终有 "name" 字段
        schema_with_name = {**entry.schema, "name": entry.name}
        result.append({"type": "function", "function": schema_with_name})
    return result


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

    Args:
        function_name: 工具名。
        function_args: 参数字典（handler 以 **args 调用）。
        task_id / user_task: 原版签名保留参数，精简版未使用。
    """
    del task_id, user_task
    _ensure_tools_loaded()
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
    _ensure_tools_loaded()
    return registry.get_all_tool_names()


def build_tool_impls_map() -> Dict[str, Callable]:
    """返回 {工具名: handler} 映射，供 agent._tool_impls 使用。

    run_agent._execute_tool_calls 按此契约调用 impl(**args)；
    handler 的签名参数名与 schema properties 一致。
    """
    _ensure_tools_loaded()
    return {entry.name: entry.handler for entry in registry.iter_entries()}


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """返回工具所属 toolset（对应原版 model_tools.py:1552）。"""
    _ensure_tools_loaded()
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """返回 toolset 可用性信息（对应原版 model_tools.py:1556 的简化）。"""
    _ensure_tools_loaded()
    toolsets: Dict[str, dict] = {}
    for entry in registry.iter_entries():
        ts = entry.toolset
        if ts not in toolsets:
            toolsets[ts] = {"available": True, "tools": []}
        toolsets[ts]["tools"].append(entry.name)
    return toolsets
