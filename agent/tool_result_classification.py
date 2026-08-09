"""工具结果负载分类共享助手（精简移植版）。

对应原版 hermes-agent 的 agent/tool_result_classification.py（40 行）。
纯独立模块，照抄原版：NO_EFFECT_TOOL_NAMES / FILE_MUTATING_TOOL_NAMES
常量保留原版全集。
"""

from __future__ import annotations

import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


# 中断/悬挂执行可安全丢弃的工具：它们既不能改外部状态也不能改 Hermes
# 会话状态。未知/插件/MCP 工具默认视为有副作用。
NO_EFFECT_TOOL_NAMES = frozenset({
    "read_file", "search_files", "session_search", "skill_view", "skills_list",
    "web_extract", "web_search", "vision_analyze", "browser_snapshot",
    "browser_get_images", "browser_console", "read_terminal",
})


def tool_may_have_side_effect(tool_name: str) -> bool:
    return tool_name not in NO_EFFECT_TOOL_NAMES


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """当文件变更结果证明写入已落盘时返回 True。"""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception:
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
