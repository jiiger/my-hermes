"""Todo 工具（精简移植版）。

对应原版 hermes-agent 的 tools/todo_tool.py（335 行）。提供内存任务
列表，供 agent 拆解复杂任务、跟踪进度、保持多轮对话聚焦。

精简版砍掉：
- 上下文压缩注入相关：TodoStore.format_for_injection /
  TODO_INJECTION_HEADER / MAX_TODO_RESULT_CHARS
  （my-hermes 没有 ContextCompressor，无需在压缩后重新注入）；
- handler 不再依赖 AIAgent 传入 store，改用模块级 TodoStore 单例
  （与 my-hermes _execute_tool_calls 的 impl(**args) 契约一致）。
"""

import json
from typing import Any, Dict, List, Optional


# 合法的 todo 状态值（原版 todo_tool.py:21）
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# 单条内容上限：todo 列表是模型的规划辅助，单条过大只会浪费上下文
# （原版 todo_tool.py:32）。
MAX_TODO_CONTENT_CHARS = 4000
# 列表总条数上限：活动列表通常是几条到几十条，而非上百条（原版 :37）。
MAX_TODO_ITEMS = 256
_TRUNCATION_MARKER = "… [truncated]"


class TodoStore:
    """内存 todo 列表（精简版为模块级单例，原版挂在 AIAgent 上）。

    条目按列表位置表达优先级。每个条目：
      - id: 唯一字符串标识（agent 自选）
      - content: 任务描述
      - status: pending | in_progress | completed | cancelled
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []

    def write(
        self,
        todos: List[Dict[str, Any]],
        merge: bool = False,
    ) -> List[Dict[str, str]]:
        """写入 todos，返回写入后的完整列表（对应原版 :78）。

        merge=False：整体替换；merge=True：按 id 更新已有项、追加新项。
        """
        if not merge:
            # 替换模式：全新列表
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
        else:
            # 合并模式：按 id 更新已有项，追加新项
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip() if isinstance(t, dict) else ""
                if not item_id:
                    continue  # 无 id 无法合并
                if item_id in existing:
                    # 只更新模型实际提供的字段
                    if isinstance(t, dict):
                        if t.get("content"):
                            existing[item_id]["content"] = self._cap_content(
                                str(t["content"]).strip()
                            )
                        if t.get("status"):
                            status = str(t["status"]).strip().lower()
                            if status in VALID_STATUSES:
                                existing[item_id]["status"] = status
                else:
                    # 新条目：完整校验后追加到末尾
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # 按原顺序重建 _items（保持已有项的相对顺序）
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        # 条数上限：保留优先级最高的头部（列表顺序即优先级）
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """返回当前列表的副本（对应原版 :106）。"""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """列表是否非空（对应原版 :112）。"""
        return bool(self._items)

    @staticmethod
    def _cap_content(content: str) -> str:
        """超长内容截断到 MAX_TODO_CONTENT_CHARS（对应原版 :150）。"""
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """校验并规范化一个条目（对应原版 :169）。

        确保必填字段存在、状态合法，返回干净的 {id, content, status}。
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"
        else:
            content = TodoStore._cap_content(content)

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """折叠重复 id，保留每个 id 最后一次出现的位置（对应原版 :196）。"""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # 非 dict 条目用合成 key，交给 _validate 处理
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


# 模块级单例 store：my-hermes 执行侧只调 impl(**args)，不传 store，
# 因此 todo 状态挂在模块级（跨会话共享；单会话场景等价于挂在 agent 上）。
_TODO_STORE = TodoStore()


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    **kwargs,
) -> str:
    """todo 工具唯一入口：给参即写、省略即读。

    对应原版 todo_tool.py:223 todo_tool()；砍掉了 store 参数（改用
    模块级 _TODO_STORE），**kwargs 吸收模型可能多传的参数。

    返回 JSON 字符串：完整列表 + 按状态统计的 summary。
    """
    del kwargs  # 吸收多余参数；精简版无其他参数

    if todos is not None:
        # 兜底：模型有时把 todos 传成 JSON 字符串而不是数组
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error(
                    "todos must be a list of objects, got unparseable string"
                )
        if not isinstance(todos, list):
            return tool_error(f"todos must be a list, got {type(todos).__name__}")
        items = _TODO_STORE.write(todos, merge)
    else:
        items = _TODO_STORE.read()

    # 按状态统计
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """todo 工具无外部依赖，恒可用（对应原版 :313）。"""
    return True


# =============================================================================
# OpenAI Function-Calling Schema（对应原版 todo_tool.py:322 TODO_SCHEMA）
# 行为指引写在 description 里，属于静态 schema，随注册缓存。
# =============================================================================

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status"
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False
            }
        },
        "required": []
    }
}


# --- 模块级自注册（对应原版 todo_tool.py:330 registry.register）---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=todo_tool,
    check_fn=check_todo_requirements,
    emoji="📋",
)
