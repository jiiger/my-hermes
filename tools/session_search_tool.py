"""session_search 工具（精简移植版）。

对应原版 hermes-agent 的 tools/session_search_tool.py。单形状工具，按传入
参数推断四种调用模式，全部基于 SQLite 会话库返回真实消息，零 LLM 调用：

  1. DISCOVERY —— 传 ``query``：跨会话全文检索（FTS5 / CJK 时 LIKE 回退），
     按会话去重返回 top N，每个结果带命中锚点 ±5 消息窗口。
  2. SCROLL    —— 传 ``session_id`` + ``around_message_id``：以锚点为中心
     取 ±``window`` 消息窗口，无检索。向前/向后滚动时把窗口首/尾消息 id
     再传回作为锚点即可。
  3. READ      —— 只传 ``session_id``：导出整个会话（大会话截首 20 尾 10）。
  4. BROWSE    —— 无参：按最近活动列出会话。

与 my-hermes SessionDB 能力对齐，砍掉原版的 profile / lineage 去重 /
title / bookend / @session 链接等依赖原版 schema 的功能。
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

# 单条消息正文展示上限（防止超大 tool result 撑爆返回；检索窗口看大意
# 足够，细节用 SCROLL 再取）。
_MSG_CONTENT_MAX = 800
# READ 模式大会话的截首/截尾条数。
_READ_HEAD = 20
_READ_TAIL = 10


def _open_db():
    """打开默认会话库；失败返回 None（调用方转成工具错误）。"""
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception as exc:
        logger.warning("SessionDB unavailable for session_search: %s", exc)
        return None


def _fmt_ts(ts) -> str:
    """Unix 时间戳 → 人类可读时间（None/异常返回 unknown）。"""
    if not ts:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError, OverflowError):
        return "unknown"


def _msg_plain(msg: Dict[str, Any]) -> Dict[str, Any]:
    """把 DB 消息行精简成模型可读形状：文本化 content 并截断。"""
    content = msg.get("content")
    if isinstance(content, list):
        # 多模态结构化 content：只保留文本部分，图片转占位符
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif part.get("type") in {"image", "image_url", "input_image"}:
                text_parts.append("[screenshot]")
        content = "\n".join(text_parts) if text_parts else ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False) if content is not None else ""
    if len(content) > _MSG_CONTENT_MAX:
        content = content[:_MSG_CONTENT_MAX] + f"...[truncated, {len(content):,} chars total]"
    out: Dict[str, Any] = {
        "id": msg.get("id"),
        "role": msg.get("role"),
        "content": content,
    }
    if msg.get("tool_name"):
        out["tool_name"] = msg["tool_name"]
    return out


def _discover(db, query: str, limit: int) -> str:
    """DISCOVERY：跨会话检索，按会话去重，每个结果锚定首个命中。"""
    try:
        # 放宽取数（limit×8，上限 200）便于按会话去重后仍凑够 top N
        hits = db.search_messages(query, limit=min(limit * 8, 200))
    except Exception as exc:
        logger.error("session_search FTS5 search failed: %s", exc, exc_info=True)
        return tool_error(f"Search failed: {exc}")

    seen: Dict[str, Dict[str, Any]] = {}
    for hit in hits:
        sid = hit["session_id"]
        if sid not in seen:
            seen[sid] = hit
        if len(seen) >= limit:
            break

    results = []
    for sid, anchor in seen.items():
        meta = db.get_session(sid) or {}
        view = db.get_messages_around(sid, anchor["id"], window=3)
        # bookend：会话开头/结尾各 3 条 user+assistant，让模型快速判断
        # "这个会话是干嘛的 / 结论是什么"（对齐原版 discovery bookends）
        _ua = [
            m for m in db.get_messages(sid, include_inactive=True)
            if m.get("role") in ("user", "assistant")
        ]
        results.append({
            "session_id": sid,
            "source": meta.get("source"),
            "model": meta.get("model"),
            "when": _fmt_ts(meta.get("last_activity_at") or meta.get("started_at")),
            "snippet": anchor.get("preview") or "",
            "match_message_id": anchor["id"],
            "bookend_start": [_msg_plain(m) for m in _ua[:3]],
            "bookend_end": [_msg_plain(m) for m in _ua[-3:]],
            "messages": [_msg_plain(m) for m in view["window"]],
            "messages_before": view["messages_before"],
            "messages_after": view["messages_after"],
        })

    return json.dumps({
        "success": True,
        "mode": "discover",
        "query": query,
        "count": len(results),
        "results": results,
    }, ensure_ascii=False)


def _scroll(db, session_id: str, around_message_id: int, window: int) -> str:
    """SCROLL：以锚点消息为中心取窗口。"""
    view = db.get_messages_around(session_id, around_message_id, window=window)
    if not view["window"]:
        return tool_error(
            f"message {around_message_id} not found in session {session_id}"
        )
    return json.dumps({
        "success": True,
        "mode": "scroll",
        "session_id": session_id,
        "around_message_id": around_message_id,
        "window": window,
        "messages": [_msg_plain(m) for m in view["window"]],
        "messages_before": view["messages_before"],
        "messages_after": view["messages_after"],
    }, ensure_ascii=False)


def _read(db, session_id: str) -> str:
    """READ：导出整个会话（大会话截首尾，中间省略标记）。"""
    try:
        msgs = db.get_messages(session_id, include_inactive=True)
    except Exception as exc:
        return tool_error(f"failed to read session {session_id}: {exc}")
    if not msgs:
        return tool_error(f"session not found: {session_id}")

    shown = msgs
    omitted = 0
    if len(msgs) > _READ_HEAD + _READ_TAIL:
        shown = msgs[:_READ_HEAD] + msgs[-_READ_TAIL:]
        omitted = len(msgs) - len(shown)
    plain = [_msg_plain(m) for m in shown]
    if omitted:
        plain.insert(_READ_HEAD, {
            "role": "omitted",
            "content": f"...[{omitted:,} messages omitted in the middle]...",
        })
    return json.dumps({
        "success": True,
        "mode": "read",
        "session_id": session_id,
        "message_count": len(msgs),
        "messages": plain,
    }, ensure_ascii=False)


def _browse(db, limit: int) -> str:
    """BROWSE：按最近活动列出会话。"""
    try:
        sessions = db.list_sessions(limit=limit)
    except Exception as exc:
        return tool_error(f"failed to list sessions: {exc}")
    return json.dumps({
        "success": True,
        "mode": "browse",
        "count": len(sessions),
        "sessions": [{
            "session_id": s["id"],
            "source": s.get("source"),
            "model": s.get("model"),
            "started": _fmt_ts(s.get("started_at")),
            "last_activity": _fmt_ts(s.get("last_activity_at")),
            "message_count": s.get("message_count"),
            "end_reason": s.get("end_reason"),
        } for s in sessions],
    }, ensure_ascii=False)


def session_search(
    query: str = "",
    session_id: Optional[str] = None,
    around_message_id: Optional[int] = None,
    window: int = 5,
    limit: int = 3,
) -> str:
    """单形状工具：模式由传入参数推断（见模块 docstring）。"""
    db = _open_db()
    if db is None:
        return tool_error("Session store unavailable: conversation history is "
                          "not persisted and cannot be searched.")

    # SCROLL 优先：显式锚点胜过一切
    if isinstance(session_id, str) and session_id.strip() \
            and around_message_id is not None:
        try:
            around_message_id = int(around_message_id)
        except (TypeError, ValueError):
            return tool_error("scroll requires integer around_message_id")
        if not isinstance(window, int):
            try:
                window = int(window)
            except (TypeError, ValueError):
                window = 5
        window = max(1, min(window, 20))
        return _scroll(db, session_id.strip(), around_message_id, window)

    # READ：只有 session_id，无锚点 → 导出整个会话
    if isinstance(session_id, str) and session_id.strip():
        return _read(db, session_id.strip())

    # limit clamp [1, 10]
    if not isinstance(limit, int):
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 3
    limit = max(1, min(limit, 10))

    # BROWSE：无 query → 最近会话
    if not query or not isinstance(query, str) or not query.strip():
        return _browse(db, limit)

    # DISCOVERY
    return _discover(db, query.strip(), limit)


def check_session_search_requirements() -> bool:
    """依赖本地 SQLite 会话库；库文件路径存在即可用。"""
    try:
        from hermes_state import default_db_path

        return default_db_path().parent.exists()
    except Exception:
        return False


# --- 模块级自注册（对应原版 tools/session_search_tool.py 的注册）---

SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Search past sessions stored in the local session DB, or scroll "
        "inside one. FTS5-backed retrieval over the SQLite message store. "
        "No LLM calls — every shape returns actual messages from the DB.\n\n"
        "FOUR CALLING SHAPES\n\n"
        "1) DISCOVERY — pass `query`:\n"
        "   session_search(query=\"auth refactor\", limit=3)\n"
        "   Searches across all past sessions and returns the top N matches, "
        "each with session_id, snippet, bookend_start (first 3 user+assistant "
        "messages: the goal), bookend_end (last 3: the decisions), and a ±3 "
        "message window around the matched message.\n\n"
        "2) SCROLL — pass `session_id` + `around_message_id`:\n"
        "   session_search(session_id=\"...\", around_message_id=12345, window=10)\n"
        "   Returns a window of ±`window` messages centered on the anchor. "
        "Use after a discovery call when you need more context. To scroll "
        "forward/backward, re-pass the last/first message id of the returned "
        "window as around_message_id. When messages_before or messages_after "
        "is less than window, you're at the start or end of the session.\n\n"
        "3) READ — pass `session_id` only (no around_message_id):\n"
        "   session_search(session_id=\"...\")\n"
        "   Dumps the whole session by id (first 20 + last 10 messages when "
        "large).\n\n"
        "4) BROWSE — no args:\n"
        "   session_search()\n"
        "   Returns recent sessions chronologically: ids, timestamps, message "
        "counts. Use when the user asks \"what was I working on\" without "
        "naming a topic.\n\n"
        "WHEN TO USE\n\n"
        "Reach for this on questions about conversation history itself, such "
        "as \"what did we do about X\", \"where did we leave Y\", or \"find "
        "the session where Z\". If the user provided a direct source (URL, "
        "file path, live system), inspect that original source first when "
        "accessible; session_search supplies historical context only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "DISCOVERY: search terms across past sessions. "
                               "Omit for BROWSE.",
            },
            "session_id": {
                "type": "string",
                "description": "SCROLL/READ: target session id.",
            },
            "around_message_id": {
                "type": "integer",
                "description": "SCROLL: anchor message id (with session_id).",
            },
            "window": {
                "type": "integer",
                "description": "SCROLL: messages around the anchor (1-20, "
                               "default 5).",
            },
            "limit": {
                "type": "integer",
                "description": "DISCOVERY/BROWSE: max results (1-10, "
                               "default 3).",
            },
        },
        "required": [],
    },
}


registry.register(
    name="session_search",
    toolset="session_search",
    schema=SESSION_SEARCH_SCHEMA,
    handler=session_search,
    check_fn=check_session_search_requirements,
    emoji="🔎",
)
