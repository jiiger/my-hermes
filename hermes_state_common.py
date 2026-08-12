"""SessionDB 家族模块共享的常量与 DDL（最小移植版）。

对应原版 hermes_state_common.py 的精简子集：只保留本任务需要的最小
schema（schema_version / system_prompts / sessions / messages / FTS5
messages_fts）与搜索安全所需的常量。原版中的 preview、skill、gateway、
billing、delegation、portability、compression lock 等一律不移植。

my-hermes 已有 ``hermes_constants.get_hermes_home``，这里直接 re-export，
供 hermes_state 解析默认数据库路径，避免与 hermes_state 形成循环导入。
"""

from __future__ import annotations

import re
from pathlib import Path

from hermes_constants import get_hermes_home


# 最小 schema 版本号（独立于原版 v25；本实现没有历史迁移链）。
SCHEMA_VERSION = 1


# 用户可控 FTS5 查询的最大长度（对应原版 MAX_FTS5_QUERY_CHARS=2048）。
MAX_FTS5_QUERY_CHARS = 2_048


# FTS5 查询语法中未加引号即报错/改变语义的字符集合（对应原版
# hermes_state_search.py:_FTS5_SPECIAL_CHARS）。``%`` 故意不在其中，
# 它由 CJK LIKE 回退路径自行处理。
_FTS5_SPECIAL_CHARS = r'"()+-*{}:~^[]&|!'
_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")


# 表结构 DDL（最小版）：只含 schema_version、system_prompts、sessions、
# messages 与必要索引。字段集与原版对齐但大幅裁剪：
#  - sessions 只保留任务指定的最小字段；
#  - messages 只保留任务指定的最小字段；
#  - 不建 gateway/billing/delegation/portability/compression 相关表。
# 全部使用 IF NOT EXISTS，保证幂等、可重复打开。
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    model TEXT,
    system_prompt TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    api_call_count INTEGER DEFAULT 0,
    last_activity_at REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0
);

-- 消息顺序按自增 id（真实插入顺序），不按 timestamp：系统时间可能回拨，
-- timestamp 排序会破坏 tool-call / tool-result 的相邻关系（对应原版
-- get_messages / get_messages_as_conversation 的 ORDER BY id 语义）。
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


# FTS5 全文索引（最小版）：external-content 风格（对应原版 FTS_SQL 的
# 精简形态），由 messages 上的触发器维护，至少索引 content；
# tool_name / tool_calls 一并索引以便工具相关检索。
# 砍掉原版的 fts_rebuild 状态机（state_meta 高水位）、trigram、cjk 扩展。
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages
WHEN (old.content IS NOT new.content
   OR old.tool_name IS NOT new.tool_name
   OR old.tool_calls IS NOT new.tool_calls)
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
"""


def escape_like(text: str) -> str:
    """转义 SQL LIKE 通配符，使模式匹配字面文本（对应原版 escape_like）。

    ``%`` / ``_`` 是 LIKE 的通配符，而它们经常出现在被检索文本
    （文件路径、分支名等）里；配合 ``ESCAPE '\\'`` 使用避免匹配意外放宽。
    """
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def default_db_path() -> Path:
    """返回默认 state.db 路径：``$HERMES_HOME/state.db``。

    在调用时解析 HERMES_HOME（对应原版 hermes_state._default_db_path 的
    语义），保证测试里重定向 HERMES_HOME 后取到的是新路径。
    """
    return get_hermes_home() / "state.db"


__all__ = [
    "SCHEMA_SQL",
    "FTS_SQL",
    "SCHEMA_VERSION",
    "MAX_FTS5_QUERY_CHARS",
    "_FTS5_SPECIAL_RE",
    "escape_like",
    "default_db_path",
    "get_hermes_home",
]
