"""SessionDB 家族模块共享的常量与 DDL（对齐原版 hermes-agent）。

对应原版 hermes_state_common.py 的完整 schema：SCHEMA_SQL（10 表全字段）、
DEFERRED_INDEX_SQL、FTS_SQL（external-content + state_meta 高水位门控）、
FTS_TRIGRAM_SQL（CJK trigram 索引）。my-hermes 的 SessionDB 只实现最小
方法集，但表结构与原版完全一致——可与原版共用同一 state.db，字段互不丢。
"""

from __future__ import annotations

import re
from pathlib import Path

from hermes_constants import get_hermes_home


# schema 版本：对齐原版 v25（无历史迁移链，仅簿记）。
SCHEMA_VERSION = 25


# 用户可控 FTS5 查询的最大长度（对应原版 MAX_FTS5_QUERY_CHARS=2048）。
MAX_FTS5_QUERY_CHARS = 2_048


# FTS5 查询语法中未加引号即报错/改变语义的字符集合（对应原版
# hermes_state_search.py:_FTS5_SPECIAL_CHARS）。`%` 故意不在其中，
# 它由 CJK LIKE 回退路径自行处理。
_FTS5_SPECIAL_CHARS = r'"()+-*{}:~^[]&|!'
_FTS5_SPECIAL_RE = re.compile(f"[{re.escape(_FTS5_SPECIAL_CHARS)}]")


# 完整表结构 DDL（原版逐字）：schema_version / system_prompts / sessions
# （48 列）/ messages（26 列）/ session_model_usage / state_meta /
# gateway_routing / compression_locks / async_delegations + 核心索引。
SCHEMA_SQL = (
"\nCREATE TABLE IF NOT EXISTS schema_version (\n    version INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS system_prompts (\n    hash TEXT PRIMARY KEY,\n    prompt TEXT NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS sessions (\n    id TEXT PRIMARY KEY,\n    source TEXT NOT NULL,\n    user_id TEXT,\n    session_key TEXT,\n    chat_id TEXT,\n    chat_type TEXT,\n    thread_id TEXT,\n    display_name TEXT,\n    origin_json TEXT,\n    expiry_finalized INTEGER DEFAULT 0,\n    model TEXT,\n    model_config TEXT,\n    system_prompt TEXT,\n    system_prompt_hash TEXT,\n    parent_session_id TEXT,\n    started_at REAL NOT NULL,\n    ended_at REAL,\n    end_reason TEXT,\n    message_count INTEGER DEFAULT 0,\n    tool_call_count INTEGER DEFAULT 0,\n    input_tokens INTEGER DEFAULT 0,\n    output_tokens INTEGER DEFAULT 0,\n    cache_read_tokens INTEGER DEFAULT 0,\n    cache_write_tokens INTEGER DEFAULT 0,\n    reasoning_tokens INTEGER DEFAULT 0,\n    cwd TEXT,\n    git_branch TEXT,\n    git_repo_root TEXT,\n    billing_provider TEXT,\n    billing_base_url TEXT,\n    billing_mode TEXT,\n    estimated_cost_usd REAL,\n    actual_cost_usd REAL,\n    cost_status TEXT,\n    cost_source TEXT,\n    pricing_version TEXT,\n    title TEXT,\n    title_source TEXT,\n    last_activity_at REAL,\n    last_activity_description TEXT,\n    last_activity_provenance TEXT,\n    api_call_count INTEGER DEFAULT 0,\n    handoff_state TEXT,\n    handoff_platform TEXT,\n    handoff_error TEXT,\n    compression_failure_cooldown_until REAL,\n    compression_failure_error TEXT,\n    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,\n    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,\n    profile_name TEXT,\n    rewind_count INTEGER NOT NULL DEFAULT 0,\n    archived INTEGER NOT NULL DEFAULT 0,\n    pinned INTEGER NOT NULL DEFAULT 0,\n    last_read_at REAL,\n    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),\n    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)\n);\n\nCREATE TABLE IF NOT EXISTS messages (\n    id INTEGER PRIMARY KEY AUTOINCREMENT,\n    session_id TEXT NOT NULL REFERENCES sessions(id),\n    role TEXT NOT NULL,\n    content TEXT,\n    tool_call_id TEXT,\n    tool_calls TEXT,\n    tool_name TEXT,\n    effect_disposition TEXT,\n    timestamp REAL NOT NULL,\n    token_count INTEGER,\n    finish_reason TEXT,\n    reasoning TEXT,\n    reasoning_content TEXT,\n    reasoning_details TEXT,\n    codex_reasoning_items TEXT,\n    codex_message_items TEXT,\n    platform_message_id TEXT,\n    observed INTEGER DEFAULT 0,\n    active INTEGER NOT NULL DEFAULT 1,\n    compacted INTEGER NOT NULL DEFAULT 0,\n    api_content TEXT,\n    display_kind TEXT,\n    display_metadata TEXT\n);\n\nCREATE TABLE IF NOT EXISTS session_model_usage (\n    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,\n    model TEXT NOT NULL,\n    billing_provider TEXT NOT NULL DEFAULT '',\n    billing_base_url TEXT NOT NULL DEFAULT '',\n    billing_mode TEXT NOT NULL DEFAULT '',\n    task TEXT NOT NULL DEFAULT '',\n    api_call_count INTEGER NOT NULL DEFAULT 0,\n    input_tokens INTEGER NOT NULL DEFAULT 0,\n    output_tokens INTEGER NOT NULL DEFAULT 0,\n    cache_read_tokens INTEGER NOT NULL DEFAULT 0,\n    cache_write_tokens INTEGER NOT NULL DEFAULT 0,\n    reasoning_tokens INTEGER NOT NULL DEFAULT 0,\n    estimated_cost_usd REAL NOT NULL DEFAULT 0,\n    actual_cost_usd REAL NOT NULL DEFAULT 0,\n    cost_status TEXT,\n    cost_source TEXT,\n    first_seen REAL,\n    last_seen REAL,\n    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)\n);\n\nCREATE TABLE IF NOT EXISTS state_meta (\n    key TEXT PRIMARY KEY,\n    value TEXT\n);\n\nCREATE TABLE IF NOT EXISTS gateway_routing (\n    scope TEXT NOT NULL DEFAULT '',\n    session_key TEXT NOT NULL,\n    entry_json TEXT NOT NULL,\n    updated_at REAL NOT NULL,\n    PRIMARY KEY (scope, session_key)\n);\n\nCREATE TABLE IF NOT EXISTS compression_locks (\n    session_id TEXT PRIMARY KEY,\n    holder TEXT NOT NULL,\n    acquired_at REAL NOT NULL,\n    expires_at REAL NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS async_delegations (\n    delegation_id TEXT PRIMARY KEY,\n    origin_session TEXT NOT NULL,\n    origin_ui_session_id TEXT NOT NULL DEFAULT '',\n    parent_session_id TEXT,\n    state TEXT NOT NULL,\n    dispatched_at REAL NOT NULL,\n    completed_at REAL,\n    updated_at REAL NOT NULL,\n    event_json TEXT,\n    result_json TEXT,\n    delivery_state TEXT NOT NULL DEFAULT 'pending',\n    delivery_attempts INTEGER NOT NULL DEFAULT 0,\n    delivered_at REAL,\n    owner_pid INTEGER,\n    owner_started_at INTEGER,\n    task_json TEXT,\n    delivery_claim TEXT,\n    delivery_claimed_at REAL\n);\n\nCREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);\nCREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);\nCREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);\nCREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);\nCREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);\nCREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);\n-- Partial index for the Insights assistant tool-call scan\n-- (agent/insights.py _get_tool_usage / _get_skill_usage): those queries filter\n-- messages by role='assistant' AND tool_calls IS NOT NULL, a small fraction of\n-- rows on a large state.db. role and tool_calls are base columns, so this can\n-- live in SCHEMA_SQL rather than DEFERRED_INDEX_SQL.\nCREATE INDEX IF NOT EXISTS idx_messages_assistant_calls_by_session\n    ON messages(session_id)\n    WHERE role = 'assistant' AND tool_calls IS NOT NULL;\nCREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);\nCREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id);\nCREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model);\nCREATE INDEX IF NOT EXISTS idx_async_delegations_delivery\n    ON async_delegations(delivery_state, completed_at);\n"
)


# 引用后加列（active/compacted 等）的索引，必须在 _reconcile_columns()
# 补齐列之后创建（原版语义）。
DEFERRED_INDEX_SQL = (
'\nCREATE INDEX IF NOT EXISTS idx_messages_session_active\n    ON messages(session_id, active, timestamp);\nCREATE INDEX IF NOT EXISTS idx_messages_active_null\n    ON messages(active) WHERE active IS NULL;\nCREATE INDEX IF NOT EXISTS idx_sessions_session_key\n    ON sessions(session_key, started_at DESC);\nCREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer\n    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);\nCREATE INDEX IF NOT EXISTS idx_sessions_handoff_state\n    ON sessions(handoff_state, started_at);\nCREATE INDEX IF NOT EXISTS idx_sessions_system_prompt_hash\n    ON sessions(system_prompt_hash);\n'
)


# FTS5 全文索引（external-content 风格，原版逐字）：由 messages 触发器
# 维护，state_meta 的 fts_rebuild_high_water / fts_rebuild_progress 做
# 重建期门控（未触发重建时 COALESCE 恒真 = 正常操作）。
FTS_SQL = (
"\nCREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(\n    content,\n    tool_name,\n    tool_calls,\n    content='messages',\n    content_rowid='id'\n);\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages\nWHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                         WHERE key = 'fts_rebuild_high_water'), -1)\n   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                          WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)\n    VALUES (new.id, new.content, new.tool_name, new.tool_calls);\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages\nWHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                         WHERE key = 'fts_rebuild_high_water'), -1)\n   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                          WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)\n    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);\nEND;\n\n-- UPDATE OF skips the trigger entirely for non-content column writes\n-- (status/compacted/observed/etc.), which is stronger than the WHEN gate\n-- alone and avoids FTS I/O saturation on large state.db (#68858 / #73639).\nCREATE TRIGGER IF NOT EXISTS messages_fts_update\nAFTER UPDATE OF content, tool_name, tool_calls ON messages\nWHEN (old.content IS NOT new.content\n    OR old.tool_name IS NOT new.tool_name\n    OR old.tool_calls IS NOT new.tool_calls)\n   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                           WHERE key = 'fts_rebuild_high_water'), -1)\n     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                            WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)\n    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);\n    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)\n    VALUES (new.id, new.content, new.tool_name, new.tool_calls);\nEND;\n"
)


# Trigram FTS5 索引（CJK 子串检索，原版逐字）：经 messages_fts_trigram_src
# 视图排除 role='tool' 行（机器噪音，不参与 CJK 检索）；my-hermes 的
# search_messages 暂不直接使用它（CJK 走 LIKE 回退），建表只为与
# 原版库结构一致。
FTS_TRIGRAM_SQL = (
"\nCREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS\n    SELECT id, role, content, tool_name, tool_calls\n    FROM messages\n    WHERE role <> 'tool';\n\nCREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(\n    content,\n    tool_name,\n    tool_calls,\n    content='messages_fts_trigram_src',\n    content_rowid='id',\n    tokenize='trigram'\n);\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages\nWHEN new.role <> 'tool'\n   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                           WHERE key = 'fts_rebuild_high_water'), -1)\n     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                            WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)\n    VALUES (new.id, new.content, new.tool_name, new.tool_calls);\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages\nWHEN old.role <> 'tool'\n   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                           WHERE key = 'fts_rebuild_high_water'), -1)\n     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                            WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)\n    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);\nEND;\n\nCREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update\nAFTER UPDATE OF content, tool_name, tool_calls, role ON messages\nWHEN (old.content IS NOT new.content\n    OR old.tool_name IS NOT new.tool_name\n    OR old.tool_calls IS NOT new.tool_calls\n    OR old.role IS NOT new.role)\n   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                           WHERE key = 'fts_rebuild_high_water'), -1)\n     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta\n                            WHERE key = 'fts_rebuild_progress'), -1))\nBEGIN\n    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)\n    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls\n    WHERE old.role <> 'tool';\n    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)\n    SELECT new.id, new.content, new.tool_name, new.tool_calls\n    WHERE new.role <> 'tool';\nEND;\n"
)


def escape_like(text: str) -> str:
    """转义 SQL LIKE 通配符，使模式匹配字面文本（对应原版 escape_like）。

    `%` 与 `_` 是 LIKE 通配符，经常出现在被检索文本（文件路径、分支名等）
    里；与 SQL 的 ESCAPE 子句配合使用，避免匹配意外放宽。
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
    "DEFERRED_INDEX_SQL",
    "FTS_SQL",
    "FTS_TRIGRAM_SQL",
    "SCHEMA_VERSION",
    "MAX_FTS5_QUERY_CHARS",
    "_FTS5_SPECIAL_RE",
    "escape_like",
    "default_db_path",
    "get_hermes_home",
]
