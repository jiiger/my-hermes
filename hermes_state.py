"""最小版 SessionDB：SQLite 会话存储 + FTS5 全文检索。

对应原版 hermes_state.py 的 SessionDB 精简移植，只保留本任务需要的
最小子集：创建/查询会话、批量追加消息、系统提示快照、结束会话、
FTS5 搜索。线程安全采用单写锁（``self._lock``）+ 单连接
（``check_same_thread=False``），每个方法在锁内用独立游标。

与原版的差异（明确裁剪）：
- 不做 gateway / title / billing / delegation / portability / compression
  lock / WAL 修复 / 导入导出 / session vacuum；
- 不做 read-only 打开、token 计数队列、FTS rebuild 状态机、
  trigram / cjk 分词扩展；
- 写失败不静默吞掉：异常原样抛给上层（由 AIAgent 的 flush 包装成
  ``_incremental_persistence_failed``）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_state_common import (
    DEFERRED_INDEX_SQL,
    FTS_SQL,
    FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    _FTS5_SPECIAL_RE,
    default_db_path,
    escape_like,
)

logger = logging.getLogger(__name__)


# ── 旧库升级迁移：SCHEMA_SQL 已含全部列，这些清单只用于给"上一版最小
#    库"补列（ALTER TABLE ADD COLUMN），保证老库打开后与原版结构一致。──
_SESSIONS_UPGRADE_COLUMNS = [
    ("origin_json", "TEXT"),
    ("expiry_finalized", "INTEGER DEFAULT 0"),
    ("model_config", "TEXT"),
    ("system_prompt_hash", "TEXT"),
    ("parent_session_id", "TEXT"),
    ("input_tokens", "INTEGER DEFAULT 0"),
    ("output_tokens", "INTEGER DEFAULT 0"),
    ("cache_read_tokens", "INTEGER DEFAULT 0"),
    ("cache_write_tokens", "INTEGER DEFAULT 0"),
    ("reasoning_tokens", "INTEGER DEFAULT 0"),
    ("cwd", "TEXT"),
    ("git_branch", "TEXT"),
    ("git_repo_root", "TEXT"),
    ("billing_provider", "TEXT"),
    ("billing_base_url", "TEXT"),
    ("billing_mode", "TEXT"),
    ("estimated_cost_usd", "REAL"),
    ("actual_cost_usd", "REAL"),
    ("cost_status", "TEXT"),
    ("cost_source", "TEXT"),
    ("pricing_version", "TEXT"),
    ("title", "TEXT"),
    ("title_source", "TEXT"),
    ("last_activity_description", "TEXT"),
    ("last_activity_provenance", "TEXT"),
    ("handoff_state", "TEXT"),
    ("handoff_platform", "TEXT"),
    ("handoff_error", "TEXT"),
    ("compression_failure_cooldown_until", "REAL"),
    ("compression_failure_error", "TEXT"),
    ("compression_fallback_streak", "INTEGER NOT NULL DEFAULT 0"),
    ("compression_ineffective_count", "INTEGER NOT NULL DEFAULT 0"),
    ("profile_name", "TEXT"),
    ("rewind_count", "INTEGER NOT NULL DEFAULT 0"),
    ("archived", "INTEGER NOT NULL DEFAULT 0"),
    ("pinned", "INTEGER NOT NULL DEFAULT 0"),
    ("last_read_at", "REAL"),
]

_MESSAGES_UPGRADE_COLUMNS = [
    ("effect_disposition", "TEXT"),
    ("token_count", "INTEGER"),
    ("reasoning_details", "TEXT"),
    ("codex_reasoning_items", "TEXT"),
    ("codex_message_items", "TEXT"),
    ("platform_message_id", "TEXT"),
    ("observed", "INTEGER DEFAULT 0"),
]


def _system_prompt_hash(system_prompt: str) -> str:
    """系统提示快照的 sha256 指纹（对应原版 hermes_state.py:164）。"""
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def _scrub_surrogates(value: Any) -> Any:
    """去掉字符串中的孤立代理对字符，防止 sqlite3 绑定时报编码错误。

    对应原版 hermes_state.py:216 的 _scrub_surrogates（精简为内联实现）。
    """
    if not isinstance(value, str):
        return value
    if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
        return value
    return "".join(ch if not (0xD800 <= ord(ch) <= 0xDFFF) else "\ufffd" for ch in value)


class SessionDB:
    """SQLite 会话存储（最小版），线程安全：所有访问经 ``self._lock``。"""

    # 结构化 content（list/dict 多模态消息）的 JSON 前缀哨兵。
    # NUL 字节在正常文本中不合法，不会与真实内容冲突（对应原版
    # SessionDB._CONTENT_JSON_PREFIX）。
    _CONTENT_JSON_PREFIX = "\x00json:"

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            # 手动管理事务（与 BEGIN/COMMIT 语义一致，避免隐式事务干扰）
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        # WAL：读写不互相阻塞；个别环境（只读文件系统/旧 sqlite）降级 DELETE
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        # 幂等建表：可重复打开同一数据库。旧最小库缺列时（SCHEMA_SQL 里的
        # 索引/外键引用新列）先补列再重试，避免 "no such column"。
        try:
            self._conn.executescript(SCHEMA_SQL)
        except sqlite3.OperationalError:
            self._reconcile_columns()
            self._conn.executescript(SCHEMA_SQL)
        # 列迁移：旧最小库升级时补齐到原版全列（旧行按默认值处理）
        self._reconcile_columns()
        # 引用后加列的索引（active/compacted 等）在补列之后创建（原版语义）
        try:
            self._conn.executescript(DEFERRED_INDEX_SQL)
        except sqlite3.Error as exc:
            logger.warning("Deferred index creation failed: %s", exc)
        try:
            self._conn.executescript(FTS_SQL)
        except sqlite3.Error as exc:
            # FTS5 不可用（编译期关闭）时降级：消息写入不受影响，仅搜索不可用
            logger.warning("FTS5 unavailable, session search disabled: %s", exc)
        try:
            self._conn.executescript(FTS_TRIGRAM_SQL)
        except sqlite3.Error as exc:
            # trigram 分词器在个别 sqlite 编译期关闭；降级仅影响 CJK 子串索引
            logger.warning("FTS trigram unavailable, CJK substring search disabled: %s", exc)
        # schema 版本簿记（幂等）
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
            )
            self._conn.commit()

    def _reconcile_columns(self) -> None:
        """补齐旧库缺失列到原版全字段（PRAGMA 检查 + ALTER TABLE ADD COLUMN）。

        SCHEMA_SQL 新库已含全部列；本方法只处理"上一版最小库"升级：
        逐表检查缺失列并补齐（含 active/compacted），存量行按默认值处理
        （历史消息视为 active=1，直到首次压缩才被归档）。
        """
        with self._conn:
            for table, columns in (
                ("messages", _MESSAGES_UPGRADE_COLUMNS),
                ("sessions", _SESSIONS_UPGRADE_COLUMNS),
            ):
                cols = {
                    r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")
                }
                for name, ddl in columns:
                    if name not in cols:
                        self._conn.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"
                        )
            # 上一版最小库遗留的 active/compacted 补列（历史原因，幂等）
            mcols = {
                r[1] for r in self._conn.execute("PRAGMA table_info(messages)")
            }
            if "active" not in mcols:
                self._conn.execute(
                    "ALTER TABLE messages ADD COLUMN "
                    "active INTEGER NOT NULL DEFAULT 1"
                )
            if "compacted" not in mcols:
                self._conn.execute(
                    "ALTER TABLE messages ADD COLUMN "
                    "compacted INTEGER NOT NULL DEFAULT 0"
                )

    # ───────────────────────── 基础生命周期 ─────────────────────────

    def close(self) -> None:
        """关闭数据库连接（幂等）。"""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def _execute_write(self, fn):
        """在锁内开启一个写事务执行 ``fn(conn)``，返回其返回值。

        事务原子性由 sqlite3 连接上下文保证（提交失败自动回滚）；
        异常原样向上抛出，不静默伪造成功。
        """
        with self._lock:
            if self._conn is None:
                raise sqlite3.ProgrammingError("SessionDB is closed")
            with self._conn:
                return fn(self._conn)

    def _execute_read(self, fn):
        """在锁内执行只读 ``fn(conn)``。"""
        with self._lock:
            if self._conn is None:
                raise sqlite3.ProgrammingError("SessionDB is closed")
            return fn(self._conn)

    # ───────────────────────── 消息编码 ─────────────────────────

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """把结构化（list/dict）content 序列化为可绑定 sqlite 的字符串。

        对应原版 SessionDB._encode_content：str 原样（清理 surrogate）；
        list/dict 加 ``\x00json:`` 前缀存 JSON；其余标量原样返回。
        """
        if isinstance(content, str):
            return _scrub_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            return _scrub_surrogates(str(content))

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """反转 :meth:`_encode_content`；标量原样返回。"""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                return content
        return content

    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """把 display_metadata（dict 或已序列化 JSON 字符串）存为 JSON 文本。"""
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            try:
                parsed = json.loads(display_metadata)
            except (json.JSONDecodeError, TypeError):
                return None
            if not isinstance(parsed, dict):
                return None
            return json.dumps(parsed)
        if isinstance(display_metadata, dict):
            return json.dumps(display_metadata)
        return None

    # ───────────────────────── 会话 ─────────────────────────

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        """把系统提示快照写入 system_prompts 表（hash 去重），返回 hash。"""
        if system_prompt is None:
            return None
        _hash = _system_prompt_hash(system_prompt)
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (_hash, _scrub_surrogates(system_prompt)),
        )
        return _hash

    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        session_key: Optional[str] = None,
        chat_id: str = None,
        chat_type: str = None,
        thread_id: str = None,
        display_name: str = None,
        parent_session_id: str = None,
        cwd: str = None,
        profile_name: str = None,
        git_repo_root: str = None,
        origin_json: str = None,
        **kwargs,
    ) -> None:
        """插入会话行；已存在时幂等补全仍为 NULL 的字段（对应原版
        hermes_state.py:3499 _insert_session_row 的 COALESCE 语义）。

        字段集与 SCHEMA_SQL 对齐：parent_session_id / cwd / profile_name /
        git_repo_root / origin_json / model_config 均可选存储；其余
        原版扩展字段（billing/title/handoff 等）由上层按需 UPDATE。
        """
        del kwargs  # 其余未知参数忽略

        def _do(conn):
            system_prompt_hash = None
            if system_prompt is not None:
                system_prompt_hash = self._store_system_prompt(
                    conn, system_prompt
                )
            model_config_json = (
                json.dumps(model_config, ensure_ascii=False)
                if isinstance(model_config, dict) and model_config
                else None
            )
            conn.execute(
                """INSERT INTO sessions (
                       id, source, user_id, session_key, chat_id, chat_type,
                       thread_id, display_name, origin_json, model, model_config,
                       system_prompt, system_prompt_hash, parent_session_id,
                       cwd, profile_name, git_repo_root, started_at
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = COALESCE(sessions.model_config, excluded.model_config),
                       system_prompt = COALESCE(sessions.system_prompt, excluded.system_prompt),
                       system_prompt_hash = COALESCE(sessions.system_prompt_hash, excluded.system_prompt_hash),
                       user_id = COALESCE(sessions.user_id, excluded.user_id),
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       display_name = COALESCE(sessions.display_name, excluded.display_name),
                       origin_json = COALESCE(sessions.origin_json, excluded.origin_json),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root)""",
                (
                    session_id,
                    source,
                    user_id,
                    session_key,
                    chat_id,
                    chat_type,
                    thread_id,
                    display_name,
                    origin_json,
                    model,
                    model_config_json,
                    system_prompt,
                    system_prompt_hash,
                    parent_session_id,
                    cwd,
                    profile_name,
                    git_repo_root,
                    time.time(),
                ),
            )

        self._execute_write(_do)

    def create_session(self, session_id: str, source: str = "cli", **kwargs) -> str:
        """创建会话记录，返回 session_id（对应原版 hermes_state.py:3673）。

        幂等：session_id 已存在时只补全缺失字段，不覆盖已有值。
        """
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 获取会话（对应原版 hermes_state.py:6246；无则返回 None）。"""
        def _do(conn):
            return conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()

        row = self._execute_read(_do)
        return dict(row) if row is not None else None

    def end_session(self, session_id: str, end_reason: str = "completed") -> None:
        """结束会话（对应原版 hermes_state.py:4587）。

        已结束（ended_at 非空）时 no-op，首个 end_reason 生效。
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )

        self._execute_write(_do)

    def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> None:
        """保存完整系统提示快照（对应原版 hermes_state.py:5313 的最小版）。

        同时写入 system_prompts 快照表（hash 去重）与 sessions.system_prompt
        字段，保证 get_session 无需 JOIN 即可读到当前提示。
        """
        def _do(conn):
            if system_prompt is None:
                conn.execute(
                    "UPDATE sessions SET system_prompt = NULL, "
                    "system_prompt_hash = NULL WHERE id = ?",
                    (session_id,),
                )
                return
            _hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                "UPDATE sessions SET system_prompt = ?, system_prompt_hash = ? "
                "WHERE id = ?",
                (_scrub_surrogates(system_prompt), _hash, session_id),
            )

        self._execute_write(_do)

    # ───────────────────────── 消息 ─────────────────────────

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """按插入顺序（id ASC）读取会话消息（对应原版 hermes_state.py:8207）。

        默认只返回 active=1 消息（当前工作上下文）；``include_inactive=True``
        时返回全部行（含压缩归档的 compacted 历史与 rewind 行）。
        content 反序列化（多模态 JSON）、tool_calls 反序列化为列表。
        """
        sql = "SELECT * FROM messages WHERE session_id = ?"
        if not include_inactive:
            sql += " AND active = 1"
        sql += " ORDER BY id ASC"
        params: List[Any] = [session_id]
        if limit is not None or offset:
            sql += " LIMIT ? OFFSET ?"
            params.append(-1 if limit is None else limit)
            params.append(offset)

        def _do(conn):
            return conn.execute(sql, params).fetchall()

        rows = self._execute_read(_do)
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = None
            result.append(msg)
        return result

    def get_messages_as_conversation(
        self,
        session_id: str,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """以对话格式恢复消息，供回合恢复历史使用（对应原版
        hermes_state.py:8469 get_messages_as_conversation 的最小版）。

        默认只返回 active=1 消息（当前工作上下文），按自增 id 排序；
        ``include_inactive=True`` 时含压缩归档历史（审计/导出用）。
        输出可直接作为模型循环的 live 消息（role/content/tool_calls/
        timestamp/finish_reason/reasoning 等，内部字段已解码）。
        """
        sql = (
            "SELECT id, role, content, tool_call_id, tool_calls, tool_name, "
            "timestamp, finish_reason, reasoning, reasoning_content, "
            "api_content, display_kind, display_metadata "
            "FROM messages WHERE session_id = ?"
        )
        if not include_inactive:
            sql += " AND active = 1"
        sql += " ORDER BY id ASC"

        def _do(conn):
            return conn.execute(sql, (session_id,)).fetchall()

        rows = self._execute_read(_do)
        result: List[Dict[str, Any]] = []
        for row in rows:
            msg: Dict[str, Any] = {"role": row["role"], "content": self._decode_content(row["content"])}
            for key in (
                "tool_call_id",
                "tool_name",
                "timestamp",
                "finish_reason",
                "reasoning",
                "reasoning_content",
                "api_content",
                "display_kind",
            ):
                if row[key] is not None:
                    msg[key] = row[key]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    msg["tool_calls"] = None
            if row["display_metadata"]:
                decoded = self._decode_display_metadata(row["display_metadata"])
                if decoded is not None:
                    msg["display_metadata"] = decoded
            result.append(msg)
        return result

    @staticmethod
    def _decode_display_metadata(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        """反序列化 display_metadata JSON 列（对应原版解码路径的最小版）。"""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _insert_message_rows(
        self, conn, session_id: str, messages: List[Dict[str, Any]]
    ) -> tuple:
        """把消息列表插入 messages 表（在调用方事务内执行）。

        返回 ``(inserted_count, tool_calls_total)``。对应原版
        hermes_state.py:7910 _insert_message_rows 的最小字段版。
        """
        now_ts = time.time()
        inserted = 0
        tool_calls_total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            tool_calls = msg.get("tool_calls")
            message_timestamp = now_ts
            ts_value = msg.get("timestamp")
            if ts_value is not None:
                try:
                    if hasattr(ts_value, "timestamp"):
                        message_timestamp = float(ts_value.timestamp())
                    else:
                        message_timestamp = float(ts_value)
                except (TypeError, ValueError):
                    pass
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            api_content = msg.get("api_content")
            conn.execute(
                """INSERT INTO messages (
                       session_id, role, content, tool_call_id, tool_calls,
                       tool_name, effect_disposition, timestamp, token_count,
                       finish_reason, reasoning, reasoning_content,
                       reasoning_details, codex_reasoning_items,
                       codex_message_items, platform_message_id, observed,
                       api_content, display_kind, display_metadata,
                       active, compacted
                   )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    self._encode_content(msg.get("content")),
                    msg.get("tool_call_id"),
                    tool_calls_json,
                    _scrub_surrogates(msg.get("tool_name")),
                    _scrub_surrogates(msg.get("effect_disposition"))
                    if isinstance(msg.get("effect_disposition"), str)
                    else None,
                    message_timestamp,
                    msg.get("token_count"),
                    msg.get("finish_reason"),
                    _scrub_surrogates(msg.get("reasoning")) if role == "assistant" else None,
                    _scrub_surrogates(msg.get("reasoning_content"))
                    if role == "assistant"
                    else None,
                    _scrub_surrogates(msg.get("reasoning_details"))
                    if role == "assistant"
                    else None,
                    _scrub_surrogates(msg.get("codex_reasoning_items"))
                    if isinstance(msg.get("codex_reasoning_items"), str)
                    else None,
                    _scrub_surrogates(msg.get("codex_message_items"))
                    if isinstance(msg.get("codex_message_items"), str)
                    else None,
                    _scrub_surrogates(msg.get("platform_message_id"))
                    if isinstance(msg.get("platform_message_id"), str)
                    else None,
                    1 if msg.get("observed") else 0,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(msg.get("display_kind"))
                    if isinstance(msg.get("display_kind"), str)
                    else None,
                    self._encode_display_metadata(msg.get("display_metadata")),
                    # 新写入的行恒为当前工作上下文（active=1, compacted=0）
                    1,
                    0,
                ),
            )
            inserted += 1
            if tool_calls is not None:
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else 1
                )
            # 保证同批消息时间戳严格递增（对齐原版），稳定后续读取顺序
            now_ts = max(now_ts + 1e-6, message_timestamp + 1e-6)
        return inserted, tool_calls_total

    def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> int:
        """在一个写事务内批量追加消息，并更新会话计数（对应原版
        hermes_state.py:7595 append_messages_batch 的最小版）。

        原子性：整批消息全部落库或全部不落库。返回插入的消息条数。
        写失败抛出原始异常（由上层决定是否标记持久化失败）。
        """
        del kwargs  # 原版 compression_lock_holder / chunk_rows 最小版不实现
        if not messages:
            return 0

        def _do(conn):
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, messages
            )
            if tool_calls_total > 0:
                conn.execute(
                    """UPDATE sessions SET
                       message_count = message_count + ?,
                       tool_call_count = tool_call_count + ?,
                       last_activity_at = ?
                       WHERE id = ?""",
                    (inserted, tool_calls_total, time.time(), session_id),
                )
            else:
                conn.execute(
                    """UPDATE sessions SET
                       message_count = message_count + ?,
                       last_activity_at = ?
                       WHERE id = ?""",
                    (inserted, time.time(), session_id),
                )
            return inserted

        return self._execute_write(_do)

    def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: List[Dict[str, Any]],
        model_config_patch: Optional[Dict[str, Any]] = None,
    ) -> int:
        """非破坏性的就地压缩提交（对应原版 hermes_state.py:8101
        archive_and_compact 的最小版）。

        在一个写事务内完成：
        1. 当前所有 active=1 消息软归档为 ``active=0, compacted=1``
           （保留在库中、不删除、不重排，仍可搜索、可恢复）；
        2. 把压缩后的消息（summary + 保留的 head/tail）作为新的
           ``active=1, compacted=0`` 行插入；
        3. 更新 sessions.message_count / tool_call_count 为新的 active 计数。

        原子性：任一步失败整体回滚，DB 保持压缩前状态（下次可重试）。
        ``model_config_patch`` 为兼容原版签名保留；my-hermes 的 sessions
        表无 model_config 列，该参数被忽略。
        返回新的 active 消息条数。
        """
        del model_config_patch  # 最小版 sessions 无 model_config 列，忽略

        def _do(conn):
            # 1. 软归档当前活动转录（内容保留，标记压缩替代）
            conn.execute(
                "UPDATE messages SET active = 0, compacted = 1 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            # 2. 压缩后的消息插入为新的 active 集
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, compacted_messages
            )
            # 3. message_count 只统计新的 active 集（归档行不计入）
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? "
                "WHERE id = ?",
                (inserted, tool_calls_total, session_id),
            )
            return inserted

        return self._execute_write(_do)

    # ───────────────────────── 搜索 ─────────────────────────

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """净化用户输入，安全用于 FTS5 MATCH（对应原版
        hermes_state_search.py:1174 _sanitize_fts5_query 的精简版）。

        策略：截断到 MAX_FTS5_QUERY_CHARS；剥离 FTS5 特殊字符；清理首尾
        悬空的布尔运算符；把带点/连线的裸词包成引号短语，避免被分词拆开。
        """
        from hermes_state_common import MAX_FTS5_QUERY_CHARS

        query = (query or "")[:MAX_FTS5_QUERY_CHARS]
        # 成对引号短语先保护起来，避免内部内容被剥离
        quoted_parts: list = []
        pieces: list = []
        i = 0
        while i < len(query):
            ch = query[i]
            if ch != '"':
                pieces.append(ch)
                i += 1
                continue
            end = query.find('"', i + 1)
            if end == -1:
                pieces.append(" ")
                i += 1
                continue
            quoted_parts.append(query[i:end + 1])
            pieces.append(f"\x00Q{len(quoted_parts) - 1}\x00")
            i = end + 1
        sanitized = "".join(pieces)
        sanitized = _FTS5_SPECIAL_RE.sub(" ", sanitized)
        sanitized = sanitized.replace("%", " ")
        sanitized = _re_sub_stars(sanitized)
        sanitized = _re_sub_leading_trailing_bool(sanitized)
        sanitized = _re_sub_dotted_terms(sanitized)
        for idx, quoted in enumerate(quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{idx}\x00", quoted)
        return sanitized.strip()

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """文本是否包含中日韩字符（对应原版 _contains_cjk 的码点范围）。"""
        for ch in text:
            cp = ord(ch)
            if (
                0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
                or 0x3400 <= cp <= 0x4DBF  # Extension A
                or 0x20000 <= cp <= 0x2A6DF  # Extension B
                or 0x3000 <= cp <= 0x303F  # CJK Symbols
                or 0x3040 <= cp <= 0x309F  # Hiragana
                or 0x30A0 <= cp <= 0x30FF  # Katakana
                or 0xAC00 <= cp <= 0xD7AF  # Hangul
            ):
                return True
        return False

    def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """跨会话（或限定会话）全文检索消息（对应原版
        hermes_state_search.py:1410 search_messages 的最小 API）。

        非 CJK 查询走 FTS5 MATCH（BM25 排序）；CJK 查询走 LIKE 子串回退
        （unicode61 分词会把中文拆成单字，短语匹配不可靠）。
        默认可见范围 = ``active=1 OR compacted=1``（当前上下文 + 压缩归档
        历史）；只有 rewind/undo（active=0, compacted=0）被排除。
        返回每条消息的 session_id / id / role / content / timestamp / preview。
        """
        if not query or not query.strip():
            return []
        limit = max(1, min(int(limit or 20), 500))
        sanitized = self._sanitize_fts5_query(query)
        if not sanitized:
            return []

        def _do(conn):
            if self._contains_cjk(sanitized):
                return self._search_like(conn, query, session_id, limit)
            try:
                return self._search_fts5(conn, sanitized, session_id, limit)
            except sqlite3.OperationalError:
                # FTS 查询语法仍可能失败（如极端输入），降级 LIKE
                return self._search_like(conn, query, session_id, limit)

        return self._execute_read(_do)

    def _search_fts5(
        self, conn, sanitized_query: str, session_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """FTS5 主查询：JOIN messages 取行，BM25 rank 排序。"""
        where = ["messages_fts MATCH ?", "(m.active = 1 OR m.compacted = 1)"]
        params: List[Any] = [sanitized_query]
        if session_id is not None:
            where.append("m.session_id = ?")
            params.append(session_id)
        params.append(limit)
        rows = conn.execute(
            f"""SELECT m.id, m.session_id, m.role, m.content, m.timestamp
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                WHERE {' AND '.join(where)}
                ORDER BY rank
                LIMIT ?""",
            params,
        ).fetchall()
        return self._shape_search_results(rows)

    def _search_like(
        self, conn, query: str, session_id: Optional[str], limit: int
    ) -> List[Dict[str, Any]]:
        """LIKE 子串回退：CJK 查询与 FTS 异常时的兜底（content 子串匹配）。"""
        escaped = escape_like(query.strip())
        where = ["(COALESCE(m.content, '') LIKE ? ESCAPE '\\' "
                 "OR COALESCE(m.tool_name, '') LIKE ? ESCAPE '\\' "
                 "OR COALESCE(m.tool_calls, '') LIKE ? ESCAPE '\\')",
                 "(m.active = 1 OR m.compacted = 1)"]
        params: List[Any] = [f"%{escaped}%"] * 3
        if session_id is not None:
            where.append("m.session_id = ?")
            params.append(session_id)
        params.append(limit)
        rows = conn.execute(
            f"""SELECT m.id, m.session_id, m.role, m.content, m.timestamp
                FROM messages m
                WHERE {' AND '.join(where)}
                ORDER BY m.id DESC
                LIMIT ?""",
            params,
        ).fetchall()
        return self._shape_search_results(rows)

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """以某条消息 id 为锚点取窗口（对齐原版 hermes_state.py:8300 精简版）。

        返回 {"window": [消息...], "messages_before": n, "messages_after": n}，
        window 内消息按 id 升序。around_message_id 不在该会话时返回空窗口。
        供 session_search 的 DISCOVERY（锚定 FTS 命中）与 SCROLL（任意锚点）
        两种模式使用；messages_before/after 用于判断会话边界。
        """
        if window < 0:
            window = 0

        def _do(conn):
            anchor_exists = conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor_exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}
            before_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()
            rows = list(reversed(before_rows)) + list(after_rows)
            result = []
            for row in rows:
                msg = dict(row)
                if "content" in msg:
                    msg["content"] = self._decode_content(msg["content"])
                if msg.get("tool_calls"):
                    try:
                        msg["tool_calls"] = json.loads(msg["tool_calls"])
                    except (json.JSONDecodeError, TypeError):
                        msg["tool_calls"] = None
                result.append(msg)
            return {
                "window": result,
                "messages_before": max(0, len(before_rows) - 1),
                "messages_after": len(after_rows),
            }

        return self._execute_read(_do)

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """按最近活动列出会话（供 session_search 的 BROWSE 模式）。"""
        limit = max(1, min(int(limit or 20), 200))

        def _do(conn):
            return conn.execute(
                "SELECT id, source, model, started_at, ended_at, end_reason, "
                "message_count, tool_call_count, last_activity_at "
                "FROM sessions "
                "ORDER BY COALESCE(last_activity_at, started_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()

        rows = self._execute_read(_do)
        return [dict(row) for row in rows]

    @staticmethod
    def _shape_search_results(rows) -> List[Dict[str, Any]]:
        """把搜索结果行统一成公开字段形状，并生成可读 preview。"""
        result = []
        for row in rows:
            content = SessionDB._decode_content(row["content"])
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            preview = (text[:57] + "...") if len(text) > 60 else text
            result.append({
                "session_id": row["session_id"],
                "id": row["id"],
                "role": row["role"],
                "content": content,
                "timestamp": row["timestamp"],
                "preview": preview,
            })
        return result


# ── FTS5 查询净化辅助 ──
def _re_sub_stars(sanitized: str) -> str:
    sanitized = re.sub(r"\*+", "*", sanitized)
    return re.sub(r"(^|\s)\*", r"\1", sanitized)


def _re_sub_leading_trailing_bool(sanitized: str) -> str:
    sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
    return re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())


def _re_sub_dotted_terms(sanitized: str) -> str:
    return re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)


__all__ = ["SessionDB", "_system_prompt_hash"]
