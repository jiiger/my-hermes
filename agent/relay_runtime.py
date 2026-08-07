"""由Hermes代理核心所拥有的、针对特定配置文件的NeMo Relay运行时环境"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import importlib
import inspect
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
_PROFILE_KEY_CACHE: dict[str, str] = {}


@dataclass
class RelaySession:
    """一个由Hermes会话拥有的独立Relay作用域堆栈"""

    session_id: str
    parent_session_id: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    closing: bool = False
    handle: Any = None
    context: contextvars.Context | None = None


@dataclass(frozen=True)
class NoopRelayRuntime:
    """这是一个降级存根(Stub)类，当平台装不了 nemo_relay 轮子(比如没有预编译的 wheel、架构不支持、或者用户没装)时，my-hermes 不会炸，而是用这个"空壳"继续跑。"""

    profile_key: str
    reason: str

    @property
    def available(self) -> bool:
        return False

    def apply_tool_request_intercepts(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del session_id, tool_name
        return args

    @staticmethod
    def retain_managed_execution(consumer: str) -> None:
        del consumer

    @staticmethod
    def release_managed_execution(consumer: str) -> None:
        del consumer

    @staticmethod
    def managed_execution_enabled() -> bool:
        return False

    def shutdown(self) -> None:
        """No resources are allocated on unsupported platforms."""


class RelayRuntime:
    """独立于任何导出器或插件，拥有自己的中继会话作用域"""

    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock = threading.RLock()
        self._sessions: dict[str, RelaySession] = {}
        self._subagent_parents: dict[str, str] = {}
        self._subagent_parent_handles: dict[str, Any] = {}
        self._execution_consumers_lock = threading.RLock()
        self._execution_consumers: set[str] = set()
        self._shutdown_registered = True
        atexit.register(self.shutdown)

    def register_subagent(
        self,
        event: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RelaySession | None:
        """Open a child Agent scope under its spawning turn when available."""
        parent_session_id = str(event.get("parent_session_id") or "")
        child_session_id = str(event.get("child_session_id") or "")
        if (
            not parent_session_id
            or not child_session_id
            or parent_session_id == child_session_id
        ):
            return None
        parent = self.ensure_session({"session_id": parent_session_id})
        parent_handle = None if parent is None else parent.handle
        turn = active_turn(parent_session_id)
        if (
            turn is not None
            and not turn.closed
            and turn.handle is not None
            and turn.lease.host is self
            and turn.lease.session is not None
            and turn.lease.session.session_id == parent_session_id
        ):
            parent_handle = turn.handle
        with self._sessions_lock:
            self._subagent_parents[child_session_id] = parent_session_id
            if parent_handle is not None:
                self._subagent_parent_handles[child_session_id] = parent_handle
        return self.ensure_session(
            {"session_id": child_session_id},
            metadata=metadata,
        )

    def ensure_session(
        self,
        event: dict[str, Any],
        *,
        data: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RelaySession | None:
        """Return the existing session scope or create it once."""
        session_id = _session_id(event)
        if not session_id:
            return None
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                parent_session_id = self._subagent_parents.get(session_id, "")
                session = RelaySession(
                    session_id=session_id,
                    parent_session_id=parent_session_id,
                )
                self._sessions[session_id] = session
        with session.lock:
            if session.closing:
                return None
            if session.handle is None:
                parent_handle = None
                scope_metadata = {
                    **(metadata or {}),
                    RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                    RUNTIME_INSTANCE_KEY: self.runtime_id,
                }
                if session.parent_session_id:
                    with self._sessions_lock:
                        parent_handle = self._subagent_parent_handles.get(session_id)
                    if parent_handle is None:
                        parent = self.ensure_session({
                            "session_id": session.parent_session_id
                        })
                        if parent is not None:
                            parent_handle = parent.handle
                    scope_metadata["nemo_relay_scope_role"] = "subagent"
                context = contextvars.Context()
                try:
                    session.handle = context.run(
                        self.relay.scope.push,
                        SESSION_SCOPE,
                        self.relay.ScopeType.Agent,
                        handle=parent_handle,
                        data=data,
                        input={},
                        metadata=scope_metadata,
                    )
                except Exception:
                    session.context = None
                    raise
                session.context = context
        return session


RelayHost = RelayRuntime | NoopRelayRuntime


class RelayHostRegistry:
    """RelayHostRegistry 是 "profile 维度的单例工厂"——确保同一个 Hermes 配置环境只有一个 Relay 运行时实例，创建失败自动降级为空壳，关闭时避免锁内阻塞。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._hosts: dict[str, RelayHost] = {}

    def for_profile(
        self,
        profile_key: str | None = None,
        *,
        create: bool = True,
    ) -> RelayHost | None:
        key = profile_key or current_profile_key()
        host = self._hosts.get(key)
        if host is not None or not create:
            return host
        with self._lock:
            host = self._hosts.get(key)
            if host is not None or not create:
                return host
            try:
                host = RelayRuntime(profile_key=key)
            except Exception as exc:
                logger.warning(
                    "Hermes Relay runtime initialization failed", exc_info=True
                )
                host = NoopRelayRuntime(profile_key=key, reason=str(exc))
            self._hosts[key] = host
            return host


HOST_REGISTRY = RelayHostRegistry()


@dataclass
class ConversationLease:
    """一次对话的使用权凭证"""

    profile_key: str
    session_id: str
    platform: str
    host: RelayHost
    session: RelaySession | None
    parent_session_id: str = ""
    released: bool = False


@dataclass
class RelayTurnContext:
    """Runtime-only context for one Hermes turn or top-level task."""

    lease: ConversationLease
    turn_id: str
    task_id: str
    handle: Any = None
    logical_llm_calls: dict[str, Any] = field(default_factory=dict, repr=False)
    logical_llm_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    finalize_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    _previous_turn: RelayTurnContext | None = field(default=None, repr=False)
    _active_registered: bool = field(default=False, repr=False)
    relay_enabled: bool = True
    closed: bool = False


_CURRENT_TURN: contextvars.ContextVar[RelayTurnContext | None] = contextvars.ContextVar(
    "hermes_relay_turn", default=None
)


class RelaySessionCoordinator:
    """这是 Hermes 核心的"会话与轮次生命周期管家"。它负责把一次对话（session）和其中每一轮交互（turn）的创建、运行、收尾全部串起来，同时对接底层的 nemo_relay 观测系统（或它的空壳替身 NoopRelayRuntime）。"""

    def __init__(self, registry: RelayHostRegistry = HOST_REGISTRY) -> None:
        self.registry = registry
        self._initializer_lock = threading.RLock()
        self._session_initializers: dict[
            str,
            Callable[[RelayRuntime, dict[str, Any]], None],
        ] = {}
        self._active_turns_lock = threading.RLock()
        self._active_turns: dict[tuple[str, str], set[int]] = {}

    def _prepare_session(
        self,
        host: RelayRuntime,
        context: dict[str, Any],
    ) -> None:
        with self._initializer_lock:
            initializers = list(self._session_initializers.items())
        for name, callback in initializers:
            try:
                callback(host, context)
            except Exception:
                logger.warning(
                    "Hermes Relay session initializer failed: %s",
                    name,
                    exc_info=True,
                )

    def acquire_conversation(
        self,
        *,
        profile_key: str,
        session_id: str,
        platform: str,
        parent_session_id: str = "",
        model: str = "",
    ) -> ConversationLease:

        host = self.registry.for_profile(profile_key)
        if host is None:
            host = NoopRelayRuntime(profile_key, "Relay host creation was disabled")
        session = None

        if isinstance(host, RelayRuntime):
            try:
                session_context = {
                    "profile_key": profile_key,
                    "session_id": session_id,
                    "platform": platform,
                    "parent_session_id": parent_session_id,
                    "model": model,
                }
                self._prepare_session(host, session_context)
                metadata = {"hermes.execution_surface": platform or "unknown"}

                if parent_session_id and parent_session_id != session_id:
                    session = host.register_subagent(
                        {
                            "parent_session_id": parent_session_id,
                            "child_session_id": session_id,
                        },
                        metadate=metadata,
                    )
                else:
                    session = host.ensure_session(
                        {"session_id": session_id},
                        metadata=metadata,
                    )
            except Exception:
                logger.warning(
                    "Hermes Relay conversation initialization failed",
                    exc_info=True,
                )
        return ConversationLease(
            profile_key=profile_key,
            session_id=session_id,
            platform=platform,
            host=host,
            session=session,
            parent_session_id=parent_session_id,
        )

    @staticmethod
    def release_conversation(lease: ConversationLease) -> None:
        """Release a caller lease without closing a resumable conversation."""
        lease.released = True

    def begin_turn(
        self,
        lease: ConversationLease,
        *,
        turn_id: str,
        task_id: str,
    ) -> RelayTurnContext:
        if lease.released:
            raise RuntimeError("Hermes Relay conversation lease is released")
        turn = RelayTurnContext(lease=lease, turn_id=turn_id, task_id=task_id)
        key = (lease.profile_key, lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active:
                # A Relay session owns one physical scope stack. Concurrent
                # Hermes turns would create sibling scopes on that stack, but
                # their completion order is not guaranteed to be LIFO.
                turn.relay_enabled = False
                logger.warning(
                    "Skipping Relay instrumentation for concurrent Hermes turn "
                    "%s in session %s",
                    turn_id,
                    lease.session_id,
                )
            else:
                self._active_turns[key] = {id(turn)}
                turn._active_registered = True
        if (
            turn.relay_enabled
            and isinstance(lease.host, RelayRuntime)
            and lease.session is not None
        ):
            try:
                turn.handle = lease.host.run_in_session(
                    lease.session,
                    lease.host.relay.scope.push,
                    TURN_SCOPE,
                    lease.host.relay.ScopeType.Function,
                    handle=lease.session.handle,
                    input={},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                        "hermes.execution_surface": lease.platform or "unknown",
                    },
                )
            except Exception:
                logger.warning("Hermes Relay turn initialization failed", exc_info=True)
        turn._previous_turn = _CURRENT_TURN.get()
        _CURRENT_TURN.set(turn)
        return turn

    def end_turn(
        self,
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        with turn.finalize_lock:
            if turn.closed:
                self._reset_turn_context(turn)
                return
            turn.closed = True
            lease = turn.lease
            try:
                if isinstance(lease.host, RelayRuntime) and lease.session is not None:
                    self._finish_logical_calls(turn, outcome=outcome)
                    if turn.handle is not None:
                        try:
                            lease.host.run_in_session(
                                lease.session,
                                lease.host.relay.scope.pop,
                                turn.handle,
                                output={"outcome": outcome},
                                metadata={
                                    RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                                    RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                                },
                            )
                        except Exception:
                            logger.warning(
                                "Hermes Relay turn finalization failed", exc_info=True
                            )
            finally:
                try:
                    # Delegated agents own one turn. Close their conversation
                    # while the active-turn guard is still held so a parent
                    # timeout fallback cannot race this terminal boundary.
                    if lease.parent_session_id and isinstance(lease.host, RelayRuntime):
                        lease.host.unregister_subagent({
                            "child_session_id": lease.session_id
                        })
                except Exception:
                    logger.warning(
                        "Hermes Relay child conversation finalization failed",
                        exc_info=True,
                    )
                finally:
                    self._unregister_active_turn(turn)
                    self._reset_turn_context(turn)

    def finish_logical_calls(
        self,
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        """Close logical LLM children before sibling task aggregation scopes."""
        with turn.finalize_lock:
            if turn.closed:
                return
            self._finish_logical_calls(turn, outcome=outcome)

    @staticmethod
    def _finish_logical_calls(
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        lease = turn.lease
        if not isinstance(lease.host, RelayRuntime) or lease.session is None:
            return
        with turn.logical_llm_lock:
            logical_calls = list(turn.logical_llm_calls.items())
            turn.logical_llm_calls.clear()
        for index in range(len(logical_calls) - 1, -1, -1):
            request_id, logical_handle = logical_calls[index]
            try:
                lease.host.run_in_session(
                    lease.session,
                    lease.host.relay.scope.pop,
                    logical_handle,
                    output={"outcome": outcome},
                    metadata={
                        RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                        RUNTIME_INSTANCE_KEY: lease.host.runtime_id,
                    },
                )
            except Exception:
                with turn.logical_llm_lock:
                    # Relay scopes are stack-owned. If the newest remaining
                    # handle cannot close, older handles cannot close safely
                    # either, so retain the unclosed prefix for diagnostics.
                    for pending_request_id, pending_handle in logical_calls[
                        : index + 1
                    ]:
                        turn.logical_llm_calls.setdefault(
                            pending_request_id,
                            pending_handle,
                        )
                logger.warning(
                    "Hermes Relay logical LLM finalization failed",
                    exc_info=True,
                )
                break


SESSION_COORDINATOR = RelaySessionCoordinator()


def current_turn() -> RelayTurnContext | None:
    """Return the turn context inherited by current async and thread work."""
    return _CURRENT_TURN.get()


def _load_nemo_relay() -> Any:
    """懒加载 NVIDIA NeMo Relay 的 Python 绑定。"""
    return importlib.import_module("nemo_relay")


def current_profile_key() -> str:
    """返回用于运行时隔离的规范配置文件标识"""
    home = get_hermes_home().expanduser()
    if not home.is_absolute():
        return str(home.resolve())

    raw = str(home)
    cache = _PROFILE_KEY_CACHE.get(raw)
    if cache is not None:
        return cache
    resolved = str(home.resolve())

    return _PROFILE_KEY_CACHE.setdefault(raw, resolved)


def active_turn(session_id: str | None = None) -> RelayTurnContext | None:
    """Return a live turn only when it belongs to the active profile/session."""
    turn = current_turn()
    if turn is None or not turn.relay_enabled or turn.closed or turn.lease.released:
        return None
    if turn.lease.profile_key != current_profile_key():
        return None
    if session_id is not None and turn.lease.session_id != session_id:
        return None
    if isinstance(turn.lease.host, RelayRuntime):
        if turn.lease.session is None:
            return None
        if turn.lease.host.get_session(turn.lease.session_id) is not turn.lease.session:
            return None
    return turn


def _session_id(event: dict[str, Any]) -> str:
    return str(event.get("session_id") or "")
