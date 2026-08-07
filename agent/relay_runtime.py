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

# ---- Relay 作用域（scope）与元数据常量 ----
# SESSION_SCOPE     : 会话级作用域名，一个 Hermes 会话对应一个 Agent 级 scope
# TURN_SCOPE        : 轮次级作用域名，一次交互对应一个 Function 级 scope
# LOGICAL_LLM_SCOPE : 逻辑 LLM 调用作用域名（本精简版中暂未直接使用）
# RUNTIME_SCHEMA_KEY / RUNTIME_SCHEMA_VERSION : 写入 scope metadata 的 schema 版本，
#                                               供观测端识别数据格式
# RUNTIME_INSTANCE_KEY : 写入 scope metadata 的运行时实例 ID，标识 scope 由哪个实例创建
SESSION_SCOPE = "hermes.session"
TURN_SCOPE = "hermes.turn"
LOGICAL_LLM_SCOPE = "hermes.logical_llm_call"
RUNTIME_SCHEMA_KEY = "hermes.relay.schema_version"
RUNTIME_SCHEMA_VERSION = "hermes.relay.runtime.v1"
RUNTIME_INSTANCE_KEY = "hermes.relay.runtime_instance"
_PROFILE_KEY_CACHE: dict[
    str, str
] = {}  # 缓存 {未解析路径: 解析后路径}，避免反复调用 resolve()


@dataclass
class RelaySession:
    """一个由 Hermes 会话拥有的独立 Relay 作用域堆栈。

    每个 Hermes 会话（由 session_id 唯一标识）对应一个 RelaySession，
    它承载该会话在 nemo_relay 中的整套 scope 栈：
    - lock   : 串行化对句柄/上下文的访问（会话可能被多个线程同时使用）；
    - handle : 会话 scope 栈顶句柄，后续的 turn / 逻辑 LLM 调用都挂在它之下；
    - context: 保存了 scope 栈的 contextvars 快照，用于在异步/线程环境下恢复。
    """

    session_id: str  # 会话唯一标识（与 Hermes 的 session_id 一致）
    parent_session_id: str = ""  # 父会话 ID；空串表示顶层会话（非子代理）
    lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )  # 会话级可重入锁，保护 handle/context 的并发访问
    closing: bool = False  # 关闭标记：True 表示会话正在收尾，拒绝新的 scope 操作
    handle: Any = None  # 会话 scope 栈顶句柄（由 nemo_relay 的 scope.push 返回）
    context: contextvars.Context | None = None  # 保存 scope 栈的上下文快照


@dataclass(frozen=True)
class NoopRelayRuntime:
    """这是一个降级存根(Stub)类，当平台装不了 nemo_relay 轮子(比如没有预编译的 wheel、架构不支持、或者用户没装)时，my-hermes 不会炸，而是用这个"空壳"继续跑。"""

    profile_key: str  # 所属配置环境（仅用于记录/诊断）
    reason: str  # 降级原因（例如导入 nemo_relay 时的异常信息）

    @property
    def available(self) -> bool:
        """是否可用——空壳永远返回 False，表示本平台不提供任何 Relay 能力。"""
        return False

    def apply_tool_request_intercepts(
        self,
        *,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        """空壳版工具请求拦截：不重写任何参数，原样返回（fail-open，宽松失败）。"""
        del session_id, tool_name
        return args

    @staticmethod
    def retain_managed_execution(consumer: str) -> None:
        """空壳版“保持受管执行”：什么都不做（本就没有受管执行管线）。"""
        del consumer

    @staticmethod
    def release_managed_execution(consumer: str) -> None:
        """空壳版“释放受管执行”：什么都不做。"""
        del consumer

    @staticmethod
    def managed_execution_enabled() -> bool:
        """空壳版“受管执行是否启用”：永远返回 False。"""
        return False

    def shutdown(self) -> None:
        """空壳版关闭：不支持的平台上没有分配任何资源，因此无事可做。"""


class RelayRuntime:
    """独立于任何导出器或插件，拥有自己的中继会话作用域"""

    def __init__(self, relay: Any = None, *, profile_key: str | None = None) -> None:
        """初始化 Relay 运行时。

        参数:
            relay: nemo_relay 模块（默认懒加载）；显式传入主要用于测试注入。
            profile_key: 所属配置环境标识；缺省时由 current_profile_key() 推导。

        每个实例都会：
        - 生成唯一 runtime_id，并随 scope 的 metadata 一起写入，
          便于在观测端区分作用域属于哪个运行时实例；
        - 在进程退出时通过 atexit 自动调用 shutdown() 收尾所有会话。
        """
        self.relay = relay or _load_nemo_relay()
        self.profile_key = profile_key or current_profile_key()
        self.runtime_id = uuid.uuid4().hex
        self._sessions_lock = threading.RLock()  # 保护 _sessions 与子代理登记表的锁
        self._sessions: dict[str, RelaySession] = {}  # {session_id: RelaySession}
        self._subagent_parents: dict[str, str] = {}  # {子会话ID: 父会话ID}
        self._subagent_parent_handles: dict[str, Any] = {}  # {子会话ID: 父会话句柄}
        self._execution_consumers_lock = threading.RLock()  # 保护受管执行消费方集合的锁
        self._execution_consumers: set[str] = (
            set()
        )  # 需要受管执行的消费方（本精简版暂未使用）
        self._shutdown_registered = True  # 是否已在 atexit 中登记 shutdown
        atexit.register(self.shutdown)

    def register_subagent(
        self,
        event: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RelaySession | None:
        """在父会话的轮次之下，为子代理开启一个子会话作用域。

        参数 event 至少需要包含 parent_session_id 与 child_session_id；
        若父会话当前有一个活跃且属于本运行时的轮次，则子会话的 scope
        会挂在父轮次句柄之下，否则挂在父会话句柄之下。
        """
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
        """返回已存在的会话作用域；不存在则只创建一次。

        幂等 + 并发安全：多线程同时调用也只会产生一个 RelaySession。
        首次创建时会向 nemo_relay 推入一个 Agent 级 scope 作为会话根，
        并把创建它的运行时实例 ID、schema 版本写入 scope metadata；
        若会话有父会话，则额外标注 nemo_relay_scope_role=subagent。
        """
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
                    # 在全新的 contextvars 上下文中推入会话 scope，
                    # 把 scope 栈与调用方上下文隔离，互不污染。
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

    def get_session(self, session_id: str) -> RelaySession | None:
        """返回活跃的 Hermes Relay 会话；不创建新会话。

        已关闭（closing）的会话视为不存在。
        """
        with self._sessions_lock:
            session = self._sessions.get(str(session_id or ""))
        if session is None:
            return None
        with session.lock:
            return None if session.closing else session

    def run_in_session(
        self,
        session: RelaySession,
        callback: Callable[..., Any],
        *args: Any,
        allow_closing: bool = False,
        **kwargs: Any,
    ) -> Any:
        """在会话的隔离 scope 栈上执行一个 Relay 操作（如 scope.push/pop）。

        原理：取出会话保存的 contextvars 快照副本，把其中的上下文变量
        恢复进当前上下文再执行回调——既复用了会话的 scope 栈，
        又不会让调用方的上下文变量泄漏进会话。
        """
        with session.lock:
            if session.closing and not allow_closing:
                raise RuntimeError("Hermes Relay session is closing")
            if session.context is None or session.handle is None:
                raise RuntimeError("Hermes Relay session context is unavailable")
            relay_context = session.context.copy()

        context = contextvars.copy_context()
        for variable, value in relay_context.items():
            context.run(variable.set, value)

        def invoke() -> Any:
            self.relay.get_scope_stack()
            return callback(*args, **kwargs)

        # 使用副本：允许被既有 Relay 回调调用的辅助函数重入同一逻辑会话，
        # 而无需重入 Context。
        return context.run(invoke)

    def close_session(self, event: dict[str, Any]) -> None:
        """关闭一个会话作用域，并从核心注册表中移除。

        先 pop 掉会话 scope（允许在 closing 状态下收尾），再冲刷订阅者，
        最后清理 _sessions 与子代理登记表；收尾过程中的错误只记警告，
        不中断流程。
        """
        session_id = _session_id(event)
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            with self._sessions_lock:
                self._subagent_parents.pop(session_id, None)
                self._subagent_parent_handles.pop(session_id, None)
            return
        failures: list[str] = []
        with session.lock:
            if session.closing:
                return
            session.closing = True
            if session.handle is not None:
                try:
                    self.run_in_session(
                        session,
                        self.relay.scope.pop,
                        session.handle,
                        output={},
                        metadata={
                            RUNTIME_SCHEMA_KEY: RUNTIME_SCHEMA_VERSION,
                            RUNTIME_INSTANCE_KEY: self.runtime_id,
                        },
                        allow_closing=True,
                    )
                except Exception as exc:
                    failures.append(f"session scope close failed: {exc}")
        try:
            self.relay.subscribers.flush()
        except Exception as exc:
            failures.append(f"subscriber flush failed: {exc}")
        with self._sessions_lock:
            if self._sessions.get(session_id) is session:
                self._sessions.pop(session_id, None)
            self._subagent_parents.pop(session_id, None)
            self._subagent_parent_handles.pop(session_id, None)
        if failures:
            logger.warning(
                "Hermes Relay session %s closed with errors: %s",
                session_id,
                "; ".join(failures),
            )

    def unregister_subagent(self, event: dict[str, Any]) -> None:
        """关闭一个被委派的子会话，并遗忘其父子关系登记。"""
        child_session_id = str(event.get("child_session_id") or "")
        if not child_session_id:
            return
        self.close_session({"session_id": child_session_id})
        with self._sessions_lock:
            self._subagent_parents.pop(child_session_id, None)
            self._subagent_parent_handles.pop(child_session_id, None)

    def shutdown(self) -> None:
        """关闭本运行时拥有的所有 Relay 会话作用域（进程退出时由 atexit 调用）。"""
        with self._sessions_lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self._safe(self.close_session, {"session_id": session_id})
        if self._shutdown_registered:
            try:
                atexit.unregister(self.shutdown)
            except Exception:
                pass
            self._shutdown_registered = False

    @staticmethod
    def _safe(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """执行回调并吞掉异常（fail-open），失败时返回 None。"""
        try:
            return callback(*args, **kwargs)
        except Exception:
            logger.warning("Hermes Relay runtime operation failed", exc_info=True)
            return None


RelayHost = RelayRuntime | NoopRelayRuntime  # 真实运行时与空壳运行时的联合类型


class RelayHostRegistry:
    """RelayHostRegistry 是 "profile 维度的单例工厂"——确保同一个 Hermes 配置环境只有一个 Relay 运行时实例，创建失败自动降级为空壳，关闭时避免锁内阻塞。"""

    def __init__(self) -> None:
        """构造一个空注册表：一把可重入锁 + 一个 {profile_key: host} 字典。"""
        self._lock = threading.RLock()
        self._hosts: dict[str, RelayHost] = {}

    def for_profile(
        self,
        profile_key: str | None = None,
        *,
        create: bool = True,
    ) -> RelayHost | None:
        """按 profile 获取（或创建）唯一的 Relay 主机实例。

        采用“双检锁”（double-checked locking）保证同一 profile 只实例化一次；
        创建 RelayRuntime 失败时自动降级为 NoopRelayRuntime 空壳并记录警告，
        确保上层流程不受影响。create=False 时只查询、不创建。
        """
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


HOST_REGISTRY = RelayHostRegistry()  # 全局唯一的 Relay 主机注册表单例


@dataclass
class ConversationLease:
    """一次对话的使用权凭证（lease）。

    由 acquire_conversation() 签发，记录该对话归属于哪个 profile/会话/平台，
    以及底层对应的 Relay 主机和会话对象；released 标记该租约是否已被释放。
    """

    profile_key: str  # 归属的 Hermes 配置环境
    session_id: str  # 会话 ID
    platform: str  # 执行平台（telegram / cli / discord 等）
    host: RelayHost  # 实际（或空壳）Relay 主机
    session: RelaySession | None  # 底层 Relay 会话；创建失败时为 None
    parent_session_id: str = ""  # 父会话 ID（子代理场景）
    released: bool = False  # 租约是否已释放（释放后不能再 begin_turn）


@dataclass
class RelayTurnContext:
    """一个 Hermes 轮次（turn）或顶层任务专用的运行时上下文。

    由 begin_turn() 创建并写入 contextvars（_CURRENT_TURN），
    让当前轮次中的异步任务/线程都能通过 current_turn() 拿到它；
    其中记录了轮次句柄、逻辑 LLM 调用簿，以及收尾所需的锁。
    """

    lease: ConversationLease  # 本轮次对应的对话租约
    turn_id: str  # 轮次唯一 ID
    task_id: str  # 所属任务 ID
    handle: Any = None  # 轮次在 Relay 中的 scope 句柄（TURN_SCOPE）
    logical_llm_calls: dict[str, Any] = field(
        default_factory=dict, repr=False
    )  # {request_id: 逻辑 LLM 调用句柄}，轮次结束时倒序回收
    logical_llm_lock: threading.RLock = field(  # 保护 logical_llm_calls 的锁
        default_factory=threading.RLock,
        repr=False,
    )
    finalize_lock: threading.RLock = field(  # 收尾锁，保证 end_turn 幂等
        default_factory=threading.RLock,
        repr=False,
    )
    _previous_turn: RelayTurnContext | None = field(
        default=None, repr=False
    )  # 被本轮次覆盖的上一个轮次上下文（用于结束后的复位）
    _active_registered: bool = field(
        default=False, repr=False
    )  # 是否已登记到活跃轮次表
    relay_enabled: bool = (
        True  # 是否启用 Relay 插桩（同一会话的并发轮次会被置为 False）
    )
    closed: bool = False  # 轮次是否已关闭


# 全局“当前轮次”上下文变量：在不同异步任务/线程之间隔离，互不串扰
_CURRENT_TURN: contextvars.ContextVar[RelayTurnContext | None] = contextvars.ContextVar(
    "hermes_relay_turn", default=None
)


class RelaySessionCoordinator:
    """这是 Hermes 核心的"会话与轮次生命周期管家"。它负责把一次对话（session）和其中每一轮交互（turn）的创建、运行、收尾全部串起来，同时对接底层的 nemo_relay 观测系统（或它的空壳替身 NoopRelayRuntime）。"""

    def __init__(self, registry: RelayHostRegistry = HOST_REGISTRY) -> None:
        """构造协调器。

        参数:
            registry: 使用的 Relay 主机注册表（默认全局单例 HOST_REGISTRY，
                      测试时可替换为隔离的注册表）。

        内部维护：
        - _session_initializers: 按名字注册的“会话初始化器”回调表，
          在创建会话 scope 之前统一执行准备工作；
        - _active_turns: {(profile_key, session_id): {turn对象id集合}}，
          用于检测同一会话内的并发轮次。
        """
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
        """执行所有已注册的“会话初始化器”（session initializer）。

        在真正创建 Relay 会话作用域之前，先让各个组件（如导出器、插件）
        有机会做准备工作；单个初始化器失败只记警告，不影响整体流程。
        """
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
        """为一个会话“租用”一次对话使用权，返回租约凭证。

        流程：按 profile_key 找到（或创建）Relay 主机 → 先跑一遍会话初始化器 →
        按是否有父会话决定注册为子代理（register_subagent）还是普通会话
        （ensure_session）→ 返回 ConversationLease。
        任何异常都会被吞掉并记警告（fail-open），保证上层对话流程不中断；
        即使主机是空壳，也照常返回租约，只是 session 为 None。
        """
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
                        metadata=metadata,
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
        """释放调用方持有的租约，但保留可恢复的会话本身（不关闭）。"""
        lease.released = True

    def begin_turn(
        self,
        lease: ConversationLease,
        *,
        turn_id: str,
        task_id: str,
    ) -> RelayTurnContext:
        """开启一轮新的 Relay 轮次（turn），返回轮次上下文。

        并发保护：同一会话内并发开启多轮时，由于 Relay 的 scope 栈是物理单栈
        且不保证后进先出回收，后开的那一轮会被降级为“不插桩”
        （relay_enabled=False），仅记一条警告。
        轮次开启成功后，会把自身写入 contextvars（_CURRENT_TURN），
        使异步/线程任务能通过 current_turn() 继承当前轮次上下文。
        """
        if lease.released:
            raise RuntimeError("Hermes Relay conversation lease is released")
        turn = RelayTurnContext(lease=lease, turn_id=turn_id, task_id=task_id)
        key = (lease.profile_key, lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active:
                # 一个 Relay 会话只拥有一套物理 scope 栈。并发开启多个 Hermes
                # 轮次会在同一套栈上产生兄弟 scope，但它们的回收顺序不保证是
                # 后进先出（LIFO），因此对并发轮次跳过 Relay 插桩。
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
                # 在会话上下文内推入一个 Function 级轮次 scope，
                # 挂在会话句柄之下，元数据带上 schema 版本与运行时实例 ID。
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
        """结束一轮 Relay 轮次，回收 scope 并复位上下文。

        幂等设计：已关闭的轮次直接返回。收尾顺序：
        1. 关闭逻辑 LLM 子作用域（倒序 pop）；
        2. pop 掉轮次自己的 scope；
        3. 若是子代理（有父会话），注销其子会话；
        4. 从活跃轮次登记表移除，并把 contextvars 中的当前轮次恢复为上一个。
        """
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
                    # 子代理只拥有一个轮次：在活跃轮次守卫仍然生效时关闭其
                    # 会话，避免父级超时回退与本终态边界发生竞态。
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

    def _unregister_active_turn(self, turn: RelayTurnContext) -> None:
        """把轮次从活跃轮次登记表（_active_turns）中移除。

        仅当该轮次确实登记过才处理；集合清空后连键一起删除，
        以便后续轮次能重新启用 Relay 插桩。
        """
        if not turn._active_registered:
            return
        key = (turn.lease.profile_key, turn.lease.session_id)
        with self._active_turns_lock:
            active = self._active_turns.get(key)
            if active is not None:
                active.discard(id(turn))
                if not active:
                    self._active_turns.pop(key, None)
            turn._active_registered = False

    @staticmethod
    def _reset_turn_context(turn: RelayTurnContext) -> None:
        """把 contextvars 中的当前轮次复位为被覆盖前的轮次。

        只复位“当前正是本 turn”的情形，避免干扰更晚开启的轮次；
        若上一轮次已关闭，则继续向上回溯，直到找到未关闭的轮次为止。
        """
        if _CURRENT_TURN.get() is not turn:
            return
        previous = turn._previous_turn
        seen = {id(turn)}
        while previous is not None and previous.closed:
            if id(previous) in seen:
                previous = None
                break
            seen.add(id(previous))
            previous = previous._previous_turn
        _CURRENT_TURN.set(previous)

    def finish_logical_calls(
        self,
        turn: RelayTurnContext,
        *,
        outcome: str,
    ) -> None:
        """在兄弟任务聚合作用域关闭之前，先关闭逻辑 LLM 子作用域。"""
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
        """按倒序逐个 pop 逻辑 LLM 调用作用域。

        Relay 的 scope 是栈式管理：必须“后来先出”。如果最外层（最新）的
        handle 关闭失败，那么更老的一批也不能安全关闭，因此把未关闭的
        前缀重新放回 logical_llm_calls 保留，供诊断使用。
        """
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
                    # Relay 的 scope 由栈管理：如果最新（最外层）的句柄关不掉，
                    # 更老的一批也不能安全关闭，所以保留未关闭的前缀供诊断。
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


SESSION_COORDINATOR = RelaySessionCoordinator()  # 全局唯一的会话协调器单例


def current_turn() -> RelayTurnContext | None:
    """返回当前异步任务/线程所继承的轮次上下文（不在轮次内时为 None）。"""
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
    """仅当轮次属于当前活跃的 profile/会话时才返回该轮次，否则返回 None。

    校验项：轮次未关闭、租约未释放、profile 一致、会话 ID 匹配，
    且底层会话仍在该主机上登记为活跃（防悬挂引用）。
    """
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
    """从事件字典中安全地取出 session_id；缺失时返回空字符串。"""
    return str(event.get("session_id") or "")
