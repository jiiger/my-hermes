"""MemoryManager —— 记忆提供方编排器（精简移植版）。

对应原版 hermes-agent 的 agent/memory_manager.py（1241 行）。agent 记忆
的单一集成点：把散落的按后端代码收敛为一个管理器，统一委托给已注册的
提供方。

同一时刻只允许一个外部插件提供方——注册第二个外部提供方会被拒绝并警告。
这防止工具 schema 膨胀与互相冲突的记忆后端。

run_agent 用法::

    self._memory_manager = MemoryManager()
    self._memory_manager.add_provider(plugin_provider)   # 只允许一个外部

    # 系统提示
    prompt_parts.append(self._memory_manager.build_system_prompt())

    # 回合前
    context = self._memory_manager.prefetch_all(user_message)

    # 回合后
    self._memory_manager.sync_all(user_msg, assistant_response)
    self._memory_manager.queue_prefetch_all(user_msg)

精简版改动（相对原版）：
- 砍掉 _strip_skill_scaffolding（依赖 agent.skill_commands；my-hermes 无
  /skill 命令），prefetch/sync 直接透传用户文本；
- toolsets._HERMES_CORE_TOOLS → 模块内 _CORE_TOOL_NAMES 常量（与
  my-hermes 注册的工具集一致）；
- tools.daemon_pool.DaemonThreadPoolExecutor → 本模块内置实现（照搬原版
  tools/daemon_pool.py，64 行）；
- 砍掉 StreamingContextScrubber（流式 delta 清洗器，my-hermes 流式路径
  未接 scrubber），保留 sanitize_context / build_memory_context_block；
- 其余（注册/路由/线程模型/生命周期广播/notify 镜像/shutdown 排空）
  照原版移植。
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import threading
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures.thread import _worker
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# shutdown_all() 等待在途后台 sync/prefetch 排空的时间上限。卡死的提供方
# 绝不能无限阻塞进程退出——工作线程是 daemon，超过窗口仍在跑的随解释器
# 一起结束。
_SYNC_DRAIN_TIMEOUT_S = 5.0
_EXTERNAL_PREFETCH_TIMEOUT_S = 8.0

# 核心工具名（my-hermes 注册的工具集）——memory provider 的工具绝不允许
# 遮蔽这些名字，内置工具永远优先（对齐原版 toolsets._HERMES_CORE_TOOLS）。
_CORE_TOOL_NAMES = frozenset({
    "todo", "read_file", "write_file", "patch", "search_files", "terminal",
    "memory",
})


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """守护线程 ThreadPoolExecutor（照搬原版 tools/daemon_pool.py）。

    标准库 ThreadPoolExecutor 的工作线程非 daemon 且注册进
    ``concurrent.futures.thread._threads_queues``，其 atexit 钩子
    （``_python_exit``）会无条件 join 每个工作线程——即使
    ``shutdown(wait=False)`` 之后也一样。单个卡死的工作线程（网络 I/O
    阻塞、挂起的 provider daemon）因此会永远阻塞解释器退出。

    本类派生 daemon 工作线程并跳过 _threads_queues 注册，所以退出钩子
    永不 join 它们。语义其余相同（initializer、工作队列、空闲线程复用）。
    """

    def _adjust_thread_count(self) -> None:
        # 镜像 CPython 实现（3.8-3.13），两处改动：daemon=True 且不注册
        # _threads_queues。
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _threads_wakeups = getattr(self, "_threads_wakeups", None)
            if _threads_wakeups is not None:
                _threads_wakeups[t] = threading.Event()


def normalize_tool_schema(schema: Any) -> Optional[Dict[str, Any]]:
    """返回带可解析顶层 ``name`` 的函数工具 dict（照抄原版）。

    上下文引擎与记忆提供方通过 ``get_tool_schemas()`` 暴露工具 schema。
    期望形态是裸函数 schema（``{"name": ..., "description": ...,
    "parameters": ...}``），调用方包装为 ``{"type": "function",
    "function": schema}``。

    有些提供方返回的条目**已经是** OpenAI 工具形态（``{"type":
    "function", "function": {"name": ...}}``）。再包一层会产生
    ``function`` 无顶层 ``name`` 的双层包装。严格提供方（如 DeepSeek）
    会用 ``tools[N].function: missing field name``（HTTP 400）拒绝**整个**
    请求——一个坏 schema 会禁用整个工具集并毁掉每回合（#47707）。

    本助手把两种形态都归一化为裸函数 schema；无法解析出名字的返回
    None，调用方跳过并警告，而不是追加一个无名工具。
    """
    if not isinstance(schema, dict):
        return None
    # 解开已是 OpenAI 工具形态的条目
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        schema = schema["function"]
        if not isinstance(schema, dict):
            return None
    name = schema.get("name", "")
    if not name or not isinstance(name, str):
        return None
    return schema


def memory_provider_tools_enabled(
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]] = None,
    *,
    memory_tool_present: bool = False,
) -> bool:
    """外部 memory-provider 工具是否应暴露（对应原版 memory_manager.py:83）。

    规则（对齐原版）：
    - disabled 显式含 "memory" → 不暴露；
    - registry 里已有 memory 工具（memory_tool_present）→ 暴露；
    - enabled 未限制（None）→ 暴露；
    - enabled 为空列表 → 不暴露；
    - enabled 含 "memory" → 暴露；
    - 否则看 enabled 各工具集解析结果里是否含 memory 工具。
    """
    if disabled_toolsets and "memory" in disabled_toolsets:
        return False
    if memory_tool_present:
        return True
    if enabled_toolsets is None:
        return True
    if not enabled_toolsets:
        return False
    if "memory" in enabled_toolsets:
        return True

    try:
        from toolsets import resolve_toolset

        return any("memory" in resolve_toolset(name) for name in enabled_toolsets)
    except Exception:
        logger.debug(
            "Failed to resolve enabled toolsets for memory-provider tools",
            exc_info=True,
        )
        return False


# ─── 上下文围栏辅助 ─────────────────────────────────────────────────────

_FENCE_TAG_RE = re.compile(r"</?\s*memory-context\s*>", re.IGNORECASE)
_INTERNAL_CONTEXT_RE = re.compile(
    r"<\s*memory-context\s*>[\s\S]*?</\s*memory-context\s*>",
    re.IGNORECASE,
)
_INTERNAL_NOTE_RE = re.compile(
    r"\[System note:\s*The following is recalled memory context,\s*NOT new user input\.\s*Treat as (?:informational background data|authoritative reference data[^\]]*)\.\]\s*",
    re.IGNORECASE,
)


def sanitize_context(text: str) -> str:
    """从 provider 输出中剥离围栏标签、注入的上下文块与系统注记。"""
    text = _INTERNAL_CONTEXT_RE.sub("", text)
    text = _INTERNAL_NOTE_RE.sub("", text)
    text = _FENCE_TAG_RE.sub("", text)
    return text


def build_memory_context_block(raw_context: str) -> str:
    """把 prefetch 到的记忆包成带系统注记的围栏块。"""
    if not raw_context or not raw_context.strip():
        return ""
    clean = sanitize_context(raw_context)
    if clean != raw_context:
        logger.warning("memory provider returned pre-wrapped context; stripped")
    return (
        "<memory-context>\n"
        "[System note: The following is recalled memory context, "
        "NOT new user input. Treat as authoritative reference data — "
        "this is the agent's persistent memory and should inform all responses.]\n\n"
        f"{clean}\n"
        "</memory-context>"
    )


class MemoryManager:
    """编排内置提供方加至多一个外部提供方。

    内置提供方永远第一。只允许一个非内置（外部）提供方。一个提供方的
    失败绝不阻塞另一个。
    """

    def __init__(self, *, external_prefetch_timeout: Optional[float] = None) -> None:
        self._providers: List[MemoryProvider] = []
        self._tool_to_provider: Dict[str, MemoryProvider] = {}
        self._has_external: bool = False  # 加入非内置提供方后为 True
        self._external_prefetch_timeout = (
            _EXTERNAL_PREFETCH_TIMEOUT_S
            if external_prefetch_timeout is None
            else float(external_prefetch_timeout)
        )
        if self._external_prefetch_timeout <= 0:
            raise ValueError("external_prefetch_timeout must be positive")
        self._external_prefetch_threads: Dict[str, threading.Thread] = {}
        self._external_prefetch_lock = threading.Lock()
        # 回合末 sync/prefetch 的后台执行器。首次使用时惰性创建，所以常见
        # 的仅内置路径不派生额外线程。单 worker 串行化一个提供方的写入
        # （第 N 轮必须先于第 N+1 轮落库），并把线程增长限制为每管理器一个。
        self._sync_executor: Optional[DaemonThreadPoolExecutor] = None
        self._sync_executor_lock = threading.Lock()
        # 按持久性类别跟踪 future，让 shutdown 能做有界 FIFO 排空，然后
        # 显式报告任何被放弃的任务。
        self._background_futures: Dict[Future, str] = {}
        self._shutting_down = False
        self._shutdown_drain_state: Dict[str, Any] = {
            "status": "not_started",
            "abandoned_writes": 0,
            "abandoned_prefetches": 0,
            "active_tasks": 0,
        }

    # -- 注册 --------------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> None:
        """注册记忆提供方。

        内置提供方（name == "builtin"）永远接受。只允许**一个**外部
        （非内置）提供方——再次尝试会被拒绝并警告。
        """
        is_builtin = provider.name == "builtin"

        if not is_builtin:
            if self._has_external:
                existing = next(
                    (p.name for p in self._providers if p.name != "builtin"), "unknown"
                )
                logger.warning(
                    "Rejected memory provider '%s' — external provider '%s' is "
                    "already registered. Only one external memory provider is "
                    "allowed at a time. Configure which one via memory.provider "
                    "in config.yaml.",
                    provider.name, existing,
                )
                return
            self._has_external = True

        self._providers.append(provider)

        # 核心工具名保留——记忆提供方绝不能注册遮蔽内置工具（如 clarify、
        # delegate_task）的 schema。内置永远优先（对齐原版 #40466）。
        _core_tool_names = set(_CORE_TOOL_NAMES)

        # 建工具名 → 提供方路由表
        for raw_schema in provider.get_tool_schemas():
            schema = normalize_tool_schema(raw_schema)
            if schema is None:
                continue
            tool_name = schema["name"]
            if tool_name in _core_tool_names:
                logger.warning(
                    "Memory provider '%s' tool '%s' shadows a reserved core "
                    "tool name; registration ignored. Core tools always win — "
                    "rename the provider's tool to something unique.",
                    provider.name, tool_name,
                )
                continue
            if tool_name and tool_name not in self._tool_to_provider:
                self._tool_to_provider[tool_name] = provider
            elif tool_name in self._tool_to_provider:
                logger.warning(
                    "Memory tool name conflict: '%s' already registered by %s, "
                    "ignoring from %s",
                    tool_name,
                    self._tool_to_provider[tool_name].name,
                    provider.name,
                )

        logger.info(
            "Memory provider '%s' registered (%d tools)",
            provider.name,
            len(provider.get_tool_schemas()),
        )

    @property
    def providers(self) -> List[MemoryProvider]:
        """所有已注册提供方（按顺序）。"""
        return list(self._providers)

    def get_provider(self, name: str) -> Optional[MemoryProvider]:
        """按名取提供方，未注册返回 None。"""
        for p in self._providers:
            if p.name == name:
                return p
        return None

    # -- 系统提示 ----------------------------------------------------------

    def build_system_prompt(self) -> str:
        """收集所有提供方的系统提示块。

        返回拼接文本；无提供方贡献返回空串。每个非空块带提供方名标签。
        """
        blocks = []
        for provider in self._providers:
            try:
                block = provider.system_prompt_block()
                if block and block.strip():
                    blocks.append(block)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' system_prompt_block() failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(blocks)

    # -- Prefetch / 召回 ---------------------------------------------------

    def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """从所有提供方收集 prefetch 上下文。

        返回按提供方标签的合并上下文文本。空提供方跳过。一个提供方的
        失败不阻塞其他。
        """
        clean_query = query
        if not clean_query:
            return ""
        parts = []
        for provider in self._providers:
            try:
                result = self._prefetch_provider(provider, clean_query, session_id=session_id)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' prefetch failed (non-fatal): %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    def _prefetch_provider(
        self, provider: MemoryProvider, query: str, *, session_id: str = ""
    ) -> str:
        if provider.name == "builtin":
            return provider.prefetch(query, session_id=session_id)

        result_box: Dict[str, str] = {}
        error_box: Dict[str, Exception] = {}

        def _run() -> None:
            try:
                result_box["value"] = provider.prefetch(query, session_id=session_id) or ""
            except Exception as exc:  # pragma: no cover - re-raised by caller
                error_box["value"] = exc

        thread = threading.Thread(
            target=_run,
            daemon=True,
            name=f"memory-prefetch-{provider.name}",
        )
        with self._external_prefetch_lock:
            existing = self._external_prefetch_threads.get(provider.name)
            if existing is not None:
                if existing.is_alive():
                    logger.debug(
                        "Memory provider '%s' prefetch is still running; skipping this turn",
                        provider.name,
                    )
                    return ""
                self._external_prefetch_threads.pop(provider.name, None)
            self._external_prefetch_threads[provider.name] = thread
            thread.start()

        thread.join(self._external_prefetch_timeout)
        if thread.is_alive():
            logger.warning(
                "Memory provider '%s' prefetch timed out after %.1fs; skipping it until "
                "the stuck call returns",
                provider.name,
                self._external_prefetch_timeout,
            )
            return ""

        with self._external_prefetch_lock:
            if self._external_prefetch_threads.get(provider.name) is thread:
                self._external_prefetch_threads.pop(provider.name, None)
        if error_box:
            raise error_box["value"]
        return result_box.get("value", "")

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """在所有提供方排队后台 prefetch，供下一回合使用。

        提供方工作派发到后台 worker，慢/卡死的提供方绝不会阻塞调用方
        （同 sync_all 的完整理由：agent 卡在回合后几分钟"运行中"）。
        """
        providers = list(self._providers)
        if not providers:
            return

        clean_query = query
        if not clean_query:
            return

        def _run() -> None:
            for provider in providers:
                try:
                    provider.queue_prefetch(clean_query, session_id=session_id)
                except Exception as e:
                    logger.debug(
                        "Memory provider '%s' queue_prefetch failed (non-fatal): %s",
                        provider.name, e,
                    )

        self._submit_background(_run, kind="prefetch")

    # -- Sync --------------------------------------------------------------

    @staticmethod
    def _provider_sync_accepts_messages(provider: MemoryProvider) -> bool:
        """提供方的 sync_turn 是否接受 messages 关键字。"""
        try:
            signature = inspect.signature(provider.sync_turn)
        except (TypeError, ValueError):
            return True
        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return True
        return "messages" in signature.parameters

    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """把完成的回合同步到所有提供方。

        在后台 worker 线程运行，**不是**回合完成路径内联执行。提供方的
        ``sync_turn`` 可能做阻塞网络/daemon 调用（原版观察到配置错误的
        Hindsight daemon 阻塞约 298 秒才失败）；内联执行会让
        run_conversation 在用户看到回复后仍挂着，界面长期显示"运行中"。
        派发到线程后，慢/坏提供方绝不会卡住回合——sync 在后台完成（或
        失败并记录）。

        写入经单 worker 串行化，保证第 N 轮先于第 N+1 轮落库。
        """
        providers = list(self._providers)
        if not providers:
            return

        clean_user_content = user_content
        if not clean_user_content:
            return
        user_content = clean_user_content

        def _run() -> None:
            for provider in providers:
                try:
                    if messages is not None and self._provider_sync_accepts_messages(provider):
                        provider.sync_turn(
                            user_content,
                            assistant_content,
                            session_id=session_id,
                            messages=messages,
                        )
                    else:
                        provider.sync_turn(
                            user_content,
                            assistant_content,
                            session_id=session_id,
                        )
                except Exception as e:
                    logger.warning(
                        "Memory provider '%s' sync_turn failed: %s",
                        provider.name, e,
                    )

        self._submit_background(_run)

    # -- 后台派发 ----------------------------------------------------------

    def _submit_background(self, fn, *, kind: str = "write") -> None:
        """把 ``fn`` 排入串行 worker 并跟踪其持久性类别。"""
        executor = self._get_sync_executor()
        if executor is None:
            if self._shutting_down:
                logger.warning("Memory manager is shutting down; rejecting late %s task", kind)
                return
            # 关闭状态外的创建失败：保留历史 fail-safe 行为，内联执行。
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)
            return
        try:
            # 让提交 + 跟踪与关闭快照原子。回调在释放锁后附加，因为已完成的
            # future 会同步触发回调。
            with self._sync_executor_lock:
                if self._shutting_down:
                    logger.warning("Memory manager is shutting down; rejecting late %s task", kind)
                    return
                future = executor.submit(fn)
                self._background_futures[future] = kind
            future.add_done_callback(self._forget_background_future)
        except RuntimeError:
            if self._shutting_down:
                logger.warning("Memory manager shut down during %s submission; task rejected", kind)
                return
            try:
                fn()
            except Exception as e:  # pragma: no cover - fn guards internally
                logger.debug("Inline memory background task failed: %s", e)

    def _forget_background_future(self, future: Future) -> None:
        with self._sync_executor_lock:
            self._background_futures.pop(future, None)

    def _get_sync_executor(self) -> Optional[DaemonThreadPoolExecutor]:
        """惰性创建单 worker 后台执行器。"""
        if self._shutting_down:
            return None
        if self._sync_executor is not None:
            return self._sync_executor
        with self._sync_executor_lock:
            if self._shutting_down:
                return None
            if self._sync_executor is None:
                try:
                    # 守护 worker：卡在网络调用上的提供方绝不阻塞解释器退出
                    self._sync_executor = DaemonThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="mem-sync",
                    )
                except Exception as e:  # pragma: no cover - resource exhaustion
                    logger.warning("Failed to create memory sync executor: %s", e)
                    return None
            return self._sync_executor

    def flush_pending(self, timeout: Optional[float] = None) -> bool:
        """阻塞直到排队的 sync/prefetch 工作排空。

        单 worker 执行器意味着提交一个哨兵并等待它，就保证每个先前提交
        的任务都跑过。timeout 内完成（或无执行器）返回 True，超时返回
        False。用于真实会话边界和需要确定性断言提供方状态的测试。
        """
        executor = self._sync_executor
        if executor is None:
            return True
        try:
            fut = executor.submit(lambda: None)
        except RuntimeError:
            # 执行器已关闭——没有待处理任务
            return True
        try:
            fut.result(timeout=timeout)
            return True
        except Exception:
            return False

    # -- 工具 --------------------------------------------------------------

    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """从所有提供方收集工具 schema。

        保留的核心工具名（todo/memory 等）被跳过——它们已在 add_provider
        被拒绝进路由表，管理器绝不能宣传一个永不路由的 schema
        （内置永远优先，原版 #40466）。
        """
        _core_tool_names = set(_CORE_TOOL_NAMES)
        schemas = []
        seen = set()
        for provider in self._providers:
            try:
                for raw_schema in provider.get_tool_schemas():
                    schema = normalize_tool_schema(raw_schema)
                    if schema is None:
                        logger.warning(
                            "Memory provider '%s' returned a tool schema with "
                            "no resolvable name; skipping (%r)",
                            provider.name, raw_schema,
                        )
                        continue
                    name = schema["name"]
                    if name in _core_tool_names:
                        continue
                    if name not in seen:
                        schemas.append(schema)
                        seen.add(name)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' get_tool_schemas() failed: %s",
                    provider.name, e,
                )
        return schemas

    def get_all_tool_names(self) -> set:
        """返回所有提供方处理的工具名集合。"""
        return set(self._tool_to_provider.keys())

    def has_tool(self, tool_name: str) -> bool:
        """是否有提供方处理此工具。"""
        return tool_name in self._tool_to_provider

    def handle_tool_call(
        self, tool_name: str, args: Dict[str, Any], **kwargs
    ) -> str:
        """把工具调用路由到正确提供方。

        返回 JSON 字符串结果。无提供方处理该工具时抛 ValueError。
        """
        provider = self._tool_to_provider.get(tool_name)
        if provider is None:
            return tool_error(f"No memory provider handles tool '{tool_name}'")
        try:
            return provider.handle_tool_call(tool_name, args, **kwargs)
        except Exception as e:
            logger.error(
                "Memory provider '%s' handle_tool_call(%s) failed: %s",
                provider.name, tool_name, e,
            )
            return tool_error(f"Memory tool '{tool_name}' failed: {e}")

    # -- 生命周期钩子 ------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """通知所有提供方新回合开始。

        kwargs 可能含：remaining_tokens、model、platform、tool_count。
        """
        for provider in self._providers:
            try:
                provider.on_turn_start(turn_number, message, **kwargs)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_turn_start failed: %s",
                    provider.name, e,
                )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """通知所有提供方会话结束。"""
        for provider in self._providers:
            try:
                provider.on_session_end(messages)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' on_session_end failed: %s",
                    provider.name, e,
                    exc_info=True,
                )

    def commit_session_boundary_async(
        self,
        messages: List[Dict[str, Any]],
        *,
        new_session_id: str,
        parent_session_id: str = "",
        reason: str = "new_session",
    ) -> None:
        """把旧会话提取 + 提供方重绑作为**一个**串行任务排队。

        会话轮换（/new）必须保证 on_session_end（会话末提取——LLM 调用，
        可能数秒）严格**先于** on_session_switch（重绑提供方内部
        session_id/回合缓冲）。内联执行提取会阻塞 /new 命令整个 LLM
        往返（#16454）；临时线程执行则与内联 switch 竞争——提供方按内部
        状态取键，迟到的 on_session_end 会跑在 switch 后的绑定上（转录
        记错会话、旧回合缓冲重复摄入）。

        把两个钩子作为一个任务提交到管理器单后台 worker，两个性质都在
        单一咽喉点获得：调用方立即返回，worker 的 FIFO 顺序把 end→switch
        与每回合 sync_all、prefetch 串行化（它们共享同一 worker）。执行器
        不可用时 _submit_background 降级为内联执行——#16454 前的同步行为，
        慢但正确。
        """
        if not self._providers:
            return
        snapshot = list(messages or [])

        def _run() -> None:
            try:
                self.on_session_end(snapshot)
            except Exception as e:  # pragma: no cover - on_session_end guards per-provider
                logger.warning("Session-boundary extraction failed: %s", e)
            try:
                self.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=True,
                    reason=reason,
                )
            except Exception as e:  # pragma: no cover - on_session_switch guards per-provider
                logger.warning("Session-boundary switch failed: %s", e)

        self._submit_background(_run)

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """通知所有提供方 agent 的 session_id 已轮换。

        触发于 /resume、/branch、/reset、/new 与上下文压缩——任何在不拆
        掉提供方的情况下重绑 AIAgent.session_id 的路径。

        提供方继续运行；只需刷新缓存的按会话状态，让后续写入落到正确
        会话记录。完整契约见 MemoryProvider.on_session_switch。

        rewound=True 表示 session_id 未变但转录被截断；缓存按回合文档
        状态的提供方应失效。
        """
        if not new_session_id:
            return
        # 只在显式设置时转发 rewound。无条件传会向每个提供方的 **kwargs
        # 注入 rewound=False（常见 /resume、/branch、/new、压缩路径），
        # 污染捕获额外 kwargs 的提供方（并破坏精确 dict 断言）。/undo
        # 路径显式设置 rewound=True；其他路径保持干净。
        if rewound:
            kwargs["rewound"] = True
        for provider in self._providers:
            try:
                provider.on_session_switch(
                    new_session_id,
                    parent_session_id=parent_session_id,
                    reset=reset,
                    **kwargs,
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_session_switch failed: %s",
                    provider.name, e,
                )

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """上下文压缩前通知所有提供方。

        返回所有提供方要进压缩摘要 prompt 的合并文本。无贡献返回空串。
        """
        parts = []
        for provider in self._providers:
            try:
                result = provider.on_pre_compress(messages)
                if result and result.strip():
                    parts.append(result)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_pre_compress failed: %s",
                    provider.name, e,
                )
        return "\n\n".join(parts)

    @staticmethod
    def _provider_memory_write_metadata_mode(provider: MemoryProvider) -> str:
        """提供方 on_memory_write 接受 metadata 的方式。"""
        try:
            signature = inspect.signature(provider.on_memory_write)
        except (TypeError, ValueError):
            return "keyword"

        params = list(signature.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
            return "keyword"
        if "metadata" in signature.parameters:
            return "keyword"

        accepted = [
            p for p in params
            if p.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        ]
        if len(accepted) >= 4:
            return "positional"
        return "legacy"

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """内置 memory 工具写入时通知外部提供方。

        跳过内置提供方本身（它就是写入源）。
        """
        for provider in self._providers:
            if provider.name == "builtin":
                continue
            try:
                metadata_mode = self._provider_memory_write_metadata_mode(provider)
                if metadata_mode == "keyword":
                    provider.on_memory_write(
                        action, target, content, metadata=dict(metadata or {})
                    )
                elif metadata_mode == "positional":
                    provider.on_memory_write(action, target, content, dict(metadata or {}))
                else:
                    provider.on_memory_write(action, target, content)
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_memory_write failed: %s",
                    provider.name, e,
                )

    # 桥接镜像到外部提供方的动作。内置 memory 工具也能返回非变更形态
    # （错误、待审批记录）；那些在到达提供方前由 notify_memory_tool_write
    # 过滤掉。
    _MIRRORED_MEMORY_ACTIONS = {"add", "replace", "remove"}

    @staticmethod
    def _memory_tool_result_succeeded(result: Any) -> bool:
        """仅当内置 memory 工具真正提交了写入时为 True。

        关闭即失败：非 JSON 的字符串、非 dict 结果、缺 success、或待审批
        写入（staged is True）都返回 False，外部提供方绝不会被告知一次
        没落地的写入。
        """
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                return False
        if not isinstance(result, dict):
            return False
        return result.get("success") is True and result.get("staged") is not True

    def notify_memory_tool_write(
        self,
        tool_result: Any,
        tool_args: Dict[str, Any],
        *,
        build_metadata: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        """把内置 memory 工具调用镜像给外部提供方。

        这是 agent 循环运行内置 memory 工具后调用的唯一入口。所有
        "是否/镜像什么"的决策都在管理器接口后面：

        * 门控在已提交（非 staged、成功）的写入上；
        * 展开单操作与批量（operations）形态；
        * 只保留变更动作（add/replace/remove）；
        * 构建每次操作的血统 metadata 并转发 old_text。

        build_metadata 是可选 agent 侧回调（循环知道会话/任务/工具调用
        血统，管理器不知道），每个被镜像的操作调用一次。
        """
        if not self._memory_tool_result_succeeded(tool_result):
            return

        target = str(tool_args.get("target") or "memory")
        operations = tool_args.get("operations")
        if isinstance(operations, list) and operations:
            raw_operations = operations
        else:
            raw_operations = [{
                "action": tool_args.get("action"),
                "content": tool_args.get("content"),
                "old_text": tool_args.get("old_text"),
            }]

        for op in raw_operations:
            if not isinstance(op, dict):
                continue
            action = str(op.get("action") or "")
            if action not in self._MIRRORED_MEMORY_ACTIONS:
                continue
            try:
                metadata = dict(build_metadata() if build_metadata else {})
                old_text = op.get("old_text")
                if old_text:
                    metadata["old_text"] = str(old_text)
                self.on_memory_write(
                    action,
                    target,
                    str(op.get("content") or ""),
                    metadata=metadata,
                )
            except Exception as e:
                logger.debug("notify_memory_tool_write failed for op %s: %s", action, e)

    def on_delegation(
        self, task: str, result: str, *, child_session_id: str = "", **kwargs
    ) -> None:
        """通知所有提供方一个子 agent 完成。"""
        for provider in self._providers:
            try:
                provider.on_delegation(
                    task, result, child_session_id=child_session_id, **kwargs
                )
            except Exception as e:
                logger.debug(
                    "Memory provider '%s' on_delegation failed: %s",
                    provider.name, e,
                )

    def shutdown_all(self) -> None:
        """关闭所有提供方（逆序干净拆除）。

        先有界排空后台 sync/prefetch 执行器（_SYNC_DRAIN_TIMEOUT_S），让
        回合最终 sync 有机会落地再拆提供方。工作线程是 daemon，超过排空
        窗口仍卡住的部分随解释器结束，而不是阻塞退出。
        """
        self._drain_sync_executor()
        for provider in reversed(self._providers):
            try:
                provider.shutdown()
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' shutdown failed: %s",
                    provider.name, e,
                )

    @property
    def shutdown_drain_state(self) -> Dict[str, Any]:
        """最近一次有界关闭排空结果的快照。"""
        with self._sync_executor_lock:
            return dict(self._shutdown_drain_state)

    def _drain_sync_executor(self) -> None:
        """给排队的 FIFO 工作一个有界机会，然后显式放弃。"""
        with self._sync_executor_lock:
            self._shutting_down = True
            executor = self._sync_executor
            self._sync_executor = None
            tracked = dict(self._background_futures)
            self._shutdown_drain_state = {
                "status": "draining" if executor is not None else "drained",
                "abandoned_writes": 0,
                "abandoned_prefetches": 0,
                "active_tasks": sum(not future.done() for future in tracked),
            }
        if executor is None:
            return

        # shutdown(wait=False) 关闭提交但不碰 FIFO。等待被跟踪的 future
        # 让真实单 worker 执行器按序跑完排队的写/边界任务直到截止。
        executor.shutdown(wait=False, cancel_futures=False)
        _, pending = wait(tuple(tracked), timeout=_SYNC_DRAIN_TIMEOUT_S)
        if not pending:
            with self._sync_executor_lock:
                self._shutdown_drain_state.update(status="drained", active_tasks=0)
            return

        abandoned_writes = 0
        abandoned_prefetches = 0
        active_tasks = 0
        for future in pending:
            kind = tracked[future]
            if future.cancel():
                if kind == "prefetch":
                    abandoned_prefetches += 1
                else:
                    abandoned_writes += 1
            else:
                active_tasks += 1

        with self._sync_executor_lock:
            self._shutdown_drain_state.update(
                status="timed_out",
                abandoned_writes=abandoned_writes,
                abandoned_prefetches=abandoned_prefetches,
                active_tasks=active_tasks,
            )
        logger.warning(
            "Memory shutdown drain timed out after %.2fs; abandoning %d queued "
            "memory write(s) and %d queued prefetch(es); %d active task(s) remain detached",
            _SYNC_DRAIN_TIMEOUT_S,
            abandoned_writes,
            abandoned_prefetches,
            active_tasks,
        )

    def initialize_all(self, session_id: str, **kwargs) -> None:
        """初始化所有提供方。

        自动把 ``hermes_home`` 注入 *kwargs*，让每个提供方无需自己 import
        get_hermes_home() 就能解析按 profile 的存储路径。
        """
        if "hermes_home" not in kwargs:
            from hermes_constants import get_hermes_home

            kwargs["hermes_home"] = str(get_hermes_home())
        for provider in self._providers:
            try:
                provider.initialize(session_id=session_id, **kwargs)
            except Exception as e:
                logger.warning(
                    "Memory provider '%s' initialize failed: %s",
                    provider.name, e,
                )
