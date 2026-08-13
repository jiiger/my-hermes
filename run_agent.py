import logging
import os
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent import relay_runtime
from agent.interrupt_compat import request_hard_interrupt
from agent.iteration_budget import IterationBudget
from agent.process_bootstrap import OpenAI, _get_proxy_for_base_url

# 经 run_agent 命名空间暴露给 agent.system_prompt._ra()（测试 patch 契约）
from agent.prompt_builder import build_environment_hints
from tools.interrupt import set_interrupt as _set_interrupt
from utils import base_url_hostname

logger = logging.getLogger(__name__)

# 消息 dict 上标记"已写入 SessionDB"的内部键（对应原版
# run_agent.py:_DB_PERSISTED_MARKER）。flush 用它去重：已标记的消息
# 后续 flush 直接跳过，不依赖 list 切片长度（压缩会重建列表）。
_DB_PERSISTED_MARKER = "_db_persisted"


class AIAgent:
    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""
        self._base_url_hostname = base_url_hostname(value)

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # Default tool-calling iterations (shared with subagents)
        tool_delay: float = None,  # Deprecated: accepted for compatibility, ignored
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        tool_progress_mode: str = "all",
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        openrouter_min_coding_score: Optional[float] = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        read_terminal_callback: callable = None,
        read_preview_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        notice_callback: callable = None,
        notice_clear_callback: callable = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
        reaction_callback: Optional[Callable[[str], None]] = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        user_id_alt: str = None,
        user_name: str = None,
        chat_id: str = None,
        chat_name: str = None,
        chat_type: str = None,
        thread_id: str = None,
        gateway_session_key: str = None,
        skip_context_files: bool = False,
        load_soul_identity: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 20,
        checkpoint_max_total_size_mb: int = 500,
        checkpoint_max_file_size_mb: int = 10,
        pass_session_id: bool = False,
        requested_provider: str = None,
    ):

        from agent.agent_init import init_agent

        init_agent(
            self,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            requested_provider=requested_provider,
            api_mode=api_mode,
            acp_command=acp_command,
            acp_args=acp_args,
            command=command,
            args=args,
            model=model,
            max_iterations=max_iterations,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            save_trajectories=save_trajectories,
            verbose_logging=verbose_logging,
            quiet_mode=quiet_mode,
            tool_progress_mode=tool_progress_mode,
            ephemeral_system_prompt=ephemeral_system_prompt,
            log_prefix_chars=log_prefix_chars,
            log_prefix=log_prefix,
            providers_allowed=providers_allowed,
            providers_ignored=providers_ignored,
            providers_order=providers_order,
            provider_sort=provider_sort,
            provider_require_parameters=provider_require_parameters,
            provider_data_collection=provider_data_collection,
            openrouter_min_coding_score=openrouter_min_coding_score,
            session_id=session_id,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            thinking_callback=thinking_callback,
            reasoning_callback=reasoning_callback,
            clarify_callback=clarify_callback,
            read_terminal_callback=read_terminal_callback,
            read_preview_callback=read_preview_callback,
            step_callback=step_callback,
            stream_delta_callback=stream_delta_callback,
            interim_assistant_callback=interim_assistant_callback,
            tool_gen_callback=tool_gen_callback,
            status_callback=status_callback,
            notice_callback=notice_callback,
            notice_clear_callback=notice_clear_callback,
            event_callback=event_callback,
            reaction_callback=reaction_callback,
            max_tokens=max_tokens,
            reasoning_config=reasoning_config,
            service_tier=service_tier,
            request_overrides=request_overrides,
            prefill_messages=prefill_messages,
            platform=platform,
            user_id=user_id,
            user_id_alt=user_id_alt,
            user_name=user_name,
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type=chat_type,
            thread_id=thread_id,
            gateway_session_key=gateway_session_key,
            skip_context_files=skip_context_files,
            load_soul_identity=load_soul_identity,
            skip_memory=skip_memory,
            session_db=session_db,
            parent_session_id=parent_session_id,
            iteration_budget=iteration_budget,
            fallback_model=fallback_model,
            credential_pool=credential_pool,
            checkpoints_enabled=checkpoints_enabled,
            checkpoint_max_snapshots=checkpoint_max_snapshots,
            checkpoint_max_total_size_mb=checkpoint_max_total_size_mb,
            checkpoint_max_file_size_mb=checkpoint_max_file_size_mb,
            pass_session_id=pass_session_id,
        )

    def _create_openai_client(
        self, client_kwargs: dict, *, reason: str, shared: bool
    ) -> Any:
        """Forwarder — see ``agent.agent_runtime_helpers.create_openai_client``."""
        from agent.agent_runtime_helpers import create_openai_client

        return create_openai_client(self, client_kwargs, reason=reason, shared=shared)

    @staticmethod
    def _build_keepalive_http_client(base_url: str = "", *, verify: Any = True) -> Any:
        """构建带空闲连接回收的 httpx.Client（原版 run_agent.py:4713）。

        keepalive_expiry=20.0：在反向代理（通常 30-60s）断开之前回收
        空闲连接，防止 CLOSE-WAIT 堆积；read=None 保证 SSE 流式响应
        不会被读超时掐断。
        """
        try:
            import httpx as _httpx

            _proxy = _get_proxy_for_base_url(base_url)
            _limits = _httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=20.0,
            )
            _timeout = _httpx.Timeout(connect=15.0, read=None, write=15.0, pool=10.0)
            _mounts = {}
            if _proxy is None:
                # 无代理时挂普通 transport，防止 httpx 默认 trust_env 读到
                # 系统级代理（macOS getproxies 不含 ExceptionsList）
                _mounts = {
                    "http://": _httpx.HTTPTransport(verify=verify),
                    "https://": _httpx.HTTPTransport(verify=verify),
                }
            return _httpx.Client(
                limits=_limits,
                timeout=_timeout,
                proxy=_proxy,
                mounts=_mounts or None,
                verify=verify,
            )
        except Exception:
            return None

    def _client_log_context(self) -> str:
        """客户端日志上下文（provider/base_url/model）。"""
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return f"provider={provider} base_url={base_url} model={model}"

    def _safe_print(self, *args, **kwargs) -> None:
        """线程安全的 print 包装（对应原版 run_agent.py 的 _safe_print）。

        精简版直接用 print + flush；以后接入流式/多线程时在此加锁即可。
        """
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

    def _needs_thinking_reasoning_pad(self) -> bool:
        """判断当前 provider 是否需要 reasoning_content 回填。

        DeepSeek / Kimi(Moonshot) / 小米 MiMo 的 thinking 模式会拒绝缺少
        reasoning_content 的 assistant 工具调用消息回放（原版 run_agent.py:7159）。
        精简版只做 provider 名 + host 的简单匹配。
        """
        provider = (getattr(self, "provider", "") or "").lower()
        host = (getattr(self, "_base_url_lower", "") or "").lower()
        return any(
            kw in provider or kw in host
            for kw in ("deepseek", "kimi", "moonshot", "mimo")
        )

    def _make_worker_client(self) -> Any:
        """为后台工作线程创建独立的 OpenAI 客户端（worker-local）。

        中断/stale 时主线程会 close() 请求客户端来打断阻塞中的 HTTP 调用；
        如果 close 的是共享的 agent.client，本轮后续请求全部报废。
        worker-local client 每次调用独立创建、用完即关，中断只影响本次请求。
        """
        from agent.agent_runtime_helpers import create_openai_client

        client_kwargs = {
            "api_key": getattr(self, "api_key", None) or "",
            "base_url": getattr(self, "base_url", "") or "",
        }
        return create_openai_client(self, client_kwargs, reason="worker", shared=False)

    def _interruptible_api_call(self, api_kwargs: dict, **kw) -> Any:
        """可中断的非流式 API 调用。

        Forwarder — see ``agent.chat_completion_helpers.interruptible_api_call``。
        后台线程执行请求，主线程轮询中断标志；用户中断时立即抛
        InterruptedError，不必等完整 HTTP 往返。
        """
        from agent.chat_completion_helpers import interruptible_api_call

        return interruptible_api_call(self, api_kwargs, **kw)

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None, **kw
    ) -> Any:
        """可中断的流式 API 调用（逐增量触发 stream_delta_callback / _stream_callback）。

        Forwarder — see ``agent.chat_completion_helpers.interruptible_streaming_api_call``。
        返回聚合后的响应对象（形状与非流式一致，主循环无需感知流式）。
        """
        from agent.chat_completion_helpers import interruptible_streaming_api_call

        return interruptible_streaming_api_call(
            self, api_kwargs, on_first_delta=on_first_delta, **kw
        )

    def _has_stream_consumers(self) -> bool:
        """是否有流式消费者（注册了 stream_delta_callback / _stream_callback）。

        主循环据此决定走流式还是非流式调用路径（对应原版 run_agent.py:6418）。
        """
        return (
            getattr(self, "stream_delta_callback", None) is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _reset_stream_delivery_tracking(self) -> None:
        """重置流式交付跟踪状态（每轮 API 调用前调用）。

        对应原版 run_agent.py:6022；精简版只清累积文本，不做 scrubber 冲刷。
        """
        self._current_streamed_assistant_text = ""

    # ─── interim 中间旁白交付（对应原版 run_agent.py:6216-6420 精简版）───

    def _strip_think_blocks(self, content: str) -> str:
        """转发器 — 见 agent.agent_runtime_helpers.strip_think_blocks。"""
        from agent.agent_runtime_helpers import strip_think_blocks

        return strip_think_blocks(self, content)

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        """归一化中间文本：空白折叠 + 去首尾（供去重键比较）。"""
        if not isinstance(text, str):
            return ""
        import re as _re

        return _re.sub(r"\s+", " ", text).strip()

    def _interim_assistant_visible_text(self, assistant_msg: dict) -> str:
        """返回 assistant 消息中可交付的中间可见文本（剥推理块 + 展平）。

        对应原版 run_agent.py:6296 的精简版：砍掉 codex commentary 优先
        分支，只处理顶层 content（字符串或结构块列表）。
        """
        content = assistant_msg.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    ptype = str(part.get("type") or "").strip().lower()
                    if ptype in {"thinking", "reasoning", "redacted_thinking"}:
                        continue
                    ptext = part.get("text")
                    if isinstance(ptext, str) and ptext:
                        parts.append(ptext)
            text = "\n".join(parts)
        elif isinstance(content, dict):
            text = str(content.get("text") or content.get("content") or "")
        else:
            text = str(content)
        return self._strip_think_blocks(text).strip()

    def _interim_text_was_delivered(self, text: str) -> bool:
        """中间文本是否已交付过（归一化查重）。"""
        normalized = self._normalize_interim_visible_text(text)
        if not normalized:
            return False
        return normalized in getattr(self, "_delivered_interim_texts", set())

    def _record_delivered_interim_text(self, text: str) -> None:
        """记录中间文本已交付。"""
        normalized = self._normalize_interim_visible_text(text)
        if normalized:
            delivered = getattr(self, "_delivered_interim_texts", None)
            if not isinstance(delivered, set):
                delivered = set()
                self._delivered_interim_texts = delivered
            delivered.add(normalized)

    def _interim_content_was_streamed(self, content: str) -> bool:
        """内容是否已作为流式 delta 显示过（前缀匹配，非精确相等）。

        最终响应可能是流式文本 + 尾部增量，或流在旁白触发时只发了一部分；
        两种情况流式内容都是最终的**前缀**——前缀匹配足够标记"已预览"，
        失败方向退化为无害的重复，绝不丢文本（对齐原版 #65919 review）。
        """
        visible = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(
                getattr(self, "_current_streamed_assistant_text", "") or ""
            )
        )
        return bool(streamed) and visible.startswith(streamed)

    def _emit_interim_assistant_message(self, assistant_msg: dict) -> None:
        """把工具循环中的中间旁白消息推给 UI 层（interim_assistant_callback）。

        对应原版 run_agent.py:6344 的精简版：砍掉 codex commentary 分支与
        redact 脱敏（my-hermes 无 redact 模块）。去重依赖每回合重置的
        _delivered_interim_texts。不设置 _response_was_previewed——本方法
        处理的是普通旁白/中间确认，与最终回答的预览标记无关。
        """
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(assistant_msg, dict):
            return
        visible = self._interim_assistant_visible_text(assistant_msg)
        if (
            not visible
            or visible == "(empty)"
            or self._interim_text_was_delivered(visible)
        ):
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            cb(visible, already_streamed=already_streamed)
            self._record_delivered_interim_text(visible)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def interrupt(self, message: Optional[str] = None, *, hard_cancel: bool = False) -> None:
        """请求中断当前工具循环（对齐原版 run_agent.py:3091）。

        从另一个线程调用（如输入处理器、消息接收方），优雅地停止 agent
        并处理新消息；同时向正在执行的长时间工具（如终端命令）发送提前
        终止信号，让 agent 能立即响应。

        Args:
            message: 触发中断的可选新消息。提供时，agent 会把它纳入响应上下文。
            hard_cancel: 标记为显式停止（而非重定向或新消息打断）。压缩
                         机制会在普通中断被屏蔽时也尊重这个原子信号。

        Example (CLI):
            # 在独立的输入线程中：
            if user_typed_something:
                agent.interrupt(user_input)
        """
        # 硬停止与重定向共用一把锁，避免 /stop 与已接受的更正消息竞争，
        # 防止意外把自身变成重试。
        def _admit_hard_cancel() -> None:
            event = getattr(self, "_hard_interrupt_requested", None)
            if event is None:
                return
            fence = vars(self).get("_active_compression_commit_fence")
            cancel_before_commit = getattr(
                type(fence), "cancel_before_commit", None
            )
            if callable(cancel_before_commit):
                try:
                    # 持与 begin_commit() 相同的锁置位 Event；若提交已获胜，
                    # 先等被追踪的变更完成再发布停止信号。
                    cancel_before_commit(fence, event)
                    return
                except Exception:
                    logger.debug(
                        "Compression hard-cancel fence admission failed",
                        exc_info=True,
                    )
            event.set()

        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                self._interrupt_requested = True
                self._interrupt_message = message
                if hard_cancel:
                    _admit_hard_cancel()
                self._pending_redirect = None
        else:
            self._interrupt_requested = True
            self._interrupt_message = message
            if hard_cancel:
                _admit_hard_cancel()
            self._pending_redirect = None

        # Codex app-server 拥有自己的模型/工具循环，监听私有中断事件而
        # 非 Hermes 的按线程标志。
        if getattr(self, "api_mode", None) == "codex_app_server":
            _codex_session = getattr(self, "_codex_session", None)
            _request_interrupt = getattr(_codex_session, "request_interrupt", None)
            if callable(_request_interrupt):
                try:
                    _request_interrupt()
                except Exception:
                    logger.debug(
                        "Failed to interrupt Codex app-server turn",
                        exc_info=True,
                    )

        # cron 轮次在对话线程上执行 API 请求，以避免嵌套中断 worker 死锁；
        # 与普通 worker 路径不同，它的客户端在这里注册，跨线程中断仍能
        # 及时关掉活跃 socket。
        _abort_active_request = getattr(self, "_active_request_abort", None)
        if callable(_abort_active_request):
            try:
                _abort_active_request("interrupt_abort")
            except Exception:
                logger.debug("Failed to abort active inline request", exc_info=True)
        # 向所有工具发送立即中止在途操作的信号。范围限定在本 agent 的执行
        # 线程，同一进程内的其他 agent（网关）不受影响。
        if self._execution_thread_id is not None:
            _set_interrupt(True, self._execution_thread_id)
            self._interrupt_thread_signal_pending = False
        else:
            # 中断在 run_conversation() 把 agent 绑定到执行线程之前到达；
            # 延后工具级中断信号到启动完成，而不是误伤调用方线程。
            self._interrupt_thread_signal_pending = True
        # 扩散到并发工具 worker 线程。这些 worker 跑在各自 tid 上
        # （ThreadPoolExecutor worker），因此工具内 is_interrupted() 只有
        # 在 _interrupted_threads 集合含其 tid 时才可见中断；没有这里的
        # 扩散，已运行的并发工具（如挂在网络 I/O 上的终端命令）将感知不到
        # 中断，只能跑到自身超时。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(True, _wtid)
                except Exception:
                    pass
        # 传播到运行中的子 agent（子 agent 委派）
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                if hard_cancel:
                    request_hard_interrupt(child, message)
                else:
                    child.interrupt(message)
            except Exception as e:
                logger.debug("Failed to propagate interrupt to child agent: %s", e)
        if not self.quiet_mode:
            print(
                "\n⚡ Interrupt requested"
                + (
                    f": '{message[:40]}...'"
                    if message and len(message) > 40
                    else f": '{message}'" if message else ""
                )
            )

    def hard_interrupt(self, message: Optional[str] = None) -> None:
        """请求显式停止，同时保留 ``interrupt()`` 的 ABI（对齐原版 :3226）。

        前端可特性检测本方法，对合成/第三方 agent 回退到旧版
        ``interrupt()`` 签名。
        """
        # 有意绕过动态分发：按旧版 interrupt(message=None) ABI 写的子类
        # 可能覆写了不带新关键字 hard_cancel 的 interrupt。
        AIAgent.interrupt(self, message, hard_cancel=True)

    @property
    def is_interrupted(self) -> bool:
        """检查是否已请求中断（对齐原版 :4547）。"""
        return self._interrupt_requested

    def clear_interrupt(self, *, preserve_redirect: bool = False) -> bool:
        """清除中断请求与按线程工具信号（对齐原版 run_agent.py:3237）。

        ``preserve_redirect`` 仅由对话循环在主动取消模型请求、重建同一
        逻辑轮次时使用；公共硬停止路径保持默认，清除一切。
        """
        _redirect_lock = getattr(self, "_pending_redirect_lock", None)
        if _redirect_lock is not None:
            with _redirect_lock:
                if preserve_redirect and not self._pending_redirect:
                    return False
                self._interrupt_requested = False
                self._interrupt_message = None
                getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
                if not preserve_redirect:
                    self._pending_redirect = None
        else:
            if preserve_redirect and not getattr(self, "_pending_redirect", None):
                return False
            self._interrupt_requested = False
            self._interrupt_message = None
            getattr(self, "_hard_interrupt_requested", threading.Event()).clear()
            if not preserve_redirect:
                self._pending_redirect = None
        self._interrupt_thread_signal_pending = False
        if self._execution_thread_id is not None:
            _set_interrupt(False, self._execution_thread_id)
        # 同时清除并发工具 worker 线程位。被追踪的 worker 通常在退出时清
        # 自己的位，但这里显式清除可保证轮次边界上不会有陈旧中断残存，
        # 防止后续不相关的工具调用恰好调度到同一复用 worker tid 时被误触发。
        _tracker = getattr(self, "_tool_worker_threads", None)
        _tracker_lock = getattr(self, "_tool_worker_threads_lock", None)
        if _tracker is not None and _tracker_lock is not None:
            with _tracker_lock:
                _worker_tids = list(_tracker)
            for _wtid in _worker_tids:
                try:
                    _set_interrupt(False, _wtid)
                except Exception:
                    pass
        # 硬中断优先于任何待处理的 /steer——steer 本意是给 agent 的下一次
        # 工具迭代注入，而那次迭代不会再发生。丢弃它，避免中断后轮次上
        # 出现迟到的注入。
        _steer_lock = getattr(self, "_pending_steer_lock", None)
        if _steer_lock is not None:
            with _steer_lock:
                self._pending_steer = None
        return True

    def _execute_tool_calls(
        self,
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        """工具执行转发器：按并行安全规划分段调度工具调用。

        对应原版 run_agent.py:7589 的 _execute_tool_calls。分段规划器把
        整批拆成并行安全的极大连续运行（只读工具、不重叠文件目标）与
        顺序 barrier（交互/不安全/未识别工具）：同质批保持原有单路径
        派发；混合批按发出顺序逐段执行，安全子集仍并发、副作用顺序
        保留。工具执行契约（_tool_impls 调用、未注册写错误串、异常
        fail-open、结果 role="tool" 追加）由 tool_executor 保持一致。
        """
        # 函数内 import：agent.tool_executor 依赖 agent.tool_dispatch_helpers，
        # 避免在模块加载期形成循环导入。
        from agent.tool_executor import (
            execute_tool_calls_concurrent,
            execute_tool_calls_sequential,
        )

        tool_calls = assistant_message.tool_calls

        # 允许工具执行期间使用 _vprint/打印（原版转发器同样置位）
        self._executing_tools = True
        try:
            if len(tool_calls) <= 1:
                return execute_tool_calls_sequential(
                    self, assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_dispatch_helpers import _plan_tool_batch_segments

            # my-hermes 无 get_active_env，不传 execution_cwd
            # （planner 内部回退 Path.cwd()）
            segments = _plan_tool_batch_segments(tool_calls)

            if len(segments) == 1:
                kind = segments[0][0]
                if kind == "parallel":
                    return execute_tool_calls_concurrent(
                        self,
                        assistant_message,
                        messages,
                        effective_task_id,
                        api_call_count,
                    )
                return execute_tool_calls_sequential(
                    self, assistant_message, messages, effective_task_id, api_call_count
                )

            from agent.tool_executor import execute_tool_calls_segmented

            return execute_tool_calls_segmented(
                self,
                assistant_message,
                messages,
                effective_task_id,
                api_call_count,
                segments=segments,
            )
        finally:
            self._executing_tools = False

    def run_conversation(
        self,
        user_message: Any,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[Any] = None,
        persist_user_timestamp: Optional[float] = None,
        persist_user_display_kind: Optional[str] = None,
        persist_user_display_metadata: Optional[Dict[str, Any]] = None,
        moa_config: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """执行一次完整的对话（转发器 + Relay 观测生命周期封装）。

        本方法不实现对话逻辑本身，它负责两件事：
        1. 用 Relay（nemo_relay 观测系统）把这次对话的生命周期包起来：
           租用会话 → 开启轮次 → 上报终态 → 收尾轮次 → 释放租约，
           无论成功、失败还是被中断，finally 都会保证按序清理；
        2. 把真正干活的部分转发给对话循环实现（见下方调用处），
           本方法只关心"轮次开始前/结束后"的观测与清理，不关心对话
           循环内部如何调用模型和执行工具。

        Args:
            user_message (Any): 用户当前输入
            system_message (str, optional): 系统提示词.
            conversation_history (List[Dict[str, Any]], optional): 对话历史.
            task_id (str, optional): 单次对话任务的唯一id. Defaults to None.
            stream_callback (Optional[callable], optional): 流式回调函数.
            persist_user_message (Optional[Any], optional): 要持久化的用户消息.
            persist_user_timestamp (Optional[float], optional): 用户消息的时间戳
            persist_user_display_kind (Optional[str], optional): 持久化消息类型.
            persist_user_display_metadata (Optional[Dict[str, Any]], optional): 持久化消息的展示元数据.
            moa_config (Optional[dict[str, Any]], optional): 多模型配置.

        Returns:
            Dict[str, Any]: 对话结果字典（由对话循环返回，通常含 final_response 与 messages）
        """

        # 延迟导入真正的对话循环实现（放在方法内而不是模块顶部，
        # 避免与 conversation_loop 之间产生循环导入）
        from agent.conversation_loop import run_conversation

        # 任务ID：调用方没传就现场生成一个 UUID（每个任务对应一次独立对话）
        effective_task_id = task_id or str(uuid.uuid4())
        # 会话ID：从 agent 属性上取；取不到时兜底为空字符串
        session_id = str(getattr(self, "session_id", None) or "")
        # 任务上下文：会话/任务/平台三要素，供 Relay 观测与终态上报使用
        task_context = {
            "session_id": session_id,
            "task_id": effective_task_id,
            "platform": getattr(self, "platform", None) or "",
        }

        # 为这一轮生成全局唯一的 Relay 轮次ID：会话:任务:随机串，
        # 用于在观测端把同一次对话里的多轮区分开
        relay_turn_id = (
            f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex}"
        )
        # 记录"正在进行的轮次ID"，供中断检测/收尾时判断本轮是否还需清理
        self._relay_pending_turn_id = relay_turn_id

        # 子代理场景才需要传递父会话的 session_id：
        # 只有当平台是 subagent 且有 _parent_session_id 时才传，否则为空串。
        # （当前精简版还没有子代理功能，_parent_session_id 无人赋值，
        #   此分支恒走 else，保留结构以备后续扩展）
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        # 预声明生命周期状态：租约 / 轮次上下文 / 轮次终态。
        # relay_outcome 默认 "failed"，确保任何异常退出路径都有明确终态可上报
        relay_lease = None
        relay_turn = None
        relay_outcome = "failed"

        try:
            # ① 租用本次对话：拿到一张"对话使用权"凭证（ConversationLease）。
            #    host 按 profile 单例复用；若 nemo_relay 不可用会自动降级为空壳，
            #    保证对话流程不因观测系统缺失而中断
            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"],
                platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            # ② 开启本轮 Relay 轮次作用域：在此之后，轮次内的所有 Relay
            #    事件都会挂到本轮作用域下。若同一会话已有并发轮次在跑，
            #    协调器会自动跳过插桩（relay_enabled=False），不会崩
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease, turn_id=relay_turn_id, task_id=effective_task_id
            )

            # TODO Nous Portal tagging（原版在此给对话打 Portal 归因标签，精简版暂不实现）

            # ③ 转发：真正执行对话循环。第一个参数 self 就是 agent 本身，
            #    循环内部会完成 LLM 调用、工具执行、上下文管理等完整流程。
            #    这里只关心调用结果，不关心其内部实现（具体逻辑在
            #    agent/conversation_loop 模块中）
            result = run_conversation(
                self,
                user_message,
                system_message,
                conversation_history,
                effective_task_id,
                stream_callback,
                persist_user_message,
                persist_user_timestamp=persist_user_timestamp,
                persist_user_display_kind=persist_user_display_kind,
                persist_user_display_metadata=persist_user_display_metadata,
                moa_config=moa_config,
            )

            # ④ 把对话结果映射为 Relay 轮次终态（outcome）：
            #    interrupted → cancelled（用户主动中断）
            #    failed      → failed（对话失败）
            #    其他        → success（正常完成）
            terminal = result if isinstance(result, dict) else {}
            if terminal.get("interrupted") is True:
                relay_outcome = "cancelled"
            elif terminal.get("failed") is True:
                relay_outcome = "failed"
            else:
                relay_outcome = "success"

            # ⑤ 正常路径：在轮次收尾前，先关闭本轮产生的所有逻辑 LLM
            #    子作用域（倒序回收），带上最终 outcome 一并上报
            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn, outcome=relay_outcome
            )

            # 正常完成，把对话结果原样返回给调用方
            return result

        except BaseException as exc:
            # ⑥ 异常路径：先把异常归类为 Relay 终态——
            #    Ctrl+C / asyncio 取消 → cancelled；超时 → timed_out；
            #    其余异常保持默认的 failed
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            # 只要轮次已开启，就把异常终态上报给 Relay（逻辑 LLM 子作用域一并收掉）
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn,
                    outcome=relay_outcome,
                )

            # 原样重新抛出：本方法只负责观测与清理，不吞异常，
            # 错误处理交给上层调用方
            raise
        finally:
            # ⑦ 无论成功/失败/中断，都按"后开先关"的顺序清理（与 Relay
            #    scope 栈的 LIFO 语义一致）。多层 try/finally 嵌套保证：
            #    即使某一步清理抛异常，后续清理仍然执行。
            try:
                # ⑦a 收尾轮次：pop 掉本轮 Relay 作用域并上报终态 outcome
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(
                        relay_turn, outcome=relay_outcome
                    )
            finally:
                try:
                    # ⑦b 释放对话租约：只标记 released，不关闭会话本身，
                    #     会话保持可恢复（下次对话还能接着用）
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(
                            relay_lease
                        )
                finally:
                    # ⑦c 清理本轮遗留的活动标签（如"压缩中/执行工具中"等
                    #     展示状态）；清理失败也直接忽略，不能影响收尾
                    try:
                        self._reset_activity_labels_after_turn()
                    except Exception:
                        pass
                    # ⑦d 若 pending 轮次ID仍指向本轮，说明没有更晚的轮次
                    #     接管，清空它（中断递归场景下用于恢复状态）
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None

    def _copy_reasoning_content_for_api(self, source_msg: dict, api_msg: dict) -> None:
        """Forwarder — see ``agent.agent_runtime_helpers.copy_reasoning_content_for_api``."""
        from agent.agent_runtime_helpers import copy_reasoning_content_for_api

        return copy_reasoning_content_for_api(self, source_msg, api_msg)

    # ───────────────────────── SessionDB（情节记忆）─────────────────────────

    def _get_session_db_for_recall(self):
        """返回 SessionDB 供检索，未注入时惰性打开默认 state.db。

        对应原版 run_agent.py:600 _get_session_db_for_recall：大部分调用方
        显式传入 session_db，但历史检索值得在缺失时降级打开默认库；
        惰性打开的库由本 agent 拥有，close() 负责关闭。
        """
        if self._session_db is not None:
            return self._session_db
        try:
            from hermes_state import SessionDB

            self._session_db = SessionDB()
            # 这里打开的是默认库，本 agent 是唯一持有者
            self._owns_session_db = True
            return self._session_db
        except Exception:
            logger.debug("SessionDB unavailable for recall", exc_info=True)
            return None

    def _ensure_db_session(self) -> None:
        """首次使用时幂等创建 SessionDB 会话行；失败保留重试机会。

        对应原版 run_agent.py:628 _ensure_db_session：_session_db_created
        置位前失败只记警告，下次 flush 会重试；成功后才置位。
        """
        if self._session_db_created or not self._session_db:
            return
        source = str(getattr(self, "platform", "") or "cli") or "cli"
        try:
            self._session_db.create_session(
                session_id=self.session_id,
                source=source,
                model=getattr(self, "model", None),
                system_prompt=getattr(self, "_cached_system_prompt", None),
            )
            self._session_db_created = True
        except Exception as e:
            # 瞬时失败（如锁竞争）：_session_db_created 保持 False，下次重试
            self._last_persistence_error = str(e)
            logger.warning(
                "Session DB creation failed (will retry next flush): %s", e
            )

    def _flush_messages_to_session_db(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ):
        """带锁的会话持久化入口（对应原版 run_agent.py:1998）。

        返回 True 表示本次 flush 成功；False 表示写入失败
        （_incremental_persistence_failed 已置位，可下次重试）；
        未启用 DB 时返回 None。
        """
        persist_lock = getattr(self, "_session_persist_lock", None)
        if persist_lock is None:
            return self._flush_messages_to_session_db_unlocked(
                messages, conversation_history
            )
        with persist_lock:
            return self._flush_messages_to_session_db_unlocked(
                messages, conversation_history
            )

    def _flush_messages_to_session_db_unlocked(
        self,
        messages: List[Dict],
        conversation_history: Optional[List[Dict]] = None,
    ):
        """把未写入 SessionDB 的消息追加落库（对应原版 run_agent.py:2010）。

        去重机制：
        - 每条消息 dict 上盖章 ``_DB_PERSISTED_MARKER``，已写即跳过；
        - conversation_history 中的 dict（同一对象）视为已持久化历史，
          只盖章不重写——因此调用方显式传入的既有历史不会重复落库；
        - 压缩重建出的新 dict（浅拷贝）会当作新消息写入，其中 head/tail
          历史通过把压缩后列表作为 conversation_history 传入而跳过。

        落库行规则：
        - persist_user_message / timestamp 覆盖只作用于写库的行，
          绝不动 live messages；
        - content 为 list 时只保留文本部分，图片转 ``[screenshot]``，
          禁止把 base64 大图写进 state.db；
        - tool_calls 原样 JSON 序列化。

        写失败返回 False 并置位 _incremental_persistence_failed 与
        _last_persistence_error；下一次 flush 重新扫描（前缀快照清空）。
        """
        if not self._session_db:
            return None
        try:
            # 会话行可能尚未创建（首轮），补一次幂等创建
            if not self._session_db_created:
                self._ensure_db_session()
                if not self._session_db_created:
                    return False  # 创建失败，等待下次 flush 重试

            history_ids = {
                id(item)
                for item in (conversation_history or [])
                if isinstance(item, dict)
            }

            # 有界扫描：跳过与上次成功 flush 快照同对象的前缀
            _scan_start = 0
            _prev_prefix = getattr(self, "_db_flush_scan_prefix", None)
            if isinstance(_prev_prefix, list):
                _limit = min(len(_prev_prefix), len(messages))
                while (
                    _scan_start < _limit
                    and messages[_scan_start] is _prev_prefix[_scan_start]
                ):
                    _scan_start += 1

            _batch_rows: List[Dict] = []
            _batch_msgs: List[Dict] = []
            _ov_idx = getattr(self, "_persist_user_message_idx", None)
            _ov_content = getattr(self, "_persist_user_message_override", None)
            _ov_timestamp = getattr(self, "_persist_user_message_timestamp", None)
            for _msg_idx in range(_scan_start, len(messages)):
                msg = messages[_msg_idx]
                if not isinstance(msg, dict):
                    continue
                if msg.get(_DB_PERSISTED_MARKER):
                    continue
                # 已在历史里的消息：视为已持久化，盖章跳过
                if id(msg) in history_ids:
                    msg[_DB_PERSISTED_MARKER] = True
                    continue
                role = msg.get("role", "unknown")
                content = msg.get("content")
                _row_timestamp = msg.get("timestamp")
                # persist override 只写库、不改 live（对应原版 #48677）
                if _ov_idx == _msg_idx and role == "user":
                    if _ov_content is not None and (
                        not isinstance(content, list) or isinstance(_ov_content, list)
                    ):
                        content = _ov_content
                    if _ov_timestamp is not None:
                        _row_timestamp = _ov_timestamp
                # 多模态/结构化 content：只保留文本，图片转占位符
                if isinstance(content, list):
                    _txt = []
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") == "text":
                            _txt.append(str(part.get("text", "")))
                        elif part.get("type") in {"image", "image_url", "input_image"}:
                            _txt.append("[screenshot]")
                    content = "\n".join(_txt) if _txt else None
                tool_calls_data = None
                if isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                _batch_rows.append({
                    "role": role,
                    "content": content,
                    "tool_name": msg.get("tool_name"),
                    "tool_calls": tool_calls_data,
                    "tool_call_id": msg.get("tool_call_id"),
                    "finish_reason": msg.get("finish_reason"),
                    "reasoning": msg.get("reasoning"),
                    "reasoning_content": msg.get("reasoning_content"),
                    "timestamp": _row_timestamp,
                    "api_content": msg.get("api_content"),
                    "display_kind": msg.get("display_kind"),
                    "display_metadata": msg.get("display_metadata"),
                })
                _batch_msgs.append(msg)

            # 单事务写整批；失败时不盖章、不回写前缀快照，下次全量重扫
            if _batch_rows:
                self._session_db.append_messages_batch(
                    session_id=self.session_id,
                    messages=_batch_rows,
                )
                for _written in _batch_msgs:
                    _written[_DB_PERSISTED_MARKER] = True
            self._db_flush_scan_prefix = messages[:]
            self._incremental_persistence_failed = False
            return True
        except Exception as e:
            # 异常中断：清空前缀快照，下次 flush 从 0 重扫
            self._db_flush_scan_prefix = None
            self._incremental_persistence_failed = True
            self._last_persistence_error = str(e)
            logger.warning("Session DB append failed: %s", e)
            return False

    def _reset_activity_labels_after_turn(self) -> None:
        """清理本轮遗留的活动标签（如"压缩中/执行工具中"等展示状态）。

        保留 ``_last_activity_ts`` 让空闲/看门狗时钟保持连续；清空描述与
        来源，避免缓存的 agent 在轮次结束后仍展示上一轮的中间状态。
        清理失败也直接忽略，不能影响收尾（对应原版 run_agent.py 的实例方法）。
        """
        from agent.session_activity import ActivityProvenance

        self._last_activity_desc = ""
        self._last_activity_provenance = ActivityProvenance.UNKNOWN
        session_id = getattr(self, "session_id", None)
        session_db = getattr(self, "_session_db", None)
        if not session_id or session_db is None:
            return
        clear = getattr(session_db, "clear_session_activity_labels", None)
        if not callable(clear):
            return
        try:
            clear(session_id)
        except Exception:
            # Never let durable cleanup I/O break turn teardown.
            pass

    def _build_system_prompt_parts(self, system_message: str = None) -> Dict[str, str]:
        """Forwarder — see ``agent.system_prompt.build_system_prompt_parts``."""
        from agent.system_prompt import build_system_prompt_parts

        return build_system_prompt_parts(self, system_message=system_message)

    def _build_system_prompt(self, system_message: str = None) -> str:
        """Forwarder — see ``agent.system_prompt.build_system_prompt``."""
        from agent.system_prompt import build_system_prompt

        return build_system_prompt(self, system_message=system_message)

    def _invalidate_system_prompt(self) -> None:
        """Forwarder — see ``agent.system_prompt.invalidate_system_prompt``."""
        from agent.system_prompt import invalidate_system_prompt

        invalidate_system_prompt(self)

    def close(self) -> None:
        """关闭 agent，释放资源（对应原版 AIAgent.close 的 SessionDB + 记忆部分）。

        SessionDB 收尾顺序（对齐原版 run_agent.py:4417-4445）：
        1. 尽力 flush 最后已知消息（_session_messages，对话循环收尾写入）；
        2. end_session("agent_close") 终结会话行（首个 end_reason 生效）；
        3. 只关闭本 agent 自己创建的 DB（外部注入的 session_db 不关闭）。
        其余资源：关闭外部记忆 provider（MemoryManager.shutdown_all）。
        幂等可重复调用。
        """
        # ① 兜底 flush 最后已知消息快照（若有）
        _session_messages = getattr(self, "_session_messages", None)
        if _session_messages:
            try:
                self._flush_messages_to_session_db(_session_messages)
            except Exception:
                pass
        # ② 终结会话行（幂等；已结束则 no-op）
        _session_db = getattr(self, "_session_db", None)
        try:
            if _session_db is not None and getattr(self, "session_id", None):
                _session_db.end_session(
                    getattr(self, "session_id"), "agent_close"
                )
        except Exception:
            pass
        # ③ 只关闭自己创建的 DB（外部注入的会话库留给调用方管理）
        try:
            if getattr(self, "_owns_session_db", False) and _session_db is not None:
                self._owns_session_db = False
                _session_db.close()
        except Exception:
            pass
        # 外部记忆 provider 会话收尾（对齐原版 shutdown_memory_provider）：
        # 先 on_session_end（会话末提取，如 holographic auto_extract），
        # 再 shutdown_all（有界排空 + 逆序关闭）。
        self.shutdown_memory_provider(_session_messages)

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """会话边界关闭外部记忆 provider（对齐原版 run_agent.py:4121）。

        调用 MemoryManager.on_session_end（会话末提取）然后 shutdown_all
        （有界排空 + 逆序关闭）。只在真正的会话边界调用（CLI 退出、/new
        轮换等），NOT 每轮——每轮 shutdown 会杀掉 provider。
        """
        _mm = getattr(self, "_memory_manager", None)
        if _mm is None:
            return
        try:
            _mm.on_session_end(messages or [])
        except Exception as exc:
            logger.warning(
                "Memory provider on_session_end failed during shutdown: %s",
                exc,
                exc_info=True,
            )
        try:
            _mm.shutdown_all()
        except Exception:
            pass


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
):

    # TODO 可用工具分类

    # 加载项目根 .env（python run_agent.py 入口同样支持 .env 凭据）
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

    # 凭据兜底：命令行没传时从环境变量/.env 读取（OPENAI_API_KEY / DEEPSEEK_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL）
    api_key = (
        api_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    base_url = base_url or os.getenv("OPENAI_BASE_URL") or ""
    model = model or os.getenv("OPENAI_MODEL") or ""

    # 创建agent
    try:
        agent = AIAgent(base_url=base_url, model=model, api_key=api_key)

    except RuntimeError as exc:
        print(f"Failed to initialize agent : {exc}")
        return

    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )

    else:
        user_query = query

    print(f"\n User Query : {user_query}")
    print("\n" + "=" * 50)

    result = agent.run_conversation(user_query)

    if result["final_response"]:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result["final_response"])

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹

    # TODO 保存样本轨迹


if __name__ == "__main__":
    # CLI 入口：python run_agent.py "你的问题" [--model ...] [--api-key ...] [--base-url ...]
    import argparse

    _parser = argparse.ArgumentParser(description="my-hermes agent 命令行入口")
    _parser.add_argument(
        "query", nargs="?", default=None, help="用户问题（不传则用内置示例问题）"
    )
    _parser.add_argument(
        "--model", default="", help="模型名（默认读环境变量 OPENAI_MODEL）"
    )
    _parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key（默认读 OPENAI_API_KEY）",
    )
    _parser.add_argument(
        "--base-url",
        dest="base_url",
        default=None,
        help="API base URL（默认读 OPENAI_BASE_URL）",
    )
    _args = _parser.parse_args()
    main(
        query=_args.query,
        model=_args.model,
        api_key=_args.api_key,
        base_url=_args.base_url,
    )
