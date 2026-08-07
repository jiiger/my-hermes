import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from agent import relay_runtime
from agent.process_bootstrap import OpenAI, _get_proxy_for_base_url
from utils import base_url_hostname

logger = logging.getLogger(__name__)


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
        api_mode: str = None,
        api_key: str = None,
        provider: str = None,
        model: str = None,
        quiet_mode: bool = False,
        max_iterations: int = 90,
    ):

        from agent.agent_init import init_agent

        init_agent(
            self,
            base_url=base_url,
            api_key=api_key,
            provider=provider,
            api_mode=api_mode,
            model=model,
            quiet_mode=quiet_mode,
            max_iterations=max_iterations,
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

    def request_interrupt(self) -> None:
        """请求中断当前轮次：置位 _interrupt_requested。

        主循环在每个迭代开头检查该标志（对应原版 run_agent.py 的中断机制），
        检查到即停止本轮对话。中断是协作式的：正在执行的工具调用不会被强杀。
        """
        self._interrupt_requested = True

    def _execute_tool_calls(
        self,
        assistant_message,
        messages: list,
        effective_task_id: str,
        api_call_count: int = 0,
    ) -> None:
        """顺序执行 assistant 消息里的工具调用，并把结果追加进 messages。

        对应原版 run_agent.py:7589 的 _execute_tool_calls；精简版不做
        并发/分段调度，按声明顺序逐个执行：
        - 参数解析失败 → 按空参执行；
        - 工具未注册（不在 _tool_impls 里）→ 结果写错误信息；
        - 工具执行抛异常 → 结果写异常信息（fail-open，对话循环可继续）。
        执行结果以 role="tool" 消息追加，模型下一轮就能看到。
        """
        del effective_task_id, api_call_count  # 精简版未使用；保留参数以对齐原版签名
        import json

        tool_impls = getattr(self, "_tool_impls", {})
        for tc in assistant_message.tool_calls or []:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            impl = tool_impls.get(name)
            if impl is None:
                result = f"错误: 未注册的工具 {name}"
            else:
                try:
                    result = impl(**args)
                except Exception as exc:
                    result = f"工具执行异常: {type(exc).__name__}: {exc}"
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": str(result),
            })

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
