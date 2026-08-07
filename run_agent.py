import logging
import uuid
from typing import Any, Dict, List, Optional

from agent import relay_runtime
from agent.process_bootstrap import _get_proxy_for_base_url
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
    ) -> Dict[str, any]:
        """_summary_

        Args:
            user_message (Any): 用户当前输入
            system_message (str, optional): 系统提示词.
            conversation_history (List[Dict[str, Any]], optional): 对话历史.
            task_id (str, optional): 单词对话任务的唯一id. Defaults to None.
            stream_callback (Optional[callable], optional): 流式回调函数.
            persist_user_message (Optional[Any], optional): 要持久化的用户消息.
            persist_user_timestamp (Optional[float], optional): 用户消息的时间戳
            persist_user_display_kind (Optional[str], optional): 持久化消息类型.
            persist_user_display_metadata (Optional[Dict[str, Any]], optional):
            moa_config (Optional[dict[str, Any]], optional): 多模型配置.

        Forwarder — see ``agent.conversation_loop.run_conversation``
        """

        effective_task_id = task_id or str(uuid.uuid4())
        session_id = str(getattr(self, "session_id", None), "")
        task_context = {
            "session_id": session_id,
            "task_id": task_id,
            "platform": getattr(self, "platform", None) or "",
        }

        relay_turn_id = (
            f"{session_id or 'session'}:{effective_task_id}:{uuid.uuid4().hex}"
        )
        self._relay_pending_turn_id = relay_turn_id

        # 判断是不是subagent,需不需要传递父会话的 session_id
        relay_parent_session_id = (
            str(getattr(self, "_parent_session_id", None) or "")
            if task_context["platform"] == "subagent"
            else ""
        )
        relay_lease = None
        relay_turn = None
        relay_outcome = "failed"
        """token = None
        acct_token = None
        task_start = False
        task_finish = False"""

        try:
            relay_lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=task_context["session_id"],
                platform=task_context["platform"],
                parent_session_id=relay_parent_session_id,
                model=str(getattr(self, "model", None) or ""),
            )
            relay_turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                relay_lease, turn_id=relay_turn_id, task_id=effective_task_id
            )
            """if getattr(relay_turn, "relay_enable", True):
                start_task_run(
                    **task_context,
                    parent_session_id=getattr(self, "_parent_session_id", None) or "",
                )"""
            task_start = True

            # TODO Nous Portal tagging

            result = self.run_conversation(
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

            terminal = result if isinstance(result, dict) else {}
            if terminal.get("interrupted") is True:
                relay_outcome = "cancelled"
            elif terminal.get("failed") is True:
                relay_outcome = "failed"
            else:
                relay_outcome = "success"

            relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                relay_turn, outcome=relay_outcome
            )

            return result

        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, InterruptedError)) or (
                type(exc).__name__ == "CancelledError"
            ):
                relay_outcome = "cancelled"
            elif isinstance(exc, TimeoutError):
                relay_outcome = "timed_out"
            if relay_turn is not None:
                relay_runtime.SESSION_COORDINATOR.finish_logical_calls(
                    relay_turn,
                    outcome=relay_outcome,
                )

            raise
        finally:
            try:
                if relay_turn is not None:
                    relay_runtime.SESSION_COORDINATOR.end_turn(
                        relay_turn, outcome=relay_outcome
                    )
            finally:
                try:
                    if relay_lease is not None:
                        relay_runtime.SESSION_COORDINATOR.release_conversation(
                            relay_lease
                        )
                finally:
                    try:
                        self._reset_activity_labels_after_turn()
                    except Exception:
                        pass
                    if getattr(self, "_relay_pending_turn_id", None) == relay_turn_id:
                        self._relay_pending_turn_id = None


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
):

    # TODO 可用工具分类

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

    # TODO 定义run_conversation（）
    resule = agent.run_conversation(user_query)

    if resule["final_response"]:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(resule["final_response"])

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


def _reset_activity_labels_after_turn(self) -> None:
    """Drop mid-turn activity labels once the turn is no longer running.

    Keeps ``_last_activity_ts`` so idle/watchdog clocks stay continuous
    across interrupt-recursive turns (#15654) and between turns. Clears
    description + provenance so idle cached agents / SessionDB listings
    do not keep advertising the last mid-turn stamp (e.g. compression
    or tool execution) after the turn ended (#72039).
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
