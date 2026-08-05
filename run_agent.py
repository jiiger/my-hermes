from typing import Any
from urllib.parse import urlparse, urlunparse
from utils import base_url_hostname

from agent.process_bootstrap import (
    OpenAI,  # noqa: F401  # re-exported for tests that mock.patch("run_agent.OpenAI")
    _SafeWriter,  # noqa: F401  # re-exported for tests that `from run_agent import _SafeWriter`
    _get_proxy_for_base_url,
)
import logging
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
        
        
        
    def __init__(self,base_url:str = None,api_mode:str = None,api_key:str = None,provider:str = None,model:str = None,quiet_mode:bool = False):
        
        from agent.agent_init import init_agent
        init_agent(self,base_url= base_url,api_key=api_key, provider= provider,api_mode=api_mode,model=model,quiet_mode= quiet_mode)
        
    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
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


def main(query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",):
    
    
    
    #TODO 可用工具分类
    
    
    
    #创建agent
    try:
        agent = AIAgent(
            base_url= base_url,
            model= model,
            api_key = api_key
        )
        
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
    
    #TODO 定义run_conversation（）
    resule = agent.run_conversation(user_query)
    
    if resule["final_response"]:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(resule['final_response'])
        
    
    
    #TODO 保存样本轨迹