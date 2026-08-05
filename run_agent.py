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