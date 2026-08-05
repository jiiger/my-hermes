import logging
from urllib.parse import parse_qs, urlparse, urlunparse

logger = logging.getLogger("run_agent")

def init_agent(agent,base_url:str = None,api_mode:str = None,api_key:str = None,provider:str = None,model:str = None,quiet_mode:bool = False):
    """
    初始化 agent
        
        api_model:指定 API 协议（如标准的 chat_completions 或 OpenAI 最新的 codex_responses）
        quiet_mode:安静模式
    """
    #TODO _install_safe_stdio()
    
    agent.model = model
    agent.quiet_mode = quiet_mode
    agent.base_url = base_url or ""
    provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
    
    
    #TODO 按api_mode做分类，默认chat_completion(openAi兼容客户端)
    if api_mode in {"chat_completions", "codex_responses", "anthropic_messages", "bedrock_converse", "codex_app_server"}:
        agent.api_mode = api_mode
    else:
        agent.api_mode = "chat_completions"
        
    
    #TODO  credential_pool 校验
    #TODO 预热 transport 缓存（_get_transport()）
    #TODO 中断/steer/redirect/子代理等状态字段初始化
    
    # TODO _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)
    
    
    #TODO 分类    anthropic_messages   moa  bedrock，以下是else
    if api_key and base_url:
            # Explicit credentials from CLI/gateway — construct directly.
            # The runtime provider resolver already handled auth for us.
            # Extract query params (e.g. Azure api-version) from base_url
            # and pass via default_query to prevent loss during SDK URL
            # joining (httpx drops query string when joining paths).
            _parsed_url = urlparse(base_url)
            if _parsed_url.query:
                _clean_url = urlunparse(_parsed_url._replace(query=""))
                _query_params = {
                    k: v[0] for k, v in parse_qs(_parsed_url.query).items()
                }
                client_kwargs = {
                    "api_key": api_key,
                    "base_url": _clean_url,
                    "default_query": _query_params,
                }
            else:
                client_kwargs = client_kwargs = {"api_key": api_key, "base_url": base_url}
                
                
            #TODO 超时设置：如果用户配置了provider 级别的超时时间，注入到客户端参数
            
            
            #TODO 按host匹配注入 Provider 专属 Headers
            #TODO Fallback：从 Provider Profile 读取默认 Headers
            
    agent._client_kwargs = client_kwargs
            
        
            
            
    agent.api_key = client_kwargs.get("api_key", "")
    agent.base_url = client_kwargs.get("base_url", agent.base_url)
            
    #构建agent
            
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback 
        verify_ca_bundle_with_fallback()
        
        agent.client = agent._create_openai_client(client_kwargs, reason="agent_init", shared=True)
        #TODO 验证
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        
    from agent.agent_runtime_helpers import sync_credential_pool_entry_id
    #TODO sync_credential_pool_entry_id(agent)
        
    #TODO 。。。。。。
        
        
        
    