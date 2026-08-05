from urllib.parse import parse_qs, urlparse, urlunparse


def init_agent(agent,base_url:str = None,api_model:str = None,api_key:str = None,provider:str = None,model:str = None,quiet_mode:bool = False):
    """
    初始化 agent
        
        api_model:指定 API 协议（如标准的 chat_completions 或 OpenAI 最新的 codex_responses）
        quiet_mode:安静模式
    """
    
    #TODO 按api_mode做分类，默认chat_completion(openAi兼容客户端)
    
    
    #TODO _install_safe_stdio()
            
            
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
            #Fallback：从 Provider Profile 读取默认 Headers
            
            
            
            
            
            agent.client = agent._create_openai_client(client_kwargs,reason = "agent_init",shared = True) 
            
    else:
        from agent.auxiliary_client import reresolve_provider_client