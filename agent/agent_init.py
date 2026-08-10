import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from agent.iteration_budget import IterationBudget

logger = logging.getLogger("run_agent")


def init_agent(
    agent,
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
    """
    初始化 agent

        base_url:OpenAI 兼容 API 的 base URL
        api_key:API key（与 base_url 同时提供才生效）
        provider:provider 名（小写；用于 reasoning pad 判断、回退切换等）
        api_mode:指定 API 协议（如标准的 chat_completions 或 OpenAI 最新的 codex_responses）
        model:模型名（主循环 create(model=...) 使用）
        max_iterations:一轮对话中最大的 API 调用次数（工具循环上限，默认 90）
        quiet_mode:安静模式：跳过非必要打印
        fallback_model:回退配置——单个 dict {provider, model, base_url, api_key}
            或 list of dict（链式），主 provider 重试耗尽后依次切换
        ephemeral_system_prompt:临时系统提示词（叠加在系统提示之后）
        prefill_messages:预填充消息（插在系统提示之后、历史之前）
        stream_delta_callback:流式文本增量回调（显示层，逐 token 触发）
        session_id:会话 ID（不传则自动生成）
        iteration_budget:迭代预算实例（不传则按 max_iterations 新建）
    """
    # ── 0. 进程级安全 stdio（守护线程/无头环境防 print 崩溃）──
    from agent.process_bootstrap import _install_safe_stdio

    _install_safe_stdio()

    # ══════════════════════════════════════════════════════════════
    # ① 基础配置（原版 agent_init.py:589-619）
    # ══════════════════════════════════════════════════════════════
    agent.model = model  # 模型名（主循环 create(model=...) 使用）
    agent.max_iterations = max_iterations  # 一轮对话允许的最大 API 调用次数
    agent.iteration_budget = iteration_budget or IterationBudget(
        max_iterations
    )  # 线程安全预算计数器（.remaining > 0 是循环条件之一）
    agent.quiet_mode = quiet_mode  # 安静模式：跳过非必要打印
    agent.ephemeral_system_prompt = (
        ephemeral_system_prompt  # 临时系统提示词（叠加在系统提示之后）
    )
    agent.platform = (
        platform  # 平台标识（"cli" / "telegram" / "discord" 等，relay 观测用）
    )
    agent._user_id = user_id  # 平台用户标识
    agent._user_id_alt = user_id_alt  # 备用平台标识
    agent._user_name = user_name  # 用户名
    agent._chat_id = chat_id  # 会话 ID（平台侧）
    agent._chat_name = chat_name  # 会话名
    agent._chat_type = chat_type  # 会话类型
    agent._thread_id = thread_id  # 线程 ID
    agent._gateway_session_key = gateway_session_key  # 每会话稳定的 key
    agent.log_prefix = f"{log_prefix} " if log_prefix else ""  # 日志前缀

    # ══════════════════════════════════════════════════════════════
    # ② provider / base_url（原版 agent_init.py:620-629）
    # ══════════════════════════════════════════════════════════════
    provider_name = (
        provider.strip().lower()
        if isinstance(provider, str) and provider.strip()
        else None
    )
    agent.base_url = base_url or ""  # 基础 URL（client_kwargs 组装后还会回填干净版）
    agent.provider = provider_name or ""  # provider 名（小写；回退切换时更新）
    agent.requested_provider = requested_provider  # 调用方显式请求的 provider
    agent._credential_pool = (
        credential_pool  # 凭据池（原版 credential_pool 校验用，精简版暂不启用）
    )

    # ══════════════════════════════════════════════════════════════
    # ③ api_mode 分类（原版 agent_init.py:633-670，多 provider 特化已裁剪）
    # ══════════════════════════════════════════════════════════════
    if api_mode in {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
    }:
        agent.api_mode = api_mode
    else:
        agent.api_mode = "chat_completions"

    # ══════════════════════════════════════════════════════════════
    # ④ 回调（原版 agent_init.py:757-774；精简版只挂有签名对应的）
    # ══════════════════════════════════════════════════════════════
    agent.stream_delta_callback = (
        stream_delta_callback  # 流式文本增量回调（显示层，逐 token 触发）
    )
    agent.thinking_callback = thinking_callback  # 思考内容回调
    agent.reasoning_callback = reasoning_callback  # 推理内容回调
    agent.status_callback = status_callback  # 状态变更回调

    # ══════════════════════════════════════════════════════════════
    # ⑤ 运行期状态（原版 agent_init.py:779-784）
    # ══════════════════════════════════════════════════════════════
    agent._interrupt_requested = (
        False  # 中断标志：request_interrupt() 置位，主循环检查到即退出
    )

    # ══════════════════════════════════════════════════════════════
    # ⑥ provider 配置（原版 agent_init.py:828-846）
    # ══════════════════════════════════════════════════════════════
    agent.providers_allowed = providers_allowed  # 允许的 provider 列表（回退时过滤）
    agent.providers_ignored = providers_ignored  # 忽略的 provider 列表
    agent.providers_order = providers_order  # provider 优先级顺序
    agent.provider_sort = provider_sort  # provider 排序规则
    agent.provider_require_parameters = (
        provider_require_parameters  # 是否强制要求模型参数
    )
    agent.provider_data_collection = provider_data_collection  # 数据收集偏好
    agent.openrouter_min_coding_score = (
        openrouter_min_coding_score  # OpenRouter 最低编码分
    )
    agent.max_tokens = max_tokens  # 单次响应最大 token 数（None = 模型默认）
    agent.reasoning_config = reasoning_config  # 推理配置（effort 等）
    agent.service_tier = service_tier  # 服务等级
    agent.request_overrides = dict(
        request_overrides or {}
    )  # 请求级覆盖（extra_body 等）
    agent.prefill_messages = list(
        prefill_messages or []
    )  # 预填充消息（插在系统提示之后、历史之前）

    # ══════════════════════════════════════════════════════════════
    # ⑦ 预算（原版 agent_init.py:894-895）
    # ══════════════════════════════════════════════════════════════
    agent._budget_grace_call = (
        False  # 预算用尽后的"宽限一次"标记（原版用于最后的收尾调用）
    )

    # ══════════════════════════════════════════════════════════════
    # ⑧ 流式状态（原版 agent_init.py:961-980；scrubber 等已裁剪）
    # ══════════════════════════════════════════════════════════════
    agent._stream_callback = None  # 流式文本增量回调（主回调，每轮由 run_conversation 的 stream_callback 参数设置）
    agent._current_streamed_assistant_text = (
        ""  # 本轮已流式输出的累积文本（运行期状态，每轮重置）
    )

    # ══════════════════════════════════════════════════════════════
    # ⑨ 凭据组装 + 共享 client（原版 agent_init.py:1300-1377）
    # ══════════════════════════════════════════════════════════════
    # TODO  credential_pool 校验
    # TODO 预热 transport 缓存（_get_transport()）
    # TODO 中断/steer/redirect/子代理等状态字段初始化
    # TODO _provider_timeout = get_provider_request_timeout(agent.provider, agent.model)

    # 凭据组装：显式凭据 → client_kwargs（提取 query 参数防 httpx 丢失）；
    # 无显式凭据 → 空 kwargs 兜底（原版走 provider router，裁剪版不路由）
    # TODO 分类    anthropic_messages   moa  bedrock，以下是else
    if api_key and base_url:
        # Extract query params (e.g. Azure api-version) from base_url
        # and pass via default_query to prevent loss during SDK URL
        # joining (httpx drops query string when joining paths).
        _parsed_url = urlparse(base_url)
        if _parsed_url.query:
            _clean_url = urlunparse(_parsed_url._replace(query=""))
            _query_params = {k: v[0] for k, v in parse_qs(_parsed_url.query).items()}
            client_kwargs = {
                "api_key": api_key,
                "base_url": _clean_url,
                "default_query": _query_params,
            }
        else:
            client_kwargs = {"api_key": api_key, "base_url": base_url}

        # TODO 超时设置：如果用户配置了provider 级别的超时时间，注入到客户端参数
        # TODO 按host匹配注入 Provider 专属 Headers
        # TODO Fallback：从 Provider Profile 读取默认 Headers
    else:
        client_kwargs = {}

    # SSL 校验 + 客户端构建（凭据/证书校验失败立即报错）
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback

        verify_ca_bundle_with_fallback()

        _client = agent._create_openai_client(
            client_kwargs, reason="agent_init", shared=True
        )
        # TODO 验证
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    agent._client_kwargs = client_kwargs  # 存储原始 kwargs（中断后重建 client 用）
    agent.api_key = client_kwargs.get("api_key", "")  # API key（worker 工厂读取）
    # base_url 回填 client_kwargs 里的干净 URL（已剥 query）
    agent.base_url = client_kwargs.get("base_url", agent.base_url)
    agent.client = _client  # 共享 OpenAI 客户端

    # ══════════════════════════════════════════════════════════════
    # ⑩ 回退（fallback）：fallback_model 是 init 参数（原版 agent_init.py:1410-1421）
    #    功能待实现——这里只保留"参数 → 状态"归一化与运行期属性占位，
    #    切换逻辑（try_activate_fallback / restore_primary_runtime）以后补
    # ══════════════════════════════════════════════════════════════
    if isinstance(fallback_model, list):
        # 链式回退：list of {provider, model, base_url, api_key}
        agent._fallback_chain = [
            f
            for f in fallback_model
            if isinstance(f, dict) and f.get("provider") and f.get("model")
        ]
    elif (
        isinstance(fallback_model, dict)
        and fallback_model.get("provider")
        and fallback_model.get("model")
    ):
        # 单个回退：dict {provider, model, base_url, api_key}
        agent._fallback_chain = [fallback_model]
    else:
        agent._fallback_chain = []
    agent._fallback_index = 0  # 回退链游标（try_activate_fallback 前进，运行期）
    agent._fallback_activated = False  # 是否已切到回退（记录主运行时快照用，运行期）
    # Legacy attribute kept for backward compat (tests, external callers)
    agent._fallback_model = agent._fallback_chain[0] if agent._fallback_chain else None
    if agent._fallback_chain and not agent.quiet_mode:
        if len(agent._fallback_chain) == 1:
            fb = agent._fallback_chain[0]
            print(f"🔄 Fallback model: {fb['model']} ({fb['provider']})")
        else:
            print(
                f"🔄 Fallback chain ({len(agent._fallback_chain)} providers): "
                + " → ".join(
                    f"{f['model']} ({f['provider']})" for f in agent._fallback_chain
                )
            )

    # ══════════════════════════════════════════════════════════════
    # ⑪ 工具注册（原版 agent_init.py:1438-1447；精简版从 model_tools 装配：
    #    显式导入工具模块触发自注册 → 按 enabled/disabled_toolsets 过滤
    #    生成 OpenAI 格式 schema → 生成 {工具名: handler} 实现映射）
    # ══════════════════════════════════════════════════════════════
    # 函数内 import 避免循环导入（model_tools → tools.* → tools.registry）
    from model_tools import build_tool_impls_map, get_tool_definitions

    agent.tools = get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=agent.quiet_mode,
    )  # OpenAI 格式的工具 schema 列表（主循环作为 tools= 传给 API）
    agent.valid_tool_names = {
        t["function"]["name"] for t in agent.tools
    }  # 合法工具名集合（校验用）
    agent._tool_impls = (
        build_tool_impls_map()
    )  # {工具名: 实现函数}，供 _execute_tool_calls 查找执行
    # TODO 这些属性被 system_prompt 的守卫读取，先占位 None，做记忆模块时激活
    agent._memory_store = None  # 记忆存储
    agent._memory_enabled = False  # 记忆开关
    agent._memory_manager = None  # 外部记忆提供者（原版 1699）
    # system_prompt 守卫读取（原版 agent_init.py:1781/613；config 读取未实现，先用默认值）
    agent._tool_use_enforcement = "auto"  # 工具使用强制指导开关（auto/true/false/list）
    agent.pass_session_id = pass_session_id  # 是否在系统提示中暴露会话 ID

    # ══════════════════════════════════════════════════════════════
    # ⑫ session 生成（原版 agent_init.py:1497-1505）
    # ══════════════════════════════════════════════════════════════
    agent.session_start = datetime.now()
    if session_id:
        # Use provided session ID (e.g., from CLI)
        agent.session_id = session_id
    else:
        # Generate a new session ID
        timestamp_str = agent.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        agent.session_id = f"{timestamp_str}_{short_uuid}"

    # ══════════════════════════════════════════════════════════════
    # ⑬ 重试（原版 agent_init.py:1851）
    # ══════════════════════════════════════════════════════════════
    # TODO _raw_api_retries = _agent_section.get("api_max_retries", 3)，读取config
    try:
        _raw_api_retries = 3
        _api_retries = int(_raw_api_retries)
        _api_retries = max(_api_retries, 1)  # 1 = no retry (single attempt)
    except (TypeError, ValueError):
        _api_retries = 3
    agent._api_max_retries = _api_retries  # API 调用失败的最大重试次数
    agent._api_stale_timeout = 180.0  # API 调用无响应判定超时（防永久挂起）

    # ══════════════════════════════════════════════════════════════
    # ⑭ 回退运行期状态（原版 agent_init.py:2789 的 _primary_runtime 在函数末尾）
    # ══════════════════════════════════════════════════════════════
    agent._primary_runtime = {}  # 主 provider 快照 {provider, model, base_url, api_key}，恢复用
    agent._unavailable_fallback_keys = set()  # 标记不可用的回退项（缺网络等）
    agent._rate_limited_until = (
        0.0  # 限流冷却截止（monotonic），冷却期不恢复主 provider
    )

    # TODO 。。。。。。
