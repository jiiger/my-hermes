import datetime
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse, urlunparse

from agent.iteration_budget import IterationBudget

logger = logging.getLogger("run_agent")


def init_agent(
    agent,
    base_url: str = None,
    api_mode: str = None,
    api_key: str = None,
    provider: str = None,
    model: str = None,
    quiet_mode: bool = False,
    max_iterations: int = 90,
    fallback_chain: list = None,
    # ── 配置性状态（原版 init_agent 的同名参数，见 hermes-agent/agent/agent_init.py:459）──
    ephemeral_system_prompt: Optional[str] = None,
    prefill_messages: Optional[List[Dict[str, Any]]] = None,
    api_max_retries: int = 3,
    api_stale_timeout: float = 180.0,
    tools: Optional[List[Dict[str, Any]]] = None,
    valid_tool_names: Optional[set] = None,
    tool_impls: Optional[Dict[str, Callable]] = None,
    stream_delta_callback: Optional[Callable] = None,
    session_id: str = None,
):
    """
    初始化 agent

        api_model:指定 API 协议（如标准的 chat_completions 或 OpenAI 最新的 codex_responses）
        quiet_mode:安静模式
        max_iterations:一轮对话中最大的 API 调用次数（工具循环上限，默认 90）
        fallback_chain:回退链 [{provider, model, base_url, api_key}]，主 provider 重试耗尽后依次切换
        ephemeral_system_prompt:临时系统提示词（叠加在系统提示之后）
        prefill_messages:预填充消息（插在系统提示之后、历史之前）
        api_max_retries:API 调用失败的最大重试次数（默认 3）
        api_stale_timeout:API 调用无响应判定超时秒数（默认 180，防永久挂起）
        tools:OpenAI 格式的工具 schema 列表（主循环作为 tools= 传给 API）
        valid_tool_names:合法工具名集合（校验用）
        tool_impls:{工具名: 实现函数}，工具执行时查找调用
        stream_delta_callback:流式文本增量回调（显示层，逐 token 触发）
    """
    # ── 0. 进程级安全 stdio（守护线程/无头环境防 print 崩溃）──
    from agent.process_bootstrap import _install_safe_stdio

    _install_safe_stdio()

    # ── 基础属性 ──
    agent.model = model  # 模型名（主循环 create(model=...) 使用）
    agent.quiet_mode = quiet_mode  # 安静模式：跳过非必要打印
    agent.provider = provider_name or ""  # provider 名（小写；回退切换时更新）
    agent.api_mode = _api_mode  # API 协议（chat_completions / codex_responses / ...）
    # base_url 优先用 client_kwargs 里的干净 URL（已剥 query），没有则用原始参数
    agent.base_url = client_kwargs.get("base_url", base_url or "")

    agent.session_start = datetime.now()
    if session_id:
        # Use provided session ID (e.g., from CLI)
        agent.session_id = session_id
    else:
        # Generate a new session ID
        timestamp_str = agent.session_start.strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:6]
        agent.session_id = f"{timestamp_str}_{short_uuid}"

    # ── 凭据 / 客户端 ──
    agent._client_kwargs = client_kwargs  # 原始 kwargs（供后续重建 client 参考）
    agent.api_key = client_kwargs.get("api_key", "")  # API key（worker 工厂读取）
    agent.client = _client  # 共享 OpenAI 客户端（构建于第 2 步）

    # ── 主循环前置依赖：迭代预算 / 中断状态（原版 agent_init.py:586-589, 780, 892）──
    agent.max_iterations = max_iterations  # 一轮对话允许的最大 API 调用次数
    agent.iteration_budget = IterationBudget(
        max_iterations
    )  # 线程安全预算计数器（.remaining > 0 是循环条件之一）
    # 以下两个是运行期状态（每轮由 request_interrupt() / 主循环管理），不作为 init 参数：
    agent._budget_grace_call = (
        False  # 预算用尽后的"宽限一次"标记（原版用于最后的收尾调用）
    )
    agent._interrupt_requested = (
        False  # 中断标志：request_interrupt() 置位，主循环检查到即退出
    )

    # ── 工具注册（tools/valid_tool_names/tool_impls 为 init 参数，默认空）──
    agent.tools = list(
        tools or []
    )  # OpenAI 格式的工具 schema 列表（循环里作为 tools= 传给 API）
    agent.valid_tool_names = set(valid_tool_names or set())  # 合法工具名集合（校验用）
    agent._tool_impls = dict(
        tool_impls or {}
    )  # {工具名: 实现函数}，供 _execute_tool_calls 查找执行

    # ── 主循环用到的其他默认状态（原版在 agent_init 各处初始化；配置项均为 init 参数）──
    agent.ephemeral_system_prompt = (
        ephemeral_system_prompt  # 临时系统提示词（叠加在系统提示之后）
    )
    agent.prefill_messages = list(
        prefill_messages or []
    )  # 预填充消息（插在系统提示之后、历史之前）
    agent._api_max_retries = api_max_retries  # API 调用失败的最大重试次数
    agent._api_stale_timeout = api_stale_timeout  # API 调用无响应判定超时（防永久挂起）

    # ── 流式回调：stream_delta_callback 是 init 参数（显示层）；_stream_callback 是
    #    运行期状态——每轮由 run_conversation 的 stream_callback 参数经 build_turn_context 设置 ──
    agent.stream_delta_callback = stream_delta_callback  # 流式文本增量回调（显示层）
    agent._stream_callback = (
        None  # 流式文本增量回调（主回调，来自 stream_callback 参数）
    )
    agent._current_streamed_assistant_text = (
        ""  # 本轮已流式输出的累积文本（运行期状态，每轮重置）
    )

    # ── 回退（fallback）状态：fallback_chain 是 init 参数，其余为运行期状态 ──
    agent._fallback_chain = list(
        fallback_chain or []
    )  # [{provider, model, base_url, api_key}]，空=不回退
    agent._fallback_index = 0  # 回退链游标（try_activate_fallback 前进，运行期）
    agent._fallback_activated = False  # 是否已切到回退（记录主运行时快照用，运行期）
    agent._primary_runtime = {}  # 主 provider 快照 {provider, model, base_url, api_key}，恢复用（运行期）
    agent._unavailable_fallback_keys = set()  # 标记不可用的回退项（缺网络等，运行期）
    agent._rate_limited_until = (
        0.0  # 限流冷却截止（monotonic），冷却期不恢复主 provider（运行期）
    )

    provider_name = (
        provider.strip().lower()
        if isinstance(provider, str) and provider.strip()
        else None
    )

    # api_mode 分类，默认 chat_completions（OpenAI 兼容客户端）
    if api_mode in {
        "chat_completions",
        "codex_responses",
        "anthropic_messages",
        "bedrock_converse",
        "codex_app_server",
    }:
        _api_mode = api_mode
    else:
        _api_mode = "chat_completions"

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

    # ══════════════════════════════════════════════════════════════
    # 2. SSL 校验 + 客户端构建（凭据/证书校验失败立即报错）
    # ══════════════════════════════════════════════════════════════
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback

        verify_ca_bundle_with_fallback()

        _client = agent._create_openai_client(
            client_kwargs, reason="agent_init", shared=True
        )
        # TODO 验证
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    # TODO 。。。。。。
