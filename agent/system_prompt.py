import logging
from typing import Any, Dict, List, Optional

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    HERMES_AGENT_HELP_GUIDANCE,
    MEMORY_GUIDANCE,
    SESSION_SEARCH_GUIDANCE,
    SKILLS_GUIDANCE,
    TASK_COMPLETION_GUIDANCE,
    TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
)

logger = logging.getLogger(__name__)


def _ra():
    """Lazy reference to the ``run_agent`` module.

    Helpers like ``load_soul_md``, ``build_environment_hints``,
    ``build_context_files_prompt``, ``build_nous_subscription_prompt``,
    ``build_skills_system_prompt`` and ``get_toolset_for_tool`` are
    imported into ``run_agent``'s namespace.  Many tests
    ``patch("run_agent.load_soul_md", ...)``; if we imported them
    directly here those patches would not reach us.  Looking them up
    through ``run_agent`` on every call preserves the patch contract.
    """
    import run_agent

    return run_agent


def build_system_prompt_parts(
    agent: Any, system_message: Optional[str] = None
) -> Dict[str, str]:
    """
    > 将系统提示词组装成三个有序的缓存层级（cache tiers）。
    >
    > 返回一个包含三个键的字典：
    >
    > - stable（稳定层）—— 跨会话稳定的前缀；当后面紧跟工作区快照时，该层一直延伸到"编码操作简报（coding
    >   operating brief）"为止。
    >
    > - context（上下文层）—— 先是工作区快照，接着是其余会话内稳定的指导内容、上下文文件（AGENTS.md 等），以及调
    >   用方传入的 system_message。
    >
    > - volatile（易变层）—— 技能索引、记忆快照、用户画像（USER.md）、外部记忆提供方块、时间戳行。
    >
    > 这三层由 :func:build_system_prompt 拼接成单个字符串，并缓存在 agent._cached_system_prompt 上，随 AIAgent 存  > 活整个生命周期。Hermes 绝不会在会话中途重新渲染这个字符串的任何部分这是让上游提示词缓存（prompt cache）在
    > 跨轮次时保持命中（warm）的唯一办法。"""

    _r = _ra()

    # 拿到agent的上下文窗口大小
    _ctx_len: Optional[int] = None
    _cc = getattr(agent, "context_compressor", None)
    if _cc is not None:
        _cc_len = getattr(_cc, "context_length", None)
        if isinstance(_cc_len, int) and _cc_len > 0:
            _ctx_len = _cc_len

    # ── Stable tier ────────────────────────────────────────────────
    stable_parts: List[str] = []

    # 1 TODO 加载soul.md或者默认，直接默认

    stable_parts.append(DEFAULT_AGENT_IDENTITY)

    # 2 Hermes 自身帮助指引
    stable_parts.append(HERMES_AGENT_HELP_GUIDANCE)

    # 3 并行工具调用指导
    if getattr(agent, "_task_completion_guidance", True) and agent.valid_tool_names:
        stable_parts.append(TASK_COMPLETION_GUIDANCE)

    # 4 按工具注入的行为指导
    tool_guidance = []
    if "memory" in agent.valid_tool_names:
        tool_guidance.append(MEMORY_GUIDANCE)
    if "session_search" in agent.valid_tool_names:
        tool_guidance.append(SESSION_SEARCH_GUIDANCE)
    if "skill_manage" in agent.valid_tool_names:
        tool_guidance.append(SKILLS_GUIDANCE)

    # 5 TODO Kanban（看板）工人/编排者生命周期指导：只在调度器 spawn 的进程里出现（由 HERMES_KANBAN_TASK 环境变量触发，普通聊天永远不会看到）

    if tool_guidance:
        stable_parts.append(" ".join(tool_guidance))

    # 6 TODO 转向通道说明（steer channel）
    # 7 TODO computer use
    # 8 TODO nous
    # 9 工具使用强制指导
    if agent.valid_tool_names:
        _enforce = agent._tool_use_enforcement
        _inject = False
        if _enforce is True or (
            isinstance(_enforce, str)
            and _enforce.lower() in {"true", "always", "yes", "on"}
        ):
            _inject = True
        elif _enforce is False or (
            isinstance(_enforce, str)
            and _enforce.lower() in {"false", "never", "no", "off"}
        ):
            _inject = False
        elif isinstance(_enforce, list):
            model_lower = (agent.model or "").lower()
            _inject = any(
                p.lower() in model_lower for p in _enforce if isinstance(p, str)
            )
        else:
            # "auto" or any unrecognised value — use hardcoded defaults
            model_lower = (agent.model or "").lower()
            _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
        if _inject:
            stable_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
            _model_lower = (agent.model or "").lower()
            # Google model operational guidance (conciseness, absolute
            # paths, parallel tool calls, verify-before-edit, etc.)
            # TODO 按模型操作指导

    # 10 TODO skill tools
    # 11 TODO 阿里云适配

    # 12 运行环境提示（wsl,win,host）
    _env_hints = _r.build_environment_hints()
    if _env_hints:
        stable_parts.append(_env_hints)

    # 13 TODO 编码姿态 agent/coding_context
    # 14 TODO 本地 Python 工具链探测，我用wsl,先跳过
    # 15 TODO  活动 profile 提示
    # 16 TODO 平台提示

    # ── Context tier (cwd-dependent, may change between sessions) ─
    context_parts: List[str] = []

    # 1 TODO coding_workspace_parts
    # 2 system_message
    if system_message is not None:
        context_parts.append(system_message)

    # 上下文文件
    """ 未禁用上下文文件时，调用 build_context_files_prompt 发现并加载 AGENTS.md / .cursorrules 等（前面           
    _scan_context_content + _truncate_content 就在这内部被调用）。                                             
                                                                                                               
  - cwd=resolve_context_cwd()：优先 TERMINAL_CWD（网关模式）；CLI 下为 None → 回退启动目录。                   
  - skip_soul=_soul_loaded：stable 层已加载 SOUL.md 就不重复扫。                                               
  - allow_install_tree_fallback=(platform in ("cli","tui"))：只有 CLI/TUI 允许回退到 Hermes 安装树内部（用户就 
    是在开发 Hermes，启动目录就是用户 shell 的 cwd，是有意选择）；其他表面（桌面聊天面板、网关守护进程）自     
    spawn 进安装树，若回退会把本仓库贡献者 AGENTS.md 注入（对应 #64590 和 _is_install_tree）。"""

    # ── Volatile tier (most likely to differ on a rebuild; kept last so the stable prefix stays reusable) ──
    volatile_parts: List[str] = []
    # Skills are runtime-mutable: the agent adds and patches them across a
    # session (SKILLS_GUIDANCE tells it to patch a skill the moment it goes
    # stale). The built prompt is cached per session and only rebuilt on
    # compaction/restore (see build_system_prompt), so a skill change is not
    # byte-stable across rebuilds. With the index in the stable band, a rebuild
    # that picked up a skill change would bust the cached prefix from the index
    # down, taking the whole scaffold with it. Render it at the FRONT of the
    # volatile band instead, ahead of the turn-varying memory/timestamp tail:
    # on an implicit longest-prefix backend an unchanged index still falls
    # inside the reused prefix, and a changed one only re-prefills from here on.
    # (No effect for single-block cache_control backends, where the whole
    # system message is one cache unit regardless of internal order.)
    # 1 TODO 技能提示词

    # 2 内置记忆（冻结快照注入，对应原版 system_prompt.py:523-535）
    if getattr(agent, "_memory_store", None):
        if agent._memory_enabled:
            mem_block = agent._memory_store.format_for_system_prompt("memory")
            if mem_block:
                volatile_parts.append(mem_block)
        # USER.md 启用时始终包含
        if agent._user_profile_enabled:
            user_block = agent._memory_store.format_for_system_prompt("user")
            if user_block:
                volatile_parts.append(user_block)

    # 3 外部记忆提供方（聚合 provider 的系统提示块，对齐原版
    #    system_prompt.py:535-541；_memory_manager 为 None 时跳过）
    if getattr(agent, "_memory_manager", None):
        try:
            _ext_mem_block = agent._memory_manager.build_system_prompt()
            if _ext_mem_block:
                volatile_parts.append(_ext_mem_block)
        except Exception:
            pass

    # 3½ 技能索引（程序记忆目录，对齐原版 system_prompt.py:299-327）：
    #   有 skills 工具时注入；空 skills 目录返回空串跳过。
    if "skills_list" in agent.valid_tool_names:
        try:
            from tools.skills_tool import build_skills_system_prompt

            _skills_block = build_skills_system_prompt()
            if _skills_block:
                volatile_parts.append(_skills_block)
        except Exception:
            pass

    from hermes_time import now as _hermes_now

    now = _hermes_now()

    timestamp_line = f"Conversation started: {now.strftime('%A, %B %d, %Y')}"
    if agent.pass_session_id and agent.session_id:
        timestamp_line += f"\nSession ID: {agent.session_id}"
    if agent.model:
        timestamp_line += f"\nModel: {agent.model}"
    if agent.provider:
        timestamp_line += f"\nProvider: {agent.provider}"
    if agent.platform:
        timestamp_line += f"\nPlatform: {agent.platform}"
    volatile_parts.append(timestamp_line)

    return {
        "stable": "\n\n".join(p.strip() for p in stable_parts if p and p.strip()),
        "context": "\n\n".join(p.strip() for p in context_parts if p and p.strip()),
        "volatile": "\n\n".join(p.strip() for p in volatile_parts if p and p.strip()),
    }


def build_system_prompt(agent: Any, system_message: Optional[str] = None) -> str:
    """组装完整系统提示词：三层拼接 + 缓存到 agent._cached_system_prompt。

    对应原版 agent/system_prompt.py:build_system_prompt。精简版差异：
    - 不做 _cached_system_prompt_static 两段式布局（无 session 恢复场景）；
    - 不做截断警告的状态通道上报（无 _emit_status），只拼接并缓存返回。
    """
    parts = build_system_prompt_parts(agent, system_message=system_message)
    joined = "\n\n".join(
        p for p in (parts["stable"], parts["context"], parts["volatile"]) if p
    )
    agent._cached_system_prompt = joined
    return joined


def invalidate_system_prompt(agent: Any) -> None:
    """清空缓存的系统提示词，强制下一轮重建。

    对应原版 agent/system_prompt.py:invalidate_system_prompt（原版在上下文
    压缩后调用）。精简版差异：
    - 没有 _cached_system_prompt_static 两段式布局，只清 _cached_system_prompt；
    - 记忆模块未实现时 _memory_store 为 None，跳过磁盘重载。
    """
    agent._cached_system_prompt = None
    _store = getattr(agent, "_memory_store", None)
    if _store is not None:
        _store.load_from_disk()


def cached_prompt_reflects_builtin_memory(agent: Any, cached_prompt: str) -> bool:
    """判断缓存系统提示是否已包含当前内置记忆块。

    对应原版 agent/conversation_compression.py:211
    _cached_prompt_reflects_builtin_memory 的精简版，供压缩后
    "保留 or 重建"系统提示判定使用。

    语义：重载后的 memory/user 块必须**逐字**出现在缓存提示里（渲染文本
    含用量头，任何条目/字符数变化都会破坏包含关系 → 重建）；已清空或
    禁用的目标不能在提示里残留块头（MEMORY_BLOCK_HEADERS），否则视为
    过期 → 重建。无内置记忆时返回 True（记忆不是重建理由，对齐原版
    _builtin_memory_prompt_snapshot 返回空块的行为）。my-hermes 无外部
    memory_manager（恒 None），不需要原版的"有外部 provider 时强制重建"
    分支。
    """
    store = getattr(agent, "_memory_store", None)
    if store is None:
        return True
    try:
        from tools.memory_tool import MEMORY_BLOCK_HEADERS

        memory = (
            store.format_for_system_prompt("memory") or ""
            if getattr(agent, "_memory_enabled", False)
            else ""
        )
        user = (
            store.format_for_system_prompt("user") or ""
            if getattr(agent, "_user_profile_enabled", False)
            else ""
        )
    except Exception:
        return False

    for target, block in (("memory", memory), ("user", user)):
        block = block.strip()
        if block:
            if block not in cached_prompt:
                return False
        elif MEMORY_BLOCK_HEADERS[target] in cached_prompt:
            # 目标已清空/禁用但提示仍带它的块头——过期，需重建
            return False
    return True
