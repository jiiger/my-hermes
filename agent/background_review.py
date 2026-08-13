"""后台记忆/技能提炼（fork review agent 版，对齐原版 background_review）。

回合收尾后 spawn 一个 daemon 线程，**fork 一个受限的 review agent**
（只暴露 memory + skills 工具），用 _MEMORY_REVIEW_PROMPT /
_SKILL_REVIEW_PROMPT / _COMBINED_REVIEW_PROMPT 回放本轮对话，让 review
agent 自主决定写入 MEMORY.md / skills —— 这就是"情节 → 语义/程序记忆"
的自动固化链路。

与原版的关键对齐点：
- fork 继承父 agent 运行时（provider/model/base_url/api_key）与
  `_cached_system_prompt` / `session_id` / `session_start`（命中同一
  prompt-cache 前缀，成本低）；
- 持久化隔离：`_persist_disabled=True` / `_session_db=None` —— review
  的 harness 轮绝不写进用户真实会话；`_end_session_on_close=False` ——
  fork 的 close() 不终结父会话行；
- 只允许 memory/skill 工具（原版用线程白名单；my-hermes 直接在 fork
  时 enabled_toolsets=["memory","skills"] 限制，效果等价）；
- 节流：memory 每 N 轮（config memory.nudge_interval，默认 10）、
  skill 每 M 轮（config skills.creation_nudge_interval，默认 10）。

my-hermes 裁剪：无 routed 辅助模型（恒回放完整 snapshot）、无 curator/
hub/pinned 所有权强制（提示词里的保护语义保留为文本指引）、无 /refine
focus 命令（保留 focus 参数签名）。
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Review 提示词（原版全文）────────────────────────────────────────

_MEMORY_REVIEW_PROMPT = (
    "Review the conversation above and consider saving to memory if appropriate.\n\n"
    "Focus on:\n"
    "1. Has the user revealed things about themselves — their persona, desires, "
    "preferences, or personal details worth remembering?\n"
    "2. Has the user expressed expectations about how you should behave, their work "
    "style, or ways they want you to operate?\n\n"
    "If something stands out, save it using the memory tool. "
    "If nothing is worth saving, just say 'Nothing to save.' and stop."
)

_SKILL_REVIEW_PROMPT = (
    "Review the conversation above and update the skill library. Be "
    "ACTIVE — most sessions produce at least one skill update, even if "
    "small. A pass that does nothing is a missed learning opportunity, "
    "not a neutral outcome.\n\n"
    "Target shape of the library: CLASS-LEVEL skills, each with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries. This "
    "shapes HOW you update, not WHETHER you update.\n\n"
    "Signals to look for (any one of these warrants action):\n"
    "  • User corrected your style, tone, format, legibility, or "
    "verbosity. Frustration signals like 'stop doing X', 'this is too "
    "verbose', 'don't format like this', 'why are you explaining', "
    "'just give me the answer', 'you always do Y and I hate it', or an "
    "explicit 'remember this' are FIRST-CLASS skill signals, not just "
    "memory signals. Update the relevant skill(s) to embed the "
    "preference so the next session starts already knowing.\n"
    "  • User corrected your workflow, approach, or sequence of steps. "
    "Encode the correction as a pitfall or explicit step in the skill "
    "that governs that class of task.\n"
    "  • Non-trivial technique, fix, workaround, debugging path, or "
    "tool-usage pattern emerged that a future session would benefit "
    "from. Capture it.\n"
    "  • A skill that got loaded or consulted this session turned out "
    "to be wrong, missing a step, or outdated. Patch it NOW.\n\n"
    "Preference order — prefer the earliest action that fits, but do "
    "pick one when a signal above fired:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. Look back through the "
    "conversation for skills the user loaded via /skill-name or you "
    "read via skill_view. If any of them covers the territory of the "
    "new learning, PATCH that one first. It is the skill that was in "
    "play, so it's the right one to extend — but only if it is "
    "curator-managed. Bundled, hub, pinned, and user-owned skills are "
    "off-limits to you no matter how relevant (see Protected skills "
    "below); for those, fall through to the next option.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (via skills_list + skill_view). "
    "If no loaded skill fits but an existing class-level skill does, "
    "patch it. Add a subsection, a pitfall, or broaden a trigger.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella. Skills can be "
    "packaged with three kinds of support files — use the right "
    "directory per kind:\n"
    "     • `references/<topic>.md` — session-specific detail (error "
    "transcripts, reproduction recipes, provider quirks) AND "
    "condensed knowledge banks: quoted research, API docs, external "
    "authoritative excerpts, or domain notes you found while working "
    "on the problem. Write it concise and for the value of the task, "
    "not as a full mirror of upstream docs.\n"
    "     • `templates/<name>.<ext>` — starter files meant to be "
    "copied and modified (boilerplate configs, scaffolding, a "
    "known-good example the agent can `reproduce with modifications`).\n"
    "     • `scripts/<name>.<ext>` — statically re-runnable actions "
    "the skill can invoke directly (verification scripts, fixture "
    "generators, deterministic probes, anything the agent should run "
    "rather than hand-type each time).\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing fits. Name at "
    "the class level — NOT a PR number, error string, codename, "
    "library-alone name, or 'fix-X / debug-Y' session artifact. If the "
    "name only fits today's task, fall back to (1), (2), or (3).\n\n"
    "Protected skills (DO NOT edit these):\n"
    "  • Bundled skills (shipped with Hermes).\n"
    "  • Hub-installed skills.\n"
    "  • PINNED skills (marked via curator pin). You are an "
    "autonomous no-user-present actor, so pin blocks your writes too.\n"
    "  • USER-OWNED skills — anything not curator-managed. A skill the "
    "user hand-wrote, installed by URL, or asked a foreground agent to "
    "create is theirs, not yours; your writes to it WILL be refused. "
    "If such a skill is wrong or outdated, say so in your reply and "
    "recommend the user adopt it — do not try to patch it.\n"
    "If the only skills that need updating are protected, say "
    "'Nothing to save.' and stop.\n\n"
    "Do NOT capture:\n"
    "  • Environment-dependent failures: missing binaries, fresh-install "
    "errors, 'command not found', unconfigured credentials, uninstalled "
    "packages. The user can fix these — they are not durable rules.\n"
    "  • Negative claims about tools or features ('X tool is broken', "
    "'cannot use Y'). These harden into refusals the agent cites "
    "against itself for months after the actual problem was fixed.\n"
    "  • Session-specific transient errors that resolved before the "
    "conversation ended.\n"
    "  • One-off task narratives. A user asking 'summarize today's "
    "market' or 'analyze this PR' is not a class of work that warrants "
    "a skill.\n"
    "  • Unresolved failures: if the session ended WITHOUT actually "
    "finding a working method, do NOT write those attempts up as a "
    "'reliable workflow'. Either say 'Nothing to save', or capture "
    "ONLY a real working alternative.\n\n"
    "If a tool failed because of setup state, capture the FIX (install "
    "command, config step, env var to set) under an existing setup or "
    "troubleshooting skill — never 'this tool does not work' as a "
    "standalone constraint.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. If the session ran smoothly with no corrections and "
    "produced no new technique, just say 'Nothing to save.' and stop. "
    "Otherwise, act."
)

_COMBINED_REVIEW_PROMPT = (
    "Review the conversation above and update two things:\n\n"
    "**Memory**: who the user is. Did the user reveal persona, "
    "desires, preferences, personal details, or expectations about "
    "how you should behave? Save facts about the user and durable "
    "preferences with the memory tool.\n\n"
    "**Skills**: how to do this class of task. Be ACTIVE — most "
    "sessions produce at least one skill update. A pass that does "
    "nothing is a missed learning opportunity, not a neutral outcome.\n\n"
    "Target shape of the skill library: CLASS-LEVEL skills with a rich "
    "SKILL.md and a `references/` directory for session-specific detail. "
    "Not a long flat list of narrow one-session-one-skill entries.\n\n"
    "Signals that warrant a skill update (any one is enough):\n"
    "  • User corrected your style, tone, format, legibility, "
    "verbosity, or approach. Frustration is a FIRST-CLASS skill "
    "signal, not just a memory signal. 'stop doing X', 'don't format "
    "like this', 'I hate when you Y' — embed the lesson in the skill "
    "that governs that task so the next session starts fixed.\n"
    "  • Non-trivial technique, fix, workaround, or debugging path "
    "emerged.\n"
    "  • A skill that was loaded or consulted turned out wrong, "
    "missing, or outdated — patch it now.\n\n"
    "Preference order for skills — pick the earliest that fits:\n"
    "  1. UPDATE A CURRENTLY-LOADED SKILL. If one of the skills loaded "
    "this session covers the learning, PATCH it first — provided it is "
    "curator-managed. Protected and user-owned skills are off-limits.\n"
    "  2. UPDATE AN EXISTING UMBRELLA (skills_list + skill_view to "
    "find the right one). Patch it.\n"
    "  3. ADD A SUPPORT FILE under an existing umbrella via "
    "skill_manage action=write_file (`references/`, `templates/`, "
    "`scripts/`). Add a one-line pointer in SKILL.md so future agents "
    "find them.\n"
    "  4. CREATE A NEW CLASS-LEVEL UMBRELLA when nothing exists. "
    "Name at the class level — NOT a PR number, error string, "
    "codename, library-alone name, or 'fix-X / debug-Y' session "
    "artifact.\n\n"
    "User-preference embedding: when the user complains about how "
    "you handled a task, update the skill that governs that task — "
    "memory alone isn't enough. Memory says 'who the user is'; "
    "skills say 'how to do this class of task for this user'. Both "
    "should carry user-preference lessons when relevant.\n\n"
    "If you notice overlapping existing skills, mention it.\n\n"
    "Protected skills (DO NOT edit these): bundled / hub-installed / "
    "pinned / user-owned skills. If the only skills that need updating "
    "are protected, say 'Nothing to save.' and stop.\n\n"
    "Do NOT capture environment-dependent failures, negative claims "
    "about tools, transient errors, one-off task narratives, or "
    "unresolved failures.\n\n"
    "'Nothing to save.' is a real option but should NOT be the "
    "default. Otherwise, act."
)


# ── Review fork 执行 ───────────────────────────────────────────────

def _fork_review_agent(agent: Any, messages_snapshot: List[Dict], prompt: str) -> None:
    """Fork 一个受限 review agent，回放对话并执行 review（对齐原版
    _run_review_in_thread 的核心；无 routed/白名单/遥测等 my-hermes
    裁剪项）。"""
    from run_agent import AIAgent

    review_agent = None
    try:
        # 继承父 agent 的实时运行时（provider/model/base_url/api_key）
        review_agent = AIAgent(
            model=getattr(agent, "model", "") or "",
            max_iterations=16,
            quiet_mode=True,
            platform=getattr(agent, "platform", None),
            provider=getattr(agent, "provider", None),
            api_mode=getattr(agent, "api_mode", None),
            base_url=getattr(agent, "base_url", None),
            api_key=getattr(agent, "api_key", None),
            max_tokens=getattr(agent, "max_tokens", None),
            parent_session_id=getattr(agent, "session_id", None),
            # 工具白名单：只允许 memory + skills（原版线程白名单的等价简化）
            enabled_toolsets=["skills"] + (
                ["memory"] if (
                    getattr(agent, "_memory_enabled", False)
                    or getattr(agent, "_user_profile_enabled", False)
                ) else []
            ),
            skip_memory=True,  # fork 不重建外部 provider；内置记忆共享（见下）
            session_id=getattr(agent, "session_id", None),
        )
        # 共享内置记忆（review 写 MEMORY.md 仍落盘，但零外部副作用）
        review_agent._memory_store = getattr(agent, "_memory_store", None)
        review_agent._memory_enabled = getattr(agent, "_memory_enabled", False)
        review_agent._user_profile_enabled = getattr(
            agent, "_user_profile_enabled", False
        )
        # fork 自身不再触发 review
        review_agent._memory_nudge_interval = 0
        review_agent._skill_nudge_interval = 0
        # 持久化隔离：harness 轮绝不写进用户真实会话（state.db）
        review_agent._persist_disabled = True
        review_agent._session_db = None
        # 共享父的 warm 缓存前缀（同模型同 tools → 命中同一 provider 缓存）
        review_agent._cached_system_prompt = getattr(agent, "_cached_system_prompt", None)
        review_agent.session_start = getattr(agent, "session_start", None)
        # fork 单生命周期：close() 不终结父会话行、不压缩
        review_agent._end_session_on_close = False
        review_agent.compression_enabled = False

        # 回放完整 snapshot（同模型 warm cache 读，等价原版非 routed 路径）
        review_agent.run_conversation(
            user_message=(
                prompt
                + "\n\nYou can only call memory and skill "
                "management tools. Other tools will be denied "
                "at runtime — do not attempt them."
            ),
            conversation_history=list(messages_snapshot),
        )
    except Exception as exc:
        logger.debug("background review fork failed: %s", exc, exc_info=True)
    finally:
        if review_agent is not None:
            try:
                review_agent.close()
            except Exception:
                pass


def spawn_background_review_thread(
    agent: Any,
    messages_snapshot: List[Dict],
    review_memory: bool = False,
    review_skills: bool = False,
    focus: Optional[str] = None,
) -> "tuple[Callable[[], None], str]":
    """构建 review 线程 target 与 prompt（对齐原版 spawn_background_review_thread）。

    ``focus`` 为可选用户指定方向（原版 /refine 路径）；自动触发不设。
    返回 ``(target, prompt)``，线程由调用方（run_agent._spawn_background_review）
    负责启动。
    """
    if review_memory and review_skills:
        prompt = _COMBINED_REVIEW_PROMPT
    elif review_memory:
        prompt = _MEMORY_REVIEW_PROMPT
    else:
        prompt = _SKILL_REVIEW_PROMPT

    focus = (focus or "").strip()
    if focus:
        prompt = (
            f"{prompt}\n\n"
            f"The user explicitly requested this review with the following "
            f"focus — prioritize it over the general instructions above:\n"
            f"{focus}"
        )

    def _target() -> None:
        # 标记写来源：本线程内 skill_manage 的写入视为 agent 自主沉淀
        # （curator 只整理这些；用户写的一律不动——对齐原版 provenance）。
        try:
            from tools.skill_provenance import set_write_origin
            set_write_origin("background_review")
        except Exception:
            pass
        _fork_review_agent(agent, messages_snapshot, prompt)

    return _target, prompt


__all__ = [
    "_MEMORY_REVIEW_PROMPT",
    "_SKILL_REVIEW_PROMPT",
    "_COMBINED_REVIEW_PROMPT",
    "spawn_background_review_thread",
]
