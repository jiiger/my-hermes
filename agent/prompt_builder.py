import contextvars
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


from agent.runtime_cwd import resolve_agent_cwd
from hermes_constants import is_wsl
from tools.threat_patterns import scan_for_threats as _scan_for_threats


def _scan_context_content(content: str, filename: str):
    if content.startswith("\ufeff"):
        content = content[1:]

    findings = _scan_for_threats(content, scope="context")
    if findings:
        logger.warning("Context file %s blocked: %s", filename, ", ".join(findings))
        return f"[BLOCKED: {filename} contained potential prompt injection ({', '.join(findings)}). Content not loaded.]"

    return content


def _find_git_root(start: Path) -> Optional[Path]:

    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return None


_HERMES_MD_NAMES = (".hermes.md", "HERMES.md")


def _find_hermes_md(cwd: Path) -> Optional[Path]:
    stop_at = _find_git_root(cwd)
    current = cwd.resolve()

    search_dirs = [current, *current.parents] if stop_at else [current]

    for directory in search_dirs:
        for name in _HERMES_MD_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        if stop_at and directory == stop_at:
            break
    return None


def _strip_yaml_frontmatter(content: str) -> str:
    """Remove optional YAML frontmatter (``---`` delimited) from *content*.

    The frontmatter may contain structured config (model overrides, tool
    settings) that will be handled separately in a future PR.  For now we
    strip it so only the human-readable markdown body is injected into the
    system prompt.
    """
    content = content.lstrip("\ufeff")  # tolerate UTF-8 BOM (Windows editors)
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            # Skip past the closing --- and any trailing newline
            body = content[end + 4 :].lstrip("\n")
            return body if body else content
    return content


# =========================================================================
# Constants
# =========================================================================

DEFAULT_AGENT_IDENTITY = (
    "You are Hermes Agent, an intelligent AI assistant created by Nous Research. "
    "You are helpful, knowledgeable, and direct. You assist users with a wide "
    "range of tasks including answering questions, writing and editing code, "
    "analyzing information, creative work, and executing actions via your tools. "
    "You communicate clearly, admit uncertainty when appropriate, and prioritize "
    "being genuinely useful over being verbose unless otherwise directed below. "
    "Be targeted and efficient in your exploration and investigations."
)

HERMES_AGENT_HELP_GUIDANCE = (
    "You run on Hermes Agent (by Nous Research). When the user needs help with "
    "Hermes itself — configuring, setting up, using, extending, or troubleshooting "
    "it — or when you need to understand your own features, tools, or capabilities, "
    "the documentation at https://hermes-agent.nousresearch.com/docs is your "
    "authoritative reference and always holds the latest, most up-to-date "
    "information. Load the `hermes-agent` skill with skill_view(name='hermes-agent') "
    "for additional guidance and proven workflows, but treat the docs as the source "
    "of truth when the two differ."
)

MEMORY_GUIDANCE = (
    "You have persistent memory across sessions. Save durable facts using the memory "
    "tool: user preferences, environment details, tool quirks, and stable conventions. "
    "Memory is injected into every turn, so keep it compact and focused on facts that "
    "will still matter later.\n"
    "Prioritize what reduces future user steering — the most valuable memory is one "
    "that prevents the user from having to correct or remind you again. "
    "User preferences and recurring corrections matter more than procedural task details.\n"
    "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
    "state to memory; use session_search to recall those from past transcripts. "
    "Specifically: do not record PR numbers, issue numbers, commit SHAs, 'fixed bug X', "
    "'submitted PR Y', 'Phase N done', file counts, or any artifact that will be stale "
    "in 7 days. If a fact will be stale in a week, it does not belong in memory. "
    "If you've discovered a new way to do something, solved a problem that could be "
    "necessary later, save it as a skill with the skill tool.\n"
    "Write memories as declarative facts, not instructions to yourself. "
    "'User prefers concise responses' ✓ — 'Always respond concisely' ✗. "
    "'Project uses pytest with xdist' ✓ — 'Run tests with pytest -n 4' ✗. "
    "Imperative phrasing gets re-read as a directive in later sessions and can "
    "cause repeated work or override the user's current request. Procedures and "
    "workflows belong in skills, not memory."
)

SESSION_SEARCH_GUIDANCE = (
    "When the user references something from a past conversation or you suspect "
    "relevant cross-session context exists, use session_search to recall it before "
    "asking them to repeat themselves."
)

SKILLS_GUIDANCE = (
    "After completing a complex task (5+ tool calls), fixing a tricky error, "
    "or discovering a non-trivial workflow, save the approach as a "
    "skill with skill_manage so you can reuse it next time.\n"
    "When using a skill and finding it outdated, incomplete, or wrong, "
    "patch it immediately with skill_manage(action='patch') — don't wait to be asked. "
    "Skills that aren't maintained become liabilities.\n"
    "\n"
    "## Skill Safety Rule\n"
    "1. **UNAVAILABLE** — If a skill placeholder contains `[SKILL_PRUNED]`, the skill content was lost in compression and is inaccessible.\n"
    "2. **RELOAD** — Before performing any action that depends on a skill, re-check its content with `skill_view(name='...')` if it shows `[SKILL_PRUNED]`.\n"
    "3. **WAIT** — If a skill is loading or was just pruned, wait for the reload confirmation before proceeding.\n"
    "4. **DEDUP** — After reloading a pruned skill, **ignore any remaining `[SKILL_PRUNED]` markers for that same skill** — they are historical artifacts from previous compactions and do not need further action."
)

KANBAN_GUIDANCE = (
    "# Kanban task execution protocol\n"
    "You have been assigned ONE task from "
    "the shared board at `~/.hermes/kanban.db`. Your task id is in "
    "`$HERMES_KANBAN_TASK`; your workspace is `$HERMES_KANBAN_WORKSPACE`. "
    "The `kanban_*` tools in your schema are your primary coordination surface — "
    "they write directly to the shared SQLite DB and work regardless of terminal "
    "backend (local/docker/modal/ssh).\n"
    "\n"
    "## Lifecycle\n"
    "\n"
    "1. **Orient.** Call `kanban_show()` first (no args — it defaults to your "
    "task). The response includes title, body, parent-task handoffs (summary + "
    "metadata), any prior attempts on this task if you're a retry, the full "
    "comment thread, and a pre-formatted `worker_context` you can treat as "
    "ground truth.\n"
    "2. **Work inside the workspace.** `cd $HERMES_KANBAN_WORKSPACE` before "
    "any file operations. The workspace is yours for this run. Don't modify "
    "files outside it unless the task explicitly asks.\n"
    "3. **Heartbeat on long operations.** Call `kanban_heartbeat(note=...)` "
    "every few minutes during long subprocesses (training, encoding, crawling). "
    "Skip heartbeats for short tasks. **If your task may run longer than 1 hour, "
    "you MUST call `kanban_heartbeat` at least once an hour** — the dispatcher "
    "reclaims tasks running past `kanban.dispatch_stale_timeout_seconds` "
    "(default 4 hours) when no heartbeat has arrived in the last hour. A "
    "reclaim re-queues the task as `ready` without penalty (no failure counter "
    "tick), but you lose your current run's progress.\n"
    "4. **Block on genuine ambiguity.** If you need a human decision you cannot "
    "infer (missing credentials, UX choice, paywalled source, peer output you "
    'need first), call `kanban_block(reason="...")` and stop. Don\'t guess. '
    "The user will unblock with context and the dispatcher will respawn you.\n"
    "5. **Complete with structured handoff.** Call `kanban_complete(summary=..., "
    "metadata=...)`. `summary` is 1–3 human-readable sentences naming concrete "
    "artifacts. `metadata` is machine-readable facts "
    "(`{changed_files: [...], tests_run: N, decisions: [...]}`). Downstream "
    "workers read both via their own `kanban_show`. Never put secrets / "
    "tokens / raw PII in either field — run rows are durable forever. "
    "Exception: if your output is a code change that needs human review "
    "before counting as merged/done (most coding tasks), drop the "
    "structured metadata (changed_files / tests_run / diff_path) into a "
    "`kanban_comment` first, then end with "
    '`kanban_block(reason="review-required: <one-line summary>")` so a '
    "reviewer can approve+unblock or request changes. Reviewing-then-"
    "completing is more honest than auto-completing work that still needs "
    "eyes on it.\n"
    "6. **If follow-up work appears, create it; don't do it.** Use "
    "`kanban_create(title=..., assignee=<right-profile>, parents=[your-task-id])` "
    "to spawn a child task for the appropriate specialist profile instead of "
    "scope-creeping into the next thing.\n"
    "\n"
    "## Orchestrator mode\n"
    "\n"
    "If your task is itself a decomposition task (e.g. a planner profile given "
    "a high-level goal), use `kanban_create` to fan out into child tasks — one "
    "per specialist, each with an explicit `assignee` and `parents=[...]` to "
    "express dependencies. Then `kanban_complete` your own task with a summary "
    "of the decomposition. Do NOT execute the work yourself; your job is "
    "routing, not implementation.\n"
    "\n"
    "## Reference details that change outcomes\n"
    "\n"
    "- **Workspace.** `cd $HERMES_KANBAN_WORKSPACE` first. For a `worktree` kind "
    "with no `.git`, `git worktree add <path> "
    "${HERMES_KANBAN_BRANCH:-wt/$HERMES_KANBAN_TASK}` from the main repo, then "
    "cd there. For a project-linked task the workspace is a fresh "
    "`<repo>/.worktrees/<task-id>` and `$HERMES_KANBAN_BRANCH` a deterministic "
    "`<project-slug>/<task-id>` — the main repo is two levels up, so run "
    "`git worktree add` from there.\n"
    "- **Deliverables.** Files a human wants go in "
    "`kanban_complete(artifacts=[<absolute paths>])` (top-level param; paths in "
    "`metadata` are NOT uploaded). Files must exist at completion.\n"
    "- **Attachments.** Attach real downloadable artifacts instead of pasting "
    "links in comments: `kanban_attach` (base64) or `kanban_attach_url` "
    "(server-side public http(s) fetch); 25 MB cap, `kanban_attachments` "
    "lists them. Workers may only attach to their own task.\n"
    "- **Created cards.** List ids in `kanban_complete(created_cards=[...])` "
    "ONLY when captured from a successful `kanban_create` return — never invent "
    "or paste ids; the kernel rejects the completion on any phantom id.\n"
    "- **Orchestrating: discover profiles first.** The dispatcher SILENTLY "
    "drops a card with an unknown assignee (it sits in `ready` forever). Ground "
    "every assignee in a real profile (`hermes profile list`, or ask the user), "
    "and express dependencies via `parents=[...]` on `kanban_create`, not prose.\n"
    "\n"
    "## Do NOT\n"
    "\n"
    "- Do not shell out to `hermes kanban <verb>` for board operations. Use "
    "the `kanban_*` tools — they work across all terminal backends.\n"
    "- Do not complete a task you didn't actually finish. Block it.\n"
    "- Do not call `clarify` to ask questions. You are running headless — "
    "there is no live user to answer. The call will time out and the task "
    "will sit silently in `running` with no signal to the operator. Instead: "
    "`kanban_comment` the context, then `kanban_block(reason=...)` so the "
    "task surfaces on the board as needing input.\n"
    "- Do not assign follow-up work to yourself. Assign it to the right "
    "specialist profile.\n"
    "- Do not call `delegate_task` as a board substitute. `delegate_task` is "
    "for short reasoning subtasks inside your own run; board tasks are for "
    "cross-agent handoffs that outlive one API loop."
)

TOOL_USE_ENFORCEMENT_GUIDANCE = (
    "# Tool-use enforcement\n"
    "You MUST use your tools to take action — do not describe what you would do "
    "or plan to do without actually doing it. When you say you will perform an "
    "action (e.g. 'I will run the tests', 'Let me check the file', 'I will create "
    "the project'), you MUST immediately make the corresponding tool call in the same "
    "response. Never end your turn with a promise of future action — execute it now.\n"
    "Keep working until the task is actually complete. Do not stop with a summary of "
    "what you plan to do next time. If you have tools available that can accomplish "
    "the task, use them instead of telling the user what you would do.\n"
    "Every response should either (a) contain tool calls that make progress, or "
    "(b) deliver a final result to the user. Responses that only describe intentions "
    "without acting are not acceptable."
)

# Model name substrings that trigger tool-use enforcement guidance.
# Add new patterns here when a model family needs explicit steering.
TOOL_USE_ENFORCEMENT_MODELS = (
    "gpt",
    "codex",
    "gemini",
    "gemma",
    "grok",
    "glm",
    "qwen",
    "deepseek",
)

# Universal "finish the job" guidance — applied to ALL models, not gated
# by model family.  Addresses two cross-model failure modes:
#   1. Stopping after a stub: writing a tiny file or running one command
#      and then ending the turn with a description of the plan instead
#      of the finished artifact.  (Observed on Opus during a real
#      Sarasota real-estate build task: 3 API calls, 85-byte file,
#      one terminal command, finish_reason=stop.)
#   2. Fabricating output when a real path is blocked.  When `pip` or a
#      tool fails, some models will synthesize plausible-looking results
#      (fake addresses, fake JSON, fake numbers) instead of reporting
#      the blocker.  (Observed on DeepSeek v4-flash on the same task:
#      pushed through PEP-668 wall, then returned fabricated listings.)
#
# Short on purpose.  This block is shipped to every user, every session,
# in the cached system prompt — token cost is paid once at install and
# then amortised across all sessions via prefix caching.  Keep it tight.
TASK_COMPLETION_GUIDANCE = (
    "# Finishing the job\n"
    "When the user asks you to build, run, or verify something, the deliverable is "
    "a working artifact backed by real tool output — not a description of one. "
    "Do not stop after writing a stub, a plan, or a single command. Keep working "
    "until you have actually exercised the code or produced the requested result, "
    "then report what real execution returned.\n"
    "If a tool, install, or network call fails and blocks the real path, say so "
    "directly and try an alternative (different package manager, different "
    "approach, ask the user). NEVER substitute plausible-looking fabricated "
    "output (made-up data, invented file contents, synthesised API responses) "
    "for results you couldn't actually produce. Reporting a blocker honestly "
    "is always better than inventing a result."
)

# Universal parallel-tool-call guidance — applied to ALL models.
#
# Why this matters for cost: every assistant turn resends the entire
# accumulated conversation (and, on cache-friendly providers, re-reads the
# cached prefix and pays for the newly-appended turn). A model that issues
# one tool call per turn multiplies the number of round-trips — and therefore
# the resent context — for any task that needs several independent reads,
# searches, or safe lookups. Batching independent calls into a single
# assistant response collapses N turns into one, cutting both latency and the
# resent-context cost that compounds over a long conversation.
#
# The hermes-agent runtime already executes a batch of tool calls
# concurrently when they are independent (read-only tools always; path-scoped
# file ops when their targets don't overlap — see
# run_agent._execute_tool_calls / tool_dispatch_helpers). The missing piece
# was telling the *model* to emit those calls together in the first place.
# Until now the only batching steer in the prompt lived in
# GOOGLE_MODEL_OPERATIONAL_GUIDANCE — Gemini/Gemma got it, every other model
# got nothing. This block makes the steer universal; the now-redundant
# Google-only bullet has been dropped so no model receives it twice.
#
# Short on purpose — shipped in the cached system prompt to every user, every
# session. Token cost is paid once at install and amortised across all
# sessions via prefix caching. Keep it tight.
#
# Ported from cline/cline#11514 ("encourage parallel tool calls"), adapted
# from Cline's TypeScript tool-surface guidance to hermes-agent's Python
# prompt-assembly architecture.
PARALLEL_TOOL_CALL_GUIDANCE = (
    "# Parallel tool calls\n"
    "When you need several pieces of information that don't depend on each "
    "other, request them together in a single response instead of one tool "
    "call per turn. Independent reads, searches, web fetches, and read-only "
    "commands should be batched into the same assistant turn — the runtime "
    "executes independent calls concurrently, and batching avoids resending "
    "the whole conversation on every extra round-trip.\n"
    "Only serialize calls when a later call genuinely depends on an earlier "
    "call's result (e.g. you must read a file before you can patch it). When "
    "in doubt and the calls are independent, batch them."
)
# ---------------------------------------------------------------------------
# Environment hints — execution-environment awareness for the agent.
# Unlike PLATFORM_HINTS (which describe the messaging channel), these describe
# the machine/OS the agent's tools actually run on.
# ---------------------------------------------------------------------------

WSL_ENVIRONMENT_HINT = (
    "You are running inside WSL (Windows Subsystem for Linux). "
    "The Windows host filesystem is mounted under /mnt/ — "
    "/mnt/c/ is the C: drive, /mnt/d/ is D:, etc. "
    "The user's Windows files are typically at "
    "/mnt/c/Users/<username>/Desktop/, Documents/, Downloads/, etc. "
    "When the user references Windows paths or desktop files, translate "
    "to the /mnt/c/ equivalent. You can list /mnt/c/Users/ to discover "
    "the Windows username if needed."
)
# Non-local terminal backends that run commands (and therefore every file
# tool: read_file, write_file, patch, search_files) inside a separate
# container / remote host rather than on the machine where Hermes itself
# runs. For these backends, host info (Windows/Linux/macOS, $HOME, cwd) is
# misleading — the agent should only see the machine it can actually touch.
_REMOTE_TERMINAL_BACKENDS = frozenset({
    "docker",
    "singularity",
    "modal",
    "daytona",
    "ssh",
    "vercel_sandbox",
    "managed_modal",
})


# Per-backend fallback descriptions — used when the live probe fails.
# Only states what we know from the backend choice itself (container type,
# likely OS family). Does NOT invent cwd, user, or $HOME — the agent is
# told to probe those directly if it needs them.
_BACKEND_FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "docker": "a Docker container (Linux)",
    "singularity": "a Singularity container (Linux)",
    "modal": "a Modal sandbox (Linux)",
    "managed_modal": "a managed Modal sandbox (Linux)",
    "daytona": "a Daytona workspace (Linux)",
    "vercel_sandbox": "a Vercel sandbox (Linux)",
    "ssh": "a remote host reached over SSH (likely Linux)",
}


# Cache the backend probe result per process so we only pay the probe cost
# on the first prompt build of a session. Keyed by (env_type, cwd_hint) so
# a mid-process backend switch rebuilds the string. Kept in-module (not on
# disk) because the probe captures live backend state that may change
# across Hermes restarts.
_BACKEND_PROBE_CACHE: dict[tuple[str, str], str] = {}


_WINDOWS_BASH_SHELL_HINT = (
    "Shell: on this Windows host your `terminal` tool runs commands through "
    "bash (git-bash / MSYS), NOT PowerShell or cmd.exe. Use POSIX shell "
    "syntax (`ls`, `$HOME`, `&&`, `|`, single-quoted strings) inside terminal "
    "calls. MSYS-style paths like `/c/Users/<user>/...` work alongside "
    "native `C:\\Users\\<user>\\...` paths. PowerShell builtins "
    "(`Get-ChildItem`, `$env:FOO`, `Select-String`) will NOT work — use their "
    "POSIX equivalents (`ls`, `$FOO`, `grep`)."
)


def _probe_remote_backend(env_type: str) -> str | None:
    """Run a tiny introspection command inside the active terminal backend.

    Returns a pre-formatted multi-line string describing the backend's OS,
    $HOME, cwd, and user — or None if the probe failed. Result is cached
    per process. Used only for non-local backends where the agent's tools
    operate on a different machine than the host Hermes runs on.
    """
    cwd_hint = os.getenv("TERMINAL_CWD", "")
    cache_key = (env_type, cwd_hint)
    cached = _BACKEND_PROBE_CACHE.get(cache_key)
    if cached is not None:
        return cached or None

    try:
        # Import locally: tools/ imports are heavy and only relevant when a
        # non-local backend is actually configured.
        from tools.terminal_tool import (  # type: ignore
            _create_environment,
            _get_env_config,
        )
    except Exception as e:
        logger.debug("Backend probe unavailable (import failed): %s", e)
        _BACKEND_PROBE_CACHE[cache_key] = ""
        return None

    try:
        config = _get_env_config()
        # Build the environment the same way tools/terminal_tool.py does for a
        # live command: select the backend image, then assemble ssh/container
        # config from the env-derived dict. (There is no `get_environment`
        # factory — the real entry point is `_create_environment`.)
        if env_type == "docker":
            image = config.get("docker_image", "")
        elif env_type == "singularity":
            image = config.get("singularity_image", "")
        elif env_type == "modal":
            image = config.get("modal_image", "")
        elif env_type == "daytona":
            image = config.get("daytona_image", "")
        else:
            image = ""

        ssh_config = None
        if env_type == "ssh":
            ssh_config = {
                "host": config.get("ssh_host", ""),
                "user": config.get("ssh_user", ""),
                "port": config.get("ssh_port", 22),
                "key": config.get("ssh_key", ""),
                "persistent": config.get("ssh_persistent", False),
            }

        container_config = None
        if env_type in {"docker", "singularity", "modal", "daytona", "vercel_sandbox"}:
            container_config = {
                "container_cpu": config.get("container_cpu", 1),
                "container_memory": config.get("container_memory", 5120),
                "container_disk": config.get("container_disk", 51200),
                "container_persistent": config.get("container_persistent", True),
                "modal_mode": config.get("modal_mode", "auto"),
                "docker_volumes": config.get("docker_volumes", []),
                "docker_mount_cwd_to_workspace": config.get(
                    "docker_mount_cwd_to_workspace", False
                ),
                "docker_forward_env": config.get("docker_forward_env", []),
                "docker_env": config.get("docker_env", {}),
                "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
                "docker_extra_args": config.get("docker_extra_args", []),
                "docker_shm_size": config.get("docker_shm_size", "1g"),
                "docker_persist_across_processes": config.get(
                    "docker_persist_across_processes", True
                ),
                "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
            }

        env = _create_environment(
            env_type=env_type,
            image=image,
            cwd=config.get("cwd", ""),
            timeout=config.get("timeout", 180),
            ssh_config=ssh_config,
            container_config=container_config,
            task_id="prompt-backend-probe",
            host_cwd=config.get("host_cwd"),
        )
        # Single-line POSIX probe — works on any Unixy backend. Wrapped in
        # `2>/dev/null` so a missing binary doesn't pollute the output.
        probe_cmd = (
            "printf 'os=%s\\nkernel=%s\\nhome=%s\\ncwd=%s\\nuser=%s\\n' "
            '"$(uname -s 2>/dev/null || echo unknown)" '
            '"$(uname -r 2>/dev/null || echo unknown)" '
            '"$HOME" "$(pwd)" "$(whoami 2>/dev/null || id -un 2>/dev/null || echo unknown)"'
        )
        result = env.execute(probe_cmd, timeout=4)
        if result.get("returncode") != 0:
            logger.debug("Backend probe returned non-zero: %r", result)
            _BACKEND_PROBE_CACHE[cache_key] = ""
            return None
        output = (result.get("output") or "").strip()
        if not output:
            _BACKEND_PROBE_CACHE[cache_key] = ""
            return None
    except Exception as e:
        logger.debug("Backend probe failed: %s", e)
        _BACKEND_PROBE_CACHE[cache_key] = ""
        return None

    # Parse key=value lines back into a tidy summary.
    parsed: dict[str, str] = {}
    for line in output.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parsed[k.strip()] = v.strip()

    pieces = []
    os_bits = " ".join(
        x for x in (parsed.get("os"), parsed.get("kernel")) if x and x != "unknown"
    )
    if os_bits:
        pieces.append(f"OS: {os_bits}")
    if parsed.get("user") and parsed["user"] != "unknown":
        pieces.append(f"User: {parsed['user']}")
    if parsed.get("home"):
        pieces.append(f"Home: {parsed['home']}")
    if parsed.get("cwd"):
        pieces.append(f"Working directory: {parsed['cwd']}")

    if not pieces:
        _BACKEND_PROBE_CACHE[cache_key] = ""
        return None

    formatted = "\n".join(f"  {p}" for p in pieces)
    _BACKEND_PROBE_CACHE[cache_key] = formatted
    return formatted


def build_environment_hints() -> str:
    """Return environment-specific guidance for the system prompt.

    Always emits a factual block describing the execution environment:
    - For **local** terminal backends: the host OS, user home, current
      working directory (plus a Windows-only note about hostname != user
      and a Windows-only note that `terminal` shells out to bash, not
      PowerShell).
    - For **remote / sandbox** terminal backends (docker, singularity,
      modal, daytona, ssh, vercel_sandbox): host info is **suppressed**
      because the agent's tools can't touch the host — only the backend
      matters. A live probe inside the backend reports its OS, user, $HOME,
      and cwd. Falls back to a static summary if the probe fails.

    The WSL environment hint is appended unchanged when running under WSL.
    """
    import platform
    import sys

    hints: list[str] = []

    backend = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    is_remote_backend = backend in _REMOTE_TERMINAL_BACKENDS

    if not is_remote_backend:
        # --- Host info block (local backend: host == where tools run) ---
        host_lines: list[str] = []
        if is_wsl():
            host_lines.append("Host: WSL (Windows Subsystem for Linux)")
        elif sys.platform == "win32":
            host_lines.append(f"Host: Windows ({platform.release()})")
        elif sys.platform == "darwin":
            mac_ver = platform.mac_ver()[0]
            host_lines.append(f"Host: macOS ({mac_ver or platform.release()})")
        else:
            host_lines.append(f"Host: {platform.system()} ({platform.release()})")

        host_lines.append(f"User home directory: {os.path.expanduser('~')}")
        try:
            host_lines.append(f"Current working directory: {resolve_agent_cwd()}")
        except OSError:
            pass

        if sys.platform == "win32" and not is_wsl():
            host_lines.append(
                "Note: on Windows, the machine hostname (e.g. from `hostname` "
                "or uname) is NOT the username. Use the 'User home directory' "
                "above to construct paths under C:\\Users\\<user>\\, never the "
                "hostname."
            )
        hints.append("\n".join(host_lines))

        # Windows-local terminal runs bash, not PowerShell — the model must
        # know this or it will issue PowerShell syntax and fail.
        if sys.platform == "win32" and not is_wsl():
            hints.append(_WINDOWS_BASH_SHELL_HINT)
    else:
        # --- Remote backend block (host info suppressed) ---
        probe = _probe_remote_backend(backend)
        if probe:
            hints.append(
                f"Terminal backend: {backend}. Your `terminal`, `read_file`, "
                f"`write_file`, `patch`, and `search_files` tools all operate "
                f"inside this {backend} environment — NOT on the machine "
                f"where Hermes itself is running. The host OS, home, and cwd "
                f"of the Hermes process are irrelevant; only the following "
                f"backend state matters:\n{probe}"
            )
        else:
            description = _BACKEND_FALLBACK_DESCRIPTIONS.get(
                backend, f"a {backend} environment (likely Linux)"
            )
            hints.append(
                f"Terminal backend: {backend}. Your `terminal`, `read_file`, "
                f"`write_file`, `patch`, and `search_files` tools all operate "
                f"inside {description} — NOT on the machine where Hermes "
                f"itself runs. The backend probe didn't respond at "
                f"prompt-build time, so the sandbox's current user, $HOME, "
                f"and working directory are unknown from here. If you need "
                f"them, probe directly with a terminal call like "
                f"`uname -a && whoami && pwd`."
            )

    if is_wsl():
        hints.append(WSL_ENVIRONMENT_HINT)

    # Embedder-supplied environment description. Lets a host that wraps Hermes
    # (e.g. a sandbox runner / managed platform) explain the environment the
    # agent is running in — proxy, credential handling, mount layout — without
    # forking the identity slot (SOUL.md). Read once at prompt-build time, so
    # it's part of the stable, cache-safe system prompt. The env var is the
    # build-time/embedder mechanism (set in a container ENV); config.yaml
    # ``agent.environment_hint`` is the user-facing surface. Env var wins.
    extra = (os.getenv("HERMES_ENVIRONMENT_HINT") or "").strip()
    if not extra:
        try:
            from hermes_cli.config import load_config_readonly

            extra = str(
                (load_config_readonly().get("agent", {}) or {}).get(
                    "environment_hint", ""
                )
            ).strip()
        except Exception as e:
            logger.debug("Could not read agent.environment_hint from config: %s", e)
    if extra:
        hints.append(extra)

    return "\n\n".join(hints)


CONTEXT_FILE_MAX_CHARS = 20_000
CONTEXT_TRUNCATE_HEAD_RATIO = 0.7
CONTEXT_TRUNCATE_TAIL_RATIO = 0.2

# Dynamic-cap parameters (used when no explicit context_file_max_chars is set).
# The cap scales with the model's context window so large-context models rarely
# truncate a project doc, while small-context models stay at the historical
# 20K floor. ~4 chars/token is the usual English heuristic; we spend a small
# slice of the window on context files since they share the cached prefix with
# the system prompt, tools, memory, and the whole conversation.
_CONTEXT_FILE_CHARS_PER_TOKEN = 4
_CONTEXT_FILE_WINDOW_FRACTION = 0.06
_CONTEXT_FILE_DYNAMIC_CEILING = 500_000


def _dynamic_context_file_max_chars(context_length: Optional[int]) -> int:
    """Derive a char cap from the model's context window.

    Returns at least ``CONTEXT_FILE_MAX_CHARS`` (the historical 20K floor) and
    at most ``_CONTEXT_FILE_DYNAMIC_CEILING``. When ``context_length`` is
    unknown/invalid, returns the flat default so behavior is unchanged.
    """
    if not isinstance(context_length, int) or context_length <= 0:
        return CONTEXT_FILE_MAX_CHARS
    budget = int(
        context_length * _CONTEXT_FILE_CHARS_PER_TOKEN * _CONTEXT_FILE_WINDOW_FRACTION
    )
    return max(CONTEXT_FILE_MAX_CHARS, min(budget, _CONTEXT_FILE_DYNAMIC_CEILING))


def _get_context_file_max_chars(context_length: Optional[int] = None) -> int:
    """Return the context-file truncation limit.

    Resolution order:
      1. Explicit ``context_file_max_chars`` in config.yaml — user knows best,
         always wins (including over the dynamic cap).
      2. Dynamic cap derived from the model's ``context_length`` when provided
         (scales the budget to the window; floor 20K, ceiling 500K).
      3. ``CONTEXT_FILE_MAX_CHARS`` (20K) as the upstream-compatible fallback.
    """
    try:
        from hermes_cli.config import load_config_readonly

        val = load_config_readonly().get("context_file_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    except Exception as e:
        logger.debug("Could not read context_file_max_chars from config: %s", e)
    return _dynamic_context_file_max_chars(context_length)


_truncation_warnings: "contextvars.ContextVar[Optional[list]]" = contextvars.ContextVar(
    "context_file_truncation_warnings", default=None
)


def _record_truncation_warning(msg: str) -> None:
    """Append a truncation warning to the current context's accumulator."""
    warnings = _truncation_warnings.get()
    if warnings is None:
        warnings = []
        _truncation_warnings.set(warnings)
    warnings.append(msg)


# =========================================================================
# Context files (SOUL.md, AGENTS.md, .cursorrules)
# =========================================================================


def _truncate_content(
    content: str,
    filename: str,
    max_chars: Optional[int] = None,
    context_length: Optional[int] = None,
    read_path: Optional[str] = None,
) -> str:
    """Head/tail truncation with a marker in the middle.

    ``filename`` is the human label used in warnings. ``read_path`` is the
    concrete path the agent should ``read_file`` to recover the full content
    (defaults to ``filename`` when not supplied). ``context_length`` lets the
    cap scale to the model's window when no explicit config override is set.
    """
    if max_chars is None:
        max_chars = _get_context_file_max_chars(context_length)
    if len(content) <= max_chars:
        return content
    target = read_path or filename
    msg = (
        f"⚠️  Context file {filename} TRUNCATED: "
        f"{len(content)} chars exceeds limit of {max_chars} — "
        f"trim the file, pin a larger context_file_max_chars, or use a "
        f"larger-context model!"
    )
    logger.warning(msg)
    _record_truncation_warning(msg)
    head_chars = int(max_chars * CONTEXT_TRUNCATE_HEAD_RATIO)
    tail_chars = int(max_chars * CONTEXT_TRUNCATE_TAIL_RATIO)
    head = content[:head_chars]
    tail = content[-tail_chars:]
    marker = (
        f"\n\n[...truncated {filename}: kept {head_chars}+{tail_chars} of "
        f"{len(content)} chars. The middle is omitted — if you need the full "
        f"instructions, read the complete file with the read_file tool: "
        f"{target}]\n\n"
    )
    return head + marker + tail


def _load_hermes_md(cwd_path: Path, context_length: Optional[int] = None) -> str:
    """.hermes.md / HERMES.md — walk to git root."""
    hermes_md_path = _find_hermes_md(cwd_path)
    if not hermes_md_path:
        return ""
    try:
        content = hermes_md_path.read_text(encoding="utf-8").strip()
        if not content:
            return ""
        content = _strip_yaml_frontmatter(content)
        rel = hermes_md_path.name
        try:
            rel = str(hermes_md_path.relative_to(cwd_path))
        except ValueError:
            pass
        content = _scan_context_content(content, rel)
        result = f"## {rel}\n\n{content}"
        return _truncate_content(
            result,
            ".hermes.md",
            context_length=context_length,
            read_path=str(hermes_md_path),
        )
    except Exception as e:
        logger.debug("Could not read %s: %s", hermes_md_path, e)
        return ""


def build_context_files_prompt(
    cwd: Optional[str] = None,
    skip_soul: bool = False,
    context_length: Optional[int] = None,
    allow_install_tree_fallback: bool = False,
) -> str:
    """Discover and load context files for the system prompt.

    Priority (first found wins — only ONE project context type is loaded):
      1. .hermes.md / HERMES.md  (walk to git root)
      2. AGENTS.md / agents.md   (cwd only)
      3. CLAUDE.md / claude.md   (cwd only)
      4. .cursorrules / .cursor/rules/*.mdc  (cwd only)

    SOUL.md from HERMES_HOME is independent and always included when present.

    Each context source is capped before injection. The cap defaults to the
    model's context window (scaled — see ``_dynamic_context_file_max_chars``)
    when *context_length* is provided, falling back to 20,000 chars otherwise.
    An explicit ``context_file_max_chars`` in config.yaml always wins.

    When *skip_soul* is True, SOUL.md is not included here (it was already
    loaded via ``load_soul_md()`` for the identity slot).
    """
    if cwd is None:
        cwd = os.getcwd()
        cwd_is_fallback = True
    else:
        cwd_is_fallback = False

    cwd_path = Path(cwd).resolve()
    sections = []

    # Never let a FALLBACK-picked directory inside the Hermes install/source
    # tree gain system-prompt authority. A backend that self-spawns into that
    # tree (the desktop app default) would otherwise load this repo's
    # contributor AGENTS.md as authoritative project context (#64590). An
    # explicitly configured cwd is honored verbatim — the Hermes tree is a
    # legitimate workspace when the user deliberately points a session at it —
    # and CLI-style surfaces pass allow_install_tree_fallback=True because
    # their launch dir IS the user's shell cwd (developing Hermes in-tree).
    from agent.runtime_cwd import _is_install_tree

    if (
        cwd_is_fallback
        and not allow_install_tree_fallback
        and _is_install_tree(cwd_path)
    ):
        logger.warning(
            "skipping project-context discovery: working-directory resolution "
            "fell back to the Hermes install tree (%s) — set terminal.cwd to "
            "your project directory",
            cwd_path,
        )
        project_context = ""
    else:
        # TODO 加载agent.md,claude.md,cursorrules...
        project_context = (
            _load_hermes_md(cwd_path, context_length)
            # or _load_agents_md(cwd_path, context_length)
            # or _load_claude_md(cwd_path, context_length)
            # or _load_cursorrules(cwd_path, context_length)
        )
    if project_context:
        sections.append(project_context)

    # TODO 加载soul.md

    if not sections:
        return ""
    return (
        "# Project Context\n\nThe following project context files have been loaded and should be followed:\n\n"
        + "\n".join(sections)
    )
