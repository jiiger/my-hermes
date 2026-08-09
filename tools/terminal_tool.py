"""终端工具（精简移植版）。

对应原版 hermes-agent 的 tools/terminal_tool.py（3419 行）。原版依赖
大量环境机制（docker/modal/ssh/singularity 后端、进程注册表、审批、
sudo、watchdog 等），my-hermes 用不上，精简版砍到最小实现：
「subprocess.run + timeout + 输出截断」。

砍掉：
- 多执行后端（local/docker/modal/ssh/...）：只跑本地 shell；
- 后台执行（background）：process_registry / 会话管理，返回明确错误；
- PTY / 审批 / sudo / 看门狗 / 环境探测 / 磁盘告警。

保留：schema 与描述照抄原版（TERMINAL_SCHEMA /
TERMINAL_TOOL_DESCRIPTION），返回结构与原版一致（output/exit_code/error）。
handler 签名与 schema 参数一致 + **kwargs 吸收多余参数。
"""

import json
import os
import subprocess
import time
from typing import List, Optional

from tools.registry import tool_error

# 前台超时上限（对应原版 terminal_tool.py:118 FOREGROUND_MAX_TIMEOUT）
FOREGROUND_MAX_TIMEOUT = 600
# 默认超时（对应原版 schema 描述里的 180）
_DEFAULT_TIMEOUT = 180
# 输出截断上限（对齐注册时的 max_result_size_chars=100_000）
_MAX_OUTPUT_CHARS = 100_000


def _truncate_output(text: str) -> str:
    """截断超长输出到 _MAX_OUTPUT_CHARS（保留头尾可读性）。"""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    keep = _MAX_OUTPUT_CHARS // 2
    return (
        text[:keep]
        + f"\n… [output truncated: {len(text):,} chars total] …\n"
        + text[-keep:]
    )


def terminal(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    **kwargs,
) -> str:
    """在本地 shell 执行命令（对应原版 terminal_tool.py:2236 terminal_tool）。

    精简版只支持前台执行：subprocess.run(shell=True) + timeout +
    输出截断。background / pty 等原版能力返回明确错误提示。
    schema 参数：command / background / timeout / workdir / pty /
    notify_on_complete / watch_patterns。
    """
    del task_id, session_id, force, notify_on_complete, watch_patterns, kwargs
    try:
        if not isinstance(command, str):
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": (
                    f"Invalid command: expected string, got {type(command).__name__}"
                ),
                "status": "error",
            }, ensure_ascii=False)
        if not command.strip():
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": "Invalid command: empty string",
                "status": "error",
            }, ensure_ascii=False)

        # 精简版不支持后台执行：明确报错，避免静默丢进程
        if background:
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": (
                    "background execution is not supported in this lite version; "
                    "run the command in the foreground with a generous timeout."
                ),
                "status": "error",
            }, ensure_ascii=False)
        # 精简版不支持 PTY：明确报错，避免交互命令挂死
        if pty:
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": (
                    "pty mode is not supported in this lite version; "
                    "pipe stdin or avoid interactive commands."
                ),
                "status": "error",
            }, ensure_ascii=False)

        # 超时校验：非正数拒绝（对应原版 :2309）
        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )
        effective_timeout = timeout or _DEFAULT_TIMEOUT
        if effective_timeout > FOREGROUND_MAX_TIMEOUT:
            return tool_error(
                f"Foreground timeout {effective_timeout}s exceeds the maximum "
                f"of {FOREGROUND_MAX_TIMEOUT}s."
            )

        # workdir 校验：必须是存在的目录
        cwd = None
        if workdir:
            if not os.path.isdir(workdir):
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": f"Working directory does not exist: {workdir}",
                    "status": "error",
                }, ensure_ascii=False)
            cwd = workdir

        started = time.monotonic()
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                timeout=effective_timeout,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except subprocess.TimeoutExpired as exc:
            # 超时：把已捕获的部分输出带回来，帮助模型判断进度
            partial = (exc.stdout or "") + (exc.stderr or "")
            return json.dumps({
                "output": _truncate_output(partial),
                "exit_code": -1,
                "error": (
                    f"Command timed out after {effective_timeout}s. "
                    "Use a larger timeout or a more targeted command."
                ),
                "status": "error",
            }, ensure_ascii=False)

        elapsed = time.monotonic() - started
        output = (result.stdout or "") + (result.stderr or "")
        payload = {
            "output": _truncate_output(output),
            "exit_code": result.returncode,
            "error": None if result.returncode == 0 else (
                f"Command exited with code {result.returncode}"
            ),
            "duration_ms": int(elapsed * 1000),
        }
        if cwd:
            payload["cwd"] = cwd
        if result.returncode != 0:
            payload["status"] = "error"
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return tool_error(str(exc))


def check_terminal_requirements() -> bool:
    """终端工具在本地环境恒可用（对应原版 :3186 的 local 分支）。"""
    return True


# =============================================================================
# Schema（对应原版 terminal_tool.py:3352 TERMINAL_SCHEMA；描述照抄）
# =============================================================================

TERMINAL_TOOL_DESCRIPTION = """Execute shell commands on a Linux environment. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers, and anything that needs a shell.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id). Pair with notify_on_complete=true for bounded tasks; leave silent only for servers/daemons that never exit. Never use nohup/setsid/trailing '&' — use background=true so Hermes tracks the process. After starting a server, verify readiness with a health check, then act in a separate call; no blind sleep loops. Manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd. When a command changes the session cwd (cd, pushd), the result includes a "cwd" field — trust it instead of prefixing every command with 'cd'.
PTY: set pty=true for interactive CLIs (they hang without it). Pipe git output to cat if it might page.
"""

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to execute on the VM"
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Run in the background, returning a session_id. Pair with "
                    "notify_on_complete=true for anything with a defined end "
                    "(tests, builds, deploys) — without it the process runs "
                    "silently. Only servers/watchers/daemons that never exit "
                    "should stay silent. Short commands: prefer foreground with "
                    "a generous timeout."
                ),
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": (
                    f"Max seconds to wait (default: 180, foreground max: "
                    f"{FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command "
                    f"finishes — set high for long tasks, you won't wait "
                    f"unnecessarily. Foreground timeout above "
                    f"{FOREGROUND_MAX_TIMEOUT}s is rejected; use background=true "
                    f"for longer commands."
                ),
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Working directory for this command (absolute path). "
                    "Defaults to the session working directory."
                )
            },
            "pty": {
                "type": "boolean",
                "description": (
                    "Run in pseudo-terminal (PTY) mode for interactive CLI "
                    "tools like Codex, Claude Code, or Python REPL. Only works "
                    "with local and SSH backends. Default: false."
                ),
                "default": False
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": (
                    "With background=true: get exactly one notification when "
                    "the process exits. The right choice for nearly every "
                    "bounded long task — set it and keep working. MUTUALLY "
                    "EXCLUSIVE with watch_patterns (watch_patterns is dropped "
                    "when both are set)."
                ),
                "default": False
            },
            "watch_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Strings to watch for in background output. ONLY for rare "
                    "one-shot mid-process signals on processes that never exit "
                    "(e.g. ['Application startup complete'] on a server). NOT "
                    "for end-of-run markers (use notify_on_complete) and NOT "
                    "for per-iteration patterns like 'ERROR' in loops — "
                    "rate-limited to 1 notification/15s; repeated over-firing "
                    "auto-disables it and falls back to notify-on-exit. When in "
                    "doubt, use notify_on_complete. MUTUALLY EXCLUSIVE with "
                    "notify_on_complete."
                )
            }
        },
        "required": ["command"]
    }
}


# --- 模块级自注册（对应原版 terminal_tool.py:3411）---
from tools.registry import registry

registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)
