"""MCP 配置安全校验 —— my-hermes 移植版。

对应原版 hermes_cli/mcp_security.py（181 行）。原样保留三个高危形态的拦截：

1. 已知 IOC 黑名单（June 2026 hermes-0day 攻击方的 SSH 公钥 / 源 IP）——
   command/args/env 任意一处命中即整体拒绝（硬编码，防预植入的 config.yaml）。
2. 渗出形态（#45620）：shell 解释器（bash/sh/zsh/...）+ 内联脚本里出现
   网络外发工具（curl/wget/nc/socat、/dev/tcp、Invoke-WebRequest 等）。
3. 持久化形态（hermes-0day 后门）：shell 解释器 + 内联脚本写入 OS 持久化
   面（authorized_keys、~/.ssh、/etc/ssh、/etc/pam.d、sudoers、crontab、
   shell rc 文件）。

不是白名单：合法的本地 MCP（python 脚本、npx、uvx、自定义命令）不受影响。
校验在 spawn 前执行（tools/mcp_tool._filter_suspicious_mcp_servers）。
"""

import os
import re
import shlex
from typing import Any

_SHELL_INTERPRETERS = frozenset({
    "bash",
    "sh",
    "zsh",
    "dash",
    "fish",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
})

_EGRESS_PATTERN = re.compile(
    r"(?<![\w.-])(?:curl|wget|nc|ncat|socat)(?![\w.-])"
    r"|/dev/tcp/"
    r"|\bInvoke-WebRequest\b"
    r"|\bInvoke-RestMethod\b"
    r"|\bSystem\.Net\.WebClient\b",
    re.IGNORECASE,
)

_EXFIL_HINT_PATTERN = re.compile(
    r"\.env\b|--data-binary|--data-raw|\b-X\s+POST\b|\bPOST\b|<\s*[^\s]+",
    re.IGNORECASE,
)

# OS 持久化面：MCP server 没有正当理由写入这些路径。
_PERSISTENCE_PATTERN = re.compile(
    r"authorized_keys"
    r"|\.ssh/"
    r"|/etc/ssh\b"
    r"|/etc/pam\.d\b|pam_[\w-]+\.so"
    r"|/etc/sudoers"
    r"|/etc/cron|crontab\b"
    r"|/etc/rc\.local|/etc/systemd"
    r"|\.bashrc\b|\.bash_profile\b|\.profile\b|\.zshrc\b",
    re.IGNORECASE,
)

# ── June 2026 hermes-0day 攻击方的 IOC（硬编码，防预植入配置）──────────────
_IOC_SUBSTRINGS = (
    "AAAAC3NzaC1lZDI1NTE5AAAAICBoh1oDC4DnsO1m5mJ4yfEKrQebaFh",  # 攻击方 SSH 公钥
    "hermes-0day",
    "60.165.167.",
    "118.182.244.156",
    "61.178.123.196",
)


def _command_basename(command: Any) -> str:
    """取 command 的可执行文件名（小写），失败时返回空串。"""
    text = str(command or "").strip()
    if not text:
        return ""
    try:
        parts = shlex.split(text, posix=(os.name != "nt"))
    except ValueError:
        parts = text.split()
    first = parts[0] if parts else text
    return os.path.basename(first).lower()


def _inline_script(args: Any) -> str:
    """把 args 展平成内联脚本字符串（list/tuple 拼接，其他转 str）。"""
    if args is None:
        return ""
    if isinstance(args, (list, tuple)):
        return " ".join(str(item) for item in args)
    return str(args)


def _entry_text(entry: dict[str, Any]) -> str:
    """把 command + args + env 值展平成单个字符串，供 IOC 扫描。"""
    parts: list[str] = [str(entry.get("command") or "")]
    parts.append(_inline_script(entry.get("args")))
    env = entry.get("env")
    if isinstance(env, dict):
        parts.extend(str(v) for v in env.values())
    return " ".join(parts)


def validate_mcp_server_entry(name: str, entry: dict[str, Any]) -> list[str]:
    """返回 MCP server 配置项的安全告警列表（空 = 不可疑）。

    只拦三个窄形态（见模块 docstring）：IOC 黑名单、shell 解释器 +
    网络外发、shell 解释器 + 持久化写入。合法自定义命令不受影响。
    """
    if not isinstance(entry, dict):
        return []

    issues: list[str] = []

    # 1. 硬编码 IOC 黑名单：与命令形态无关，命中一个即拒绝（不泄漏全表）。
    flat = _entry_text(entry)
    for ioc in _IOC_SUBSTRINGS:
        if ioc in flat:
            issues.append(
                f"MCP server '{name}' contains a known hermes-0day "
                f"indicator-of-compromise ('{ioc}')"
            )
            return issues

    command = entry.get("command")
    basename = _command_basename(command)
    if basename not in _SHELL_INTERPRETERS:
        return issues

    script = _inline_script(entry.get("args"))
    if not script:
        return issues

    # 2. 网络渗出形态（#45620）。
    if _EGRESS_PATTERN.search(script):
        issue = (
            f"MCP server '{name}' uses shell interpreter '{command}' with "
            f"network egress in args"
        )
        if _EXFIL_HINT_PATTERN.search(script):
            issue += " and exfiltration-shaped arguments"
        issues.append(issue)

    # 3. OS 持久化形态（SSH 密钥 / PAM / sudoers / cron / rc 文件）。
    if _PERSISTENCE_PATTERN.search(script):
        issues.append(
            f"MCP server '{name}' uses shell interpreter '{command}' to write "
            f"to an OS persistence surface (SSH keys / PAM / sudoers / cron / "
            f"shell rc) — this is the hermes-0day backdoor shape, not a real "
            f"MCP server"
        )

    return issues


def is_mcp_server_entry_suspicious(name: str, entry: dict[str, Any]) -> bool:
    """便捷布尔包装：配置项是否可疑。"""
    return bool(validate_mcp_server_entry(name, entry))
