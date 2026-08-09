"""上下文窗口安全扫描的共享威胁模式库（精简移植版）。

对应原版 hermes-agent 的 tools/threat_patterns.py（284 行），照抄原版，
纯标准库无外部依赖。

本模块是 prompt 注入 / promptware / 数据外泄模式集的唯一事实来源，
供上下文组装扫描器（原版 agent/prompt_builder.py、tools/memory_tool.py）
与工具结果分隔符系统（agent/tool_dispatch_helpers.py）使用。

模式按**攻击类别**组织，而不是按来源文件。每个模式是一个
``(regex, pattern_id, scope)`` 三元组，``scope`` 控制哪些扫描器使用它：

- ``"all"`` —— 到处应用（经典 prompt 注入、外泄）
- ``"context"`` —— 应用到上下文文件 + 记忆 + 工具结果
  （promptware / C2 / 行为劫持；检测面更广）
- ``"strict"`` —— 仅应用到记忆写入 + 技能安装
  （对用户精选内容可激进，但对工具结果太吵）

模式锚定在 **C2 特定词汇或明确攻击行为** 上，而不是命令式英文
（"you are obligated to" 之类在合法指令写作中太常见，不能作为标志）。

多词绕过防护：模式在关键 token 之间使用有界的
``(?:\\w+\\s+){0,8}`` 填充，防止攻击者插入几个词（如 "ignore all prior
instructions"）绕过检测，同时避免无界正则回溯。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

# 用正则扫描文本的硬上限。上下文/工具结果字符串可以任意大，而扫描器是
# 咨询性守卫而非档案检索；限制输入让最坏情况运行时可预测，同时保留对
# 注入内容开头附近的检测。
MAX_SCAN_CHARS = 65_536

# 关键攻击词之间的有界填充。早期模式用 ``(?:\w+\s+)*``，有歧义且会在
# 对抗性近似命中上大量回溯。8 个填充词足以覆盖预期混淆绕过而不引入
# 无界重复。
_FILLER = r"(?:\w+\s+){0,8}"

# 每个条目：(regex, pattern_id, scope)
# scope ∈ {"all", "context", "strict"}
_PATTERNS: List[Tuple[str, str, str]] = [
    # ── 经典 prompt 注入（到处应用）────────────────────────────────
    (rf'ignore\s+{_FILLER}(previous|all|above|prior)\s+{_FILLER}instructions', "prompt_injection", "all"),
    (r'system\s+prompt\s+override', "sys_prompt_override", "all"),
    (rf'disregard\s+{_FILLER}(your|all|any)\s+{_FILLER}(instructions|rules|guidelines)', "disregard_rules", "all"),
    (rf'act\s+as\s+(if|though)\s+{_FILLER}you\s+{_FILLER}(have\s+no|don\'t\s+have)\s+{_FILLER}(restrictions|limits|rules)', "bypass_restrictions", "all"),
    (r'<!--[^>]{0,512}(?:ignore|override|system|secret|hidden)[^>]{0,512}-->', "html_comment_injection", "all"),
    (r'<\s*div\s+style\s*=\s*["\'][^>]{0,2048}display\s*:\s*none', "hidden_div", "all"),
    (r'translate\s+[^\n]{0,512}\s+into\s+[^\n]{0,512}\s+and\s+(execute|run|eval)', "translate_execute", "all"),
    (rf'do\s+not\s+{_FILLER}tell\s+{_FILLER}the\s+user', "deception_hide", "all"),

    # ── 角色扮演 / 身份劫持（context + strict；抓取的网页内容和被污染
    #    的上下文文件常见攻击面）────────────────────────────────────
    (rf'you\s+are\s+{_FILLER}now\s+(?:a|an|the)\s+', "role_hijack", "context"),
    (rf'pretend\s+{_FILLER}(you\s+are|to\s+be)\s+', "role_pretend", "context"),
    (rf'output\s+{_FILLER}(system|initial)\s+prompt', "leak_system_prompt", "context"),
    (rf'(respond|answer|reply)\s+without\s+{_FILLER}(restrictions|limitations|filters|safety)', "remove_filters", "context"),
    (rf'you\s+have\s+been\s+{_FILLER}(updated|upgraded|patched)\s+to', "fake_update", "context"),
    # "name yourself X" 是 Brainworm 特有的信号 —— 通过规范做身份覆盖
    # 而非越狱。锚定动词对，避免误匹配 "name your variables" 等。
    (r'\bname\s+yourself\s+\w+', "identity_override", "context"),

    # ── C2 / Brainworm 风格 promptware（context scope）──────────────
    # 锚定在 C2 特定词汇上。"register as a node" 出现在合法分布式系统
    # 文档里，但与其他模式组合时信号很强；我们 WARN 而非阻断，所以安全
    # 研究员在网页里读 Brainworm 帖子不会打断会话。
    (r'register\s+(as\s+)?a?\s*node', "c2_node_registration", "context"),
    (r'(heartbeat|beacon|check[\s\-]?in)\s+(to|with)\s+', "c2_heartbeat", "context"),
    (r'pull\s+(down\s+)?(?:new\s+)?task(?:ing|s)?\b', "c2_task_pull", "context"),
    (r'connect\s+to\s+the\s+network\b', "c2_network_connect", "context"),
    # 动词锚定的 "you must register/connect/report/beacon" —— 这些动词是
    # C2 特有的，避免更宽泛 "you must X" 的误报。
    (r'you\s+must\s+(?:\w+\s+){0,3}(register|connect|report|beacon)\b', "forced_action", "context"),
    # 反取证指令（"never write to disk"、"one-liners only"）—— 合法内容
    # 中极罕见；几乎零误报。
    (r'only\s+use\s+one[\s\-]?liners?\b', "anti_forensic_oneliner", "context"),
    (rf'never\s+{_FILLER}(?:create|write)\s+{_FILLER}(?:script|file)\s+{_FILLER}disk', "anti_forensic_disk", "context"),
    # 针对已知 agent 运行时取消环境变量 —— 纯攻击行为
    # （Brainworm 子会话绕过）。
    (r'unset\s+\w*(?:CLAUDE|CODEX|HERMES|AGENT|OPENAI|ANTHROPIC)\w*', "env_var_unset_agent", "context"),

    # ── 已知 C2 / 红队框架名（安全研究外几乎零误报；默认仅警告）───
    # 注意：不要在这里加普通英文词。每个 token 必须是独特的攻防安全
    # 工具品牌，否则合法 AGENTS.md / SOUL.md 内容会误报，整个文件被阻断。
    # "praxis" 正是因此被移除 —— 它是普通词（希腊语"实践/行动"），
    # 也是合法 agent 名，不是下面这些品牌那样的 C2 特有信号。
    (r'\b(?:cobalt\s*strike|sliver|havoc|mythic|metasploit|brainworm)\b', "known_c2_framework", "context"),
    (r'\bc2\s+(?:server|channel|infrastructure|beacon)\b', "c2_explicit", "context"),
    (r'\bcommand\s+and\s+control\b', "c2_explicit_long", "context"),

    # ── 通过 curl/wget/cat 带密钥外泄（到处应用）──────────────────
    (r'curl\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl", "all"),
    (r'wget\s+[^\n]{0,2048}\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget", "all"),
    (r'cat\s+[^\n]{0,2048}(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets", "all"),
    (r'(send|post|upload|transmit)\s+[^\n]{0,2048}\s+(to|at)\s+https?://', "send_to_url", "strict"),
    (rf'(include|output|print|share)\s+{_FILLER}(conversation|chat\s+history|previous\s+messages|full\s+context|entire\s+context)', "context_exfil", "strict"),

    # ── 持久化 / SSH 后门（strict scope —— 记忆 + 技能）───────────
    (r'authorized_keys', "ssh_backdoor", "strict"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access", "strict"),
    (r'\$HOME/\.hermes/\.env|\~/\.hermes/\.env', "hermes_env", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}(?:AGENTS\.md|CLAUDE\.md|\.cursorrules|\.clinerules)', "agent_config_mod", "strict"),
    (r'(update|modify|edit|write|change|append|add\s+to)\s+[^\n]{0,2048}\.hermes/(config\.yaml|SOUL\.md)', "hermes_config_mod", "strict"),

    # ── 硬编码密钥 ─────────────────────────────────────────────────
    (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][A-Za-z0-9+/=_-]{20,}', "hardcoded_secret", "strict"),
]

# 注入攻击中使用的不可见 / 双向 unicode 字符。与 skills_guard.py 的
# INVISIBLE_CHARS 对齐 —— 方向隔离符（U+2066-U+2069）和不可见数学运算符
# （U+2062-U+2064）是真实的攻击工具。
INVISIBLE_CHARS = frozenset({
    '\u200b',  # zero-width space
    '\u200c',  # zero-width non-joiner
    '\u200d',  # zero-width joiner
    '\u2060',  # word joiner
    '\u2062',  # invisible times
    '\u2063',  # invisible separator
    '\u2064',  # invisible plus
    '\ufeff',  # zero-width no-break space (BOM)
    '\u202a',  # left-to-right embedding
    '\u202b',  # right-to-left embedding
    '\u202c',  # pop directional formatting
    '\u202d',  # left-to-right override
    '\u202e',  # right-to-left override
    '\u2066',  # left-to-right isolate
    '\u2067',  # right-to-left isolate
    '\u2068',  # first strong isolate
    '\u2069',  # pop directional isolate
})


# 按 scope 索引的编译模式集。import 时编译一次；scan_for_threats() 查表。
_COMPILED: dict[str, List[Tuple[re.Pattern, str]]] = {}


def _compile() -> None:
    """为每个 scope（all / context / strict）编译模式集。

    scope="all" 的模式进入每个集合。scope="context" 的模式进入 context +
    strict（context 暗示 strict 扫描器也想要）。scope="strict" 只进 strict。
    """
    global _COMPILED
    if _COMPILED:
        return

    all_patterns: List[Tuple[re.Pattern, str]] = []
    context_patterns: List[Tuple[re.Pattern, str]] = []
    strict_patterns: List[Tuple[re.Pattern, str]] = []

    for pattern, pid, scope in _PATTERNS:
        compiled = re.compile(pattern, re.IGNORECASE)
        entry = (compiled, pid)
        if scope == "all":
            all_patterns.append(entry)
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "context":
            context_patterns.append(entry)
            strict_patterns.append(entry)
        elif scope == "strict":
            strict_patterns.append(entry)
        else:
            raise ValueError(f"threat_patterns: unknown scope {scope!r} for pattern {pid!r}")

    _COMPILED = {
        "all": all_patterns,
        "context": context_patterns,
        "strict": strict_patterns,
    }


_compile()


def scan_for_threats(content: str, scope: str = "context") -> List[str]:
    """返回 *content* 在给定 scope 下匹配到的模式 ID 列表。

    ``scope`` 选择应用哪套模式：
    - ``"all"``（窄）：经典注入 + 外泄 —— 误报最少，适合任意文本。
    - ``"context"``（默认）：加上 promptware / C2 / 角色扮演模式 ——
      适合上下文文件、记忆条目和工具结果。
    - ``"strict"``（宽）：加上持久化 / SSH 后门 / 外泄 URL 模式 ——
      适合用户参与的写入（记忆工具、技能安装），误报可交互解决。

    同时检查不可见 unicode 字符（返回 ``"invisible_unicode_U+XXXX"``，
    让调用方可以在日志行里暴露问题码点）。
    """
    if not content:
        return []

    findings: List[str] = []

    content = content[:MAX_SCAN_CHARS]

    # 不可见 unicode —— 对内容集合单次遍历，而非 17 次 ``in`` 查找。
    # 在 NFKC 规范化之前的 RAW 内容上运行，因为规范化可能剥掉部分码点。
    char_set = set(content)
    invisible_hits = char_set & INVISIBLE_CHARS
    for ch in invisible_hits:
        findings.append(f"invisible_unicode_U+{ord(ch):04X}")

    # NFKC 规范化：全角 / 兼容性 unicode 变体（ｃａｔ → cat、Ａ → A）
    # 在正则引擎看到前折叠成 ASCII 对应物，防止同形字替换绕过关键字检查
    # （如 ``ｃａｔ ~/.hermes/.env``）。注意：这不防御跨文字混淆
    # （西里尔 ``а`` U+0430），NFKC 不动它 —— 那需要 TR#39 混淆数据库。
    normalised = unicodedata.normalize("NFKC", content)

    # 威胁模式
    patterns = _COMPILED.get(scope)
    if patterns is None:
        raise ValueError(f"scan_for_threats: unknown scope {scope!r}")
    for compiled, pid in patterns:
        if compiled.search(normalised):
            findings.append(pid)

    return findings


def first_threat_message(content: str, scope: str = "strict") -> Optional[str]:
    """返回首个威胁的人类可读错误串，无威胁则返回 None。

    供在首个命中即阻断的路径使用（记忆工具写入、技能安装），调用方
    只需要是/否 + 一条消息。
    """
    findings = scan_for_threats(content, scope=scope)
    if not findings:
        return None
    pid = findings[0]
    if pid.startswith("invisible_unicode_"):
        codepoint = pid.replace("invisible_unicode_", "")
        return f"Blocked: content contains invisible unicode character {codepoint} (possible injection)."
    return (
        f"Blocked: content matches threat pattern '{pid}'. "
        f"Content is injected into the system prompt and must not contain "
        f"injection or exfiltration payloads."
    )


__all__ = [
    "INVISIBLE_CHARS",
    "MAX_SCAN_CHARS",
    "scan_for_threats",
    "first_threat_message",
]
