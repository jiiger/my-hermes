"""MCP (Model Context Protocol) Client Support —— my-hermes 移植版（Phase 1.5）。

对应原版 hermes-agent tools/mcp_tool.py（7640 行）。架构对齐原版三层：
模块级状态层 / MCPServerTask / 模块级编排层。

Phase 1.5 相比 Phase 1 修复的问题（2026-08-14 对照原版审计）：
- 命名对齐原版 ``mcp__server__tool`` 双下划线（Claude Code/Codex/OpenCode 通用约定）
- keepalive 心跳（ping → list_tools 降级），子进程死亡/会话过期可被感知并重连
- park 状态自探复活（_PARKED_RETRY_INTERVAL），不再永久下线
- handler 熔断器 + 断线重建请求（_signal_reconnect），防模型反复烧迭代
- 注册碰撞 fail-closed（归一化重名/跨 toolset 占用全部跳过 + error 日志）
- schema 归一化四步修复（$ref 提升 / nullable union 剥离 / const union 折叠 / 形状修复）
- 连接冷却（指数退避，防 restart storm）+ 连接错误记录（status 可见）
- ${env:VAR} / ${userHome} 等插值、HERMES_SAFE_MODE 门、可疑 server 过滤、隐藏空白警告
- 中断返回统一 tool_error 格式

Phase 2/3 未实现（预留，保持原版结构占位）：OAuth、sampling/elicitation、
stdio watchdog / schema cache / OSV 预检、trust gate、utility schema（resources/prompts）。

配置：config.yaml 的 ``mcp_servers``（stdio: command/args/env；HTTP: url/headers）。
``mcp`` Python 包为可选依赖；未安装时本模块为 no-op（对应原版语义）。

线程模型（对齐原版）：专用后台 asyncio event loop（``_mcp_loop``）跑在 daemon
线程里；每个 server 是该 loop 上的长驻 asyncio Task（``MCPServerTask``）；
工具调用经 ``_run_on_mcp_loop`` 调度到 loop 并阻塞等待（轮询中断）。
"""

import asyncio
import concurrent.futures
import fnmatch
import importlib.util
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tools.registry import tool_error

logger = logging.getLogger(__name__)

# mcp SDK 可用性：用 find_spec 探测（不导入，避免模块加载成本；原版同款）。
_MCP_AVAILABLE = importlib.util.find_spec("mcp") is not None


# ── 常量（对齐原版）──────────────────────────────────────────────────────

_DEFAULT_TOOL_TIMEOUT = 300.0       # 单次工具调用超时（秒）
_DEFAULT_CONNECT_TIMEOUT = 60.0     # 单 server 首次连接超时（秒）
_MAX_INITIAL_CONNECT_RETRIES = 3    # 首次连接失败重试次数
_MAX_RECONNECT_RETRIES = 5          # 已连接后断线重连次数
_MAX_BACKOFF_SECONDS = 60.0         # 指数退避上限
_PARKED_RETRY_INTERVAL = 300.0      # park 状态自探间隔（park 后每 300s 自醒重连一次）
_MCP_LIST_MAX_PAGES = 50            # list_* 分页上限（防恶意 server 死循环）

_DEFAULT_KEEPALIVE_INTERVAL = 180.0  # 空闲时 liveness 心跳间隔（秒）
_MIN_KEEPALIVE_INTERVAL = 5.0        # 心跳间隔下限（配置值会被夹到该值以上）

_CIRCUIT_BREAKER_THRESHOLD = 3       # 连续失败多少次后熔断
_CIRCUIT_BREAKER_COOLDOWN_SEC = 60.0  # 熔断冷却时长；冷却后下一次调用为半开探测

_CONNECT_RETRY_BASE_BACKOFF_SEC = 30.0   # 连接失败冷却：基础退避
_CONNECT_RETRY_MAX_BACKOFF_SEC = 600.0   # 连接失败冷却：退避上限

_JSONRPC_METHOD_NOT_FOUND = -32601   # JSON-RPC "method not found" 错误码（ping 可选）

# 工具名前缀约定：``mcp__<server>__<tool>``（双下划线分隔符消歧 server/tool
# 边界，即使组件本身含下划线也不碰撞；与 Claude Code/Codex/OpenCode 一致，
# 也是模型训练时见过的命名）。对应原版 :5890-5908。
MCP_TOOL_NAME_PREFIX = "mcp__"
_MCP_NAME_DELIM = "__"


# ── 模块级状态（受 _lock 保护；对应原版 :3726/:4475-4480）────────────────

_servers: Dict[str, "MCPServerTask"] = {}          # 已连接/已注册的 server
_server_connecting: Set[str] = set()               # 正在连接（防重复 spawn）
_parallel_safe_servers: Set[str] = set()           # 允许并行调用的 server
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None
_lock = threading.Lock()

# 连接失败冷却 / 错误记录（防 restart storm，原版 :3764-3795）
_server_connect_failures: Dict[str, int] = {}      # 连续失败次数
_server_connect_retry_after: Dict[str, float] = {}  # 下次允许重试的 monotonic 时刻
_server_connect_errors: Dict[str, str] = {}        # 最近一次连接错误（status 展示）

# 熔断器状态（连续失败计数 + 熔断开启时刻；原版 :3814-3817）
_server_error_counts: Dict[str, int] = {}
_server_breaker_opened_at: Dict[str, float] = {}


def _parse_boolish(value: Any, default: bool = True) -> bool:
    """宽松解析布尔配置（对应原版 :6038）。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _jittered(seconds: float) -> float:
    """给退避时长加 ±20% 抖动，避免多个 server 同步重连（对应原版 :355）。"""
    return max(0.1, seconds * (1.0 + random.uniform(-0.2, 0.2)))


# ── 工具名/schema 转换（对齐原版）────────────────────────────────────────


def sanitize_mcp_name_component(value: str) -> str:
    """把 server/tool 名转成 LLM API 安全名（- . 等 → _）。

    对应原版 mcp_tool.py:5879：只把 ``[A-Za-z0-9_]`` 之外的字符替换成
    ``_``，保留内部下划线不压缩——压缩会让 ``a__b`` 与 ``a_b`` 归一化
    碰撞（双下划线分隔符的消歧就白做了）。
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def mcp_prefixed_tool_name(server_name: str, tool_name: str) -> str:
    """生成 ``mcp__{server}__{tool}`` 注册名（对应原版 :5901）。"""
    safe_server = sanitize_mcp_name_component(server_name)
    safe_tool = sanitize_mcp_name_component(tool_name)
    return f"{MCP_TOOL_NAME_PREFIX}{safe_server}{_MCP_NAME_DELIM}{safe_tool}"


def _normalize_mcp_input_schema(schema: Optional[dict]) -> dict:
    """把 MCP inputSchema 归一化成 OpenAI 兼容 parameters（对应原版 :5725）。

    四步修复（原版逐个踩坑后加的，缺一不可）：
    ① ``definitions`` → ``$defs`` 提升 + ``#/definitions/`` 引用重写
       （Kimi/Moonshot 拒绝 draft-07 的 definitions 形态）；
    ② nullable union（``anyOf: [X, {"type": "null"}]``）折叠成非空分支
       （Anthropic 拒绝 null 分支）；
    ③ 同类型 const union 折叠成 enum（block/goose 移植）；
    ④ object 形状修复：缺 type 补 "object"、缺 properties 补空 dict、
       required 裁剪到 properties 内（Google/Gemini 400）。
    """
    if not schema:
        return {"type": "object", "properties": {}}

    def _rewrite_local_refs(node):
        """遍历 schema，把旧式 ``definitions`` 提升为 ``$defs`` 并重写引用。

        关键门：``definitions`` 作为 JSON Schema 元关键字（properties 的
        兄弟节点）才提升；作为 properties 里的属性名（如 CI 工具的
        ``definitions`` 参数）必须原样保留——改名会把合法属性名变成
        ``$defs``（含 ``$``，OpenAI/Anthropic 直接 400）。
        """
        if isinstance(node, dict):
            normalized = {}
            for key, value in node.items():
                if key in ("properties", "patternProperties") and isinstance(value, dict):
                    # 属性名是用户可见名称，原样保留，只递归进每个属性的 schema
                    normalized[key] = {
                        prop_name: _rewrite_local_refs(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                else:
                    out_key = "$defs" if key == "definitions" else key
                    normalized[out_key] = _rewrite_local_refs(value)
            ref = normalized.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/definitions/"):
                normalized["$ref"] = "#/$defs/" + ref[len("#/definitions/"):]
            return normalized
        if isinstance(node, list):
            return [_rewrite_local_refs(item) for item in node]
        return node

    def _strip_nullable_union(node):
        """折叠 nullable union（委托 tools.schema_sanitizer，与全局清洗共用一套）。"""
        from tools.schema_sanitizer import strip_nullable_unions

        return strip_nullable_unions(node, keep_nullable_hint=True)

    def _collapse_const_unions(node):
        """把同类型 const 的 anyOf/oneOf 折叠成 property enum（原版 :5814）。"""
        from tools.schema_sanitizer import collapse_const_unions

        return collapse_const_unions(node)

    def _repair_object_shape(node):
        """递归修复 object 节点：补 type、补 properties、裁剪 required。"""
        if isinstance(node, list):
            return [_repair_object_shape(item) for item in node]
        if not isinstance(node, dict):
            return node

        repaired = {k: _repair_object_shape(v) for k, v in node.items()}

        # 有 properties/required 但缺 type → 按 object 处理（原版 PR #4897）
        if not repaired.get("type") and (
            "properties" in repaired or "required" in repaired
        ):
            repaired["type"] = "object"

        if repaired.get("type") == "object":
            # 保证 properties 存在且是 dict，required 不会悬空
            if "properties" not in repaired or not isinstance(
                repaired.get("properties"), dict
            ):
                repaired["properties"] = {}
            # required 只保留 properties 里真实存在的名字（原版 PR #4651）
            required = repaired.get("required")
            if isinstance(required, list):
                props = repaired.get("properties") or {}
                valid = [r for r in required if isinstance(r, str) and r in props]
                if len(valid) != len(required):
                    if valid:
                        repaired["required"] = valid
                    else:
                        repaired.pop("required", None)

        return repaired

    normalized = _rewrite_local_refs(schema)
    normalized = _strip_nullable_union(normalized)
    normalized = _collapse_const_unions(normalized)
    normalized = _repair_object_shape(normalized)

    # 顶层必须是良构 object schema
    if not isinstance(normalized, dict):
        return {"type": "object", "properties": {}}
    if normalized.get("type") == "object" and "properties" not in normalized:
        normalized = {**normalized, "properties": {}}

    return normalized


def _convert_mcp_schema(server_name: str, mcp_tool: Any) -> dict:
    """把 MCP tool 对象转成 registry 注册用的 OpenAI 格式 schema。

    对应原版 :5911 的精简版（不含 utility schema / annotations 处理）。
    返回 dict 含 name / description / parameters。
    """
    prefixed = mcp_prefixed_tool_name(server_name, mcp_tool.name)
    return {
        "name": prefixed,
        "description": (
            getattr(mcp_tool, "description", None)
            or f"MCP tool {mcp_tool.name} from {server_name}"
        ),
        "parameters": _normalize_mcp_input_schema(
            getattr(mcp_tool, "inputSchema", None)
        ),
    }


# ── 配置读取（对齐原版 _load_mcp_config，适配 my-hermes）─────────────────

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _env_ref_name(ref: str) -> str:
    """把 ``${...}`` 引用体归一化成环境变量名。

    兼容 Cursor 风格 ``${env:VAR}``：剥离 ``env:`` 前缀（教程文档里
    Cursor/Claude 的配置直接拷过来也能解析）。对应原版 :434。
    """
    ref = ref.strip()
    if ref.startswith("env:"):
        ref = ref[len("env:"):].strip()
    return ref


def _context_var_value(ref: str) -> Optional[str]:
    """解析 Cursor 风格上下文变量（原版 :468 精简版）。

    my-hermes 无 file_tools 权威工作区概念，``${workspaceFolder}`` 用
    ``os.getcwd()`` 兜底。未知引用返回 None（走普通环境变量查找）。
    """
    if ref == "userHome":
        return os.path.expanduser("~")
    if ref == "workspaceFolder":
        return os.getcwd()
    if ref == "workspaceFolderBasename":
        root = os.getcwd()
        return os.path.basename(root.rstrip("/\\")) or root
    if ref in ("pathSeparator", "/"):
        return os.sep
    return None


def _interpolate_env_vars(value: Any) -> Any:
    """递归解析 ``${VAR}`` / ``${env:VAR}`` / ``${userHome}`` 占位符。

    对应原版 :4875。未设置的变量保留字面量（不替换），与原版一致。
    """
    if isinstance(value, str):
        def _replace(m):
            ctx = _context_var_value(m.group(1).strip())
            if ctx is not None:
                return ctx
            name = _env_ref_name(m.group(1))
            return os.environ.get(name, m.group(0))
        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(v) for v in value]
    return value


def _env_var_enabled(name: str) -> bool:
    """宽松判断布尔环境变量是否开启（对应原版 utils.env_var_enabled）。"""
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


# (server_name, 点分键路径) 已警告过的组合——_load_mcp_config 每次发现都会跑，
# 重复警告是噪音，每个组合只警告一次。
_whitespace_warned: Set[Tuple[str, str]] = set()


def _warn_hidden_whitespace(server_name: str, config: dict) -> List[str]:
    """警告配置值里隐藏的首尾空白（对应原版 :4912）。

    粘贴 token 带换行、URL 前带空格会产生无法排查的认证/连接失败；
    只警告不修改（空白理论上可能是故意的）。只报键路径，绝不打印值
    （值通常是秘密）。
    """
    flagged: List[str] = []

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, str):
            if value != value.strip():
                flagged.append(path)
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(v, f"{path}[{i}]")

    _walk(config, "")
    for key_path in flagged:
        dedupe_key = (server_name, key_path)
        if dedupe_key in _whitespace_warned:
            continue
        _whitespace_warned.add(dedupe_key)
        logger.warning(
            "MCP server '%s': config value '%s' has hidden leading or "
            "trailing whitespace — this often causes authentication or "
            "connection failures",
            server_name, key_path,
        )
    return flagged


def _filter_suspicious_mcp_servers(servers: Dict[str, dict]) -> Dict[str, dict]:
    """丢弃渗出/持久化形态的 MCP 配置（对应原版 :4958）。

    委托 hermes_cli/mcp_security.py（hermes-0day IOC 黑名单 + shell
    解释器内联脚本的 egress/persistence 形状）。校验模块缺失时放行。
    """
    try:
        from hermes_cli.mcp_security import (
            validate_mcp_server_entry as _validate_mcp_server_entry,
        )
    except Exception:
        return servers

    safe_servers: Dict[str, dict] = {}
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            safe_servers[name] = cfg
            continue
        issues = _validate_mcp_server_entry(name, cfg)
        if issues:
            logger.warning(
                "Skipping suspicious MCP server '%s': %s",
                name, "; ".join(issues),
            )
            continue
        safe_servers[name] = cfg
    return safe_servers


def _load_mcp_config() -> Dict[str, dict]:
    """读 ``mcp_servers`` 配置（对应原版 :4985，my-hermes 无插件 portable）。

    安全层：HERMES_SAFE_MODE 开启时整体禁用；可疑 server 过滤；
    隐藏空白警告。``${VAR}`` 插值从 os.environ 解析。
    """
    try:
        from hermes_cli.config import load_config
    except Exception:
        return {}
    if _env_var_enabled("HERMES_SAFE_MODE"):
        return {}
    try:
        config = load_config()
    except Exception as exc:
        logger.debug("Failed to load MCP config: %s", exc)
        return {}
    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return {}
    safe: Dict[str, dict] = {}
    for name, cfg in _filter_suspicious_mcp_servers(servers).items():
        if not isinstance(cfg, dict):
            continue
        interpolated = _interpolate_env_vars(cfg)
        if isinstance(interpolated, dict):
            _warn_hidden_whitespace(name, interpolated)
            safe[name] = interpolated
    return safe


# ── 环境安全（对齐原版 _build_safe_env）─────────────────────────────────

# stdio 子进程继承的白名单基础变量（其余环境变量一律不传，防凭据泄漏）。
_SAFE_ENV_KEYS = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
)


def _build_safe_env(user_env: Optional[dict]) -> dict:
    """构造传给 MCP 子进程的安全环境（对应原版 :493）。

    只保留白名单基础变量 + XDG_* + 用户显式 env；API key 等敏感变量
    除非显式配置，否则绝不透传。
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _SAFE_ENV_KEYS or k.startswith("XDG_")
    }
    if user_env:
        env.update({str(k): str(v) for k, v in user_env.items()})
    return env


def _resolve_stdio_command(command: str, env: dict) -> tuple:
    """把命令解析成绝对路径（对应原版 :702 精简，无 watchdog 包装）。"""
    if "/" in command or os.path.isabs(command):
        return command, env
    resolved = shutil.which(command)
    if resolved:
        return resolved, env
    return command, env


# ── MCP 子进程 stderr 重定向（对齐原版 :138-200）────────────────────────
# MCP SDK 的 stdio_client(server, errlog=sys.stderr) 默认把子进程 stderr
# 接到父进程真实 stderr（用户 TTY）。FastMCP banner、context7 等启动横幅
# 会直接写到终端——交互界面渲染时破坏显示，看起来像"自动输入到输入栏"。
# 这里重定向到共享日志文件 ~/.hermes/logs/mcp-stderr.log（按 server 名打
# 时间戳标记），保留下日志可查。
_mcp_stderr_log_fh: Optional[Any] = None
_mcp_stderr_log_lock = threading.Lock()


def _get_mcp_stderr_log() -> Any:
    """返回共享 append 模式文件句柄（进程内复用，所有 stdio server 共用）。

    必须有真实 OS 文件描述符（fileno()）——asyncio subprocess 把子进程
    stderr 直接接到该 fd。打开失败回退 os.devnull，再失败回退真实 stderr。
    """
    global _mcp_stderr_log_fh
    with _mcp_stderr_log_lock:
        if _mcp_stderr_log_fh is not None:
            return _mcp_stderr_log_fh
        try:
            from hermes_constants import get_hermes_home

            log_dir = get_hermes_home() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "mcp-stderr.log"
            # 行缓冲让 server 输出及时落盘；errors=replace 容忍乱码输出。
            fh = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
            fh.fileno()  # 确认有真实 fd
            _mcp_stderr_log_fh = fh
        except Exception as exc:
            logger.debug("Failed to open MCP stderr log, using devnull: %s", exc)
            try:
                _mcp_stderr_log_fh = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                _mcp_stderr_log_fh = sys.stderr  # 最后兜底：原行为
        return _mcp_stderr_log_fh


def _write_stderr_log_header(server_name: str) -> None:
    """spawn 前写人类可读的会话标记（在共享日志文件里定位各 server 输出）。"""
    fh = _get_mcp_stderr_log()
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        fh.write(f"\n===== [{ts}] starting MCP server '{server_name}' =====\n")
        fh.flush()
    except Exception:
        pass


# ── 工具描述注入扫描（对应原版 :593-638）─────────────────────────────────

# MCP 工具描述里的 prompt injection 模式。WARNING 级：只记日志不拦截
# （误报会误伤合法 server）。
_MCP_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
     "prompt override attempt ('ignore previous instructions')"),
    (re.compile(r"you\s+are\s+now\s+a", re.I),
     "identity override attempt ('you are now a...')"),
    (re.compile(r"your\s+new\s+(task|role|instructions?)\s+(is|are)", re.I),
     "task override attempt"),
    (re.compile(r"system\s*:\s*", re.I),
     "system prompt injection attempt"),
    (re.compile(r"<\s*(system|human|assistant)\s*>", re.I),
     "role tag injection attempt"),
    (re.compile(r"do\s+not\s+(tell|inform|mention|reveal)", re.I),
     "concealment instruction"),
    (re.compile(r"(curl|wget|fetch)\s+https?://", re.I),
     "network command in description"),
    (re.compile(r"base64\.(b64decode|decodebytes)", re.I),
     "base64 decode reference"),
    (re.compile(r"exec\s*\(|eval\s*\(", re.I),
     "code execution reference"),
    (re.compile(r"import\s+(subprocess|os|shutil|socket)", re.I),
     "dangerous import reference"),
]


def _scan_mcp_description(
    server_name: str, tool_name: str, description: str
) -> List[str]:
    """扫描 MCP 工具描述中的注入模式（对应原版 :620）。"""
    findings = []
    if not description:
        return findings
    for pattern, reason in _MCP_INJECTION_PATTERNS:
        if pattern.search(description):
            findings.append(reason)
    if findings:
        logger.warning(
            "MCP server '%s' tool '%s': suspicious description content — %s. "
            "Description: %.200s",
            server_name, tool_name, "; ".join(findings),
            description,
        )
    return findings


# ── 错误工具（对应原版 :557/:1027/:1338）─────────────────────────────────


def _is_method_not_found_error(exc: BaseException) -> bool:
    """是否 JSON-RPC ``method not found``（-32601）。

    ``ping`` 是 MCP 可选工具，未实现的 server 回 -32601。先结构化检查
    McpError.error.code，再子串兜底（防 SDK 版本漂移，如 agentmemory 的
    "Unknown method: <name>" 措辞，原版 #50028）。
    """
    err = getattr(exc, "error", None)
    code = getattr(err, "code", None)
    if code == _JSONRPC_METHOD_NOT_FOUND:
        return True
    msg = str(exc).lower()
    if not msg:
        return False
    return (
        str(_JSONRPC_METHOD_NOT_FOUND) in msg
        or "method not found" in msg
        or "unknown method" in msg
        or "not found: ping" in msg
    )


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    """解包 anyio TaskGroup 异常组，取出第一个真实异常（对应原版 :1027）。"""
    if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        return _unwrap_exception_group(exc.exceptions[0])
    return exc


def _format_connect_error(exc: BaseException) -> str:
    """把嵌套的连接错误渲染成可操作的短消息（对应原版 :1338）。

    重点：递归找 FileNotFoundError（缺失的可执行文件），对 npx/npm/node
    给出 Node 安装提示；其余消息扁平化去重后取前 3 条。
    """
    def _find_missing(current: BaseException) -> Optional[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            for child in nested:
                missing = _find_missing(child)
                if missing:
                    return missing
            return None
        if isinstance(current, FileNotFoundError):
            if getattr(current, "filename", None):
                return str(current.filename)
            match = re.search(r"No such file or directory: '([^']+)'", str(current))
            if match:
                return match.group(1)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                missing = _find_missing(nested_exc)
                if missing:
                    return missing
        return None

    def _flatten_messages(current: BaseException) -> List[str]:
        nested = getattr(current, "exceptions", None)
        if nested:
            flattened: List[str] = []
            for child in nested:
                flattened.extend(_flatten_messages(child))
            return flattened
        messages = []
        text = str(current).strip()
        if text:
            messages.append(text)
        for attr in ("__cause__", "__context__"):
            nested_exc = getattr(current, attr, None)
            if isinstance(nested_exc, BaseException):
                messages.extend(_flatten_messages(nested_exc))
        return messages or [current.__class__.__name__]

    missing = _find_missing(exc)
    if missing:
        message = f"missing executable '{missing}'"
        if os.path.basename(missing) in {"npx", "npm", "node"}:
            message += (
                " (ensure Node.js is installed and PATH includes its bin directory, "
                "or set mcp_servers.<name>.command to an absolute path and include "
                "that directory in mcp_servers.<name>.env.PATH)"
            )
        return _sanitize_error(message)

    deduped: List[str] = []
    for item in _flatten_messages(exc):
        if item not in deduped:
            deduped.append(item)
    return _sanitize_error("; ".join(deduped[:3]))


# ── 发现分页（对齐原版 _paginate_full_list）──────────────────────────────


async def _paginate_full_list(
    list_method: Callable, items_attr: str, server_name: str
) -> list:
    """按 nextCursor 分页拉全 list_* 结果（对应原版 :661）。"""
    items: list = []
    cursor = None
    for _ in range(_MCP_LIST_MAX_PAGES):
        result = await (list_method(cursor=cursor) if cursor else list_method())
        items.extend(getattr(result, items_attr, None) or [])
        cursor = getattr(result, "nextCursor", None)
        if not cursor:
            break
    else:
        logger.warning(
            "MCP server '%s': %s pagination exceeded %d pages",
            server_name, items_attr, _MCP_LIST_MAX_PAGES,
        )
    return items


# ── MCPServerTask：单 server 全生命周期（对应原版 :1975）─────────────────


class MCPServerTask:
    """管理单个 MCP server 连接的 asyncio Task。

    连接生命周期（connect/discover/serve/disconnect）都跑在同一个
    asyncio Task 里，保证 anyio cancel-scope 进出同 Task（原版要求）。
    支持 stdio 与 HTTP/StreamableHTTP 两种传输。

    注意：故意不用 ``__slots__``（对齐原版）——后续按原版补字段
    （_recycled_reason / _pending_call_context 等）时不会 AttributeError。
    """

    def __init__(self, name: str):
        self.name = name
        self.session: Optional[Any] = None
        self.tool_timeout: float = _DEFAULT_TOOL_TIMEOUT
        self._task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()
        self._tools: list = []
        self._error: Optional[Exception] = None
        self._config: dict = {}
        self._registered_tool_names: list = []
        # stdio session 是单条 JSON-RPC 流：串行化客户端发起的 RPC，防止
        # list_tools 刷新与工具调用互相楔死（原版 _rpc_lock 语义）。
        self._rpc_lock = asyncio.Lock()
        self._reconnect_retries: int = 0
        # 快速掉线预算（原版 #62212）：新会话在证明健康前（存活 ≥1 个
        # keepalive 周期或成功调用过 ≥1 次）不释放重连预算——握手成功
        # 但立刻掉线的 flapping 传输必须继续计费直到 park，否则无限 spawn。
        self._session_proven: bool = False
        # park 过（重连预算耗尽）；会话再次证明健康时记录 parked→revived
        # 转换日志（_mark_session_proven）。
        self._was_parked: bool = False
        # ``session.initialize()`` 的返回值：下游检查 server 真实声明
        # 的能力（capabilities.tools 等），原版 #18051。
        self.initialize_result: Optional[Any] = None
        # 首次 keepalive ping 收到 -32601 后置 True（server 不支持可选
        # ping），后续心跳降级 list_tools；每次新连接重置。
        self._ping_unsupported: bool = False
        now = time.monotonic()
        self._lifecycle_started_at: float = now
        self._last_tool_call_at: float = now

    # ── 状态辅助 ──────────────────────────────────────────────────────────

    def _is_http(self) -> bool:
        """是否 HTTP 传输（url 配置存在即 HTTP）。"""
        return "url" in self._config

    def mark_tool_call(self) -> None:
        """记录一次用户可见的 MCP 操作（生命周期统计用）。"""
        self._last_tool_call_at = time.monotonic()

    def _advertises_tools(self) -> bool:
        """server 是否声明 tools 能力（原版 :2074）。

        按 MCP 规范，``InitializeResult.capabilities.tools`` 非 None 才
        说明 server 实现了 tools/* 请求族；prompt-only / resource-only
        server 会省略它，对它调 list_tools 会抛 McpError(-32601) 并杀掉
        连接（opencode#31271 移植）。无能力信息（legacy server）返回
        True 保持旧行为（一律尝试 list_tools）。
        """
        init_result = self.initialize_result
        caps = (
            getattr(init_result, "capabilities", None)
            if init_result is not None
            else None
        )
        if caps is None:
            return True
        return getattr(caps, "tools", None) is not None

    # ── keepalive 心跳（对应原版 :2322）──────────────────────────────────

    async def _keepalive_probe(self) -> None:
        """探测会话活性：优先 ping，-32601 时降级 list_tools。

        真实连接失败（超时/传输关闭/会话过期）会抛异常，由调用方触发
        重连；存活则正常返回。``_ping_unsupported`` 锁存一次，避免对
        不支持 ping 的 server 反复触发 reconnect 循环。
        """
        if not self._ping_unsupported:
            try:
                await asyncio.wait_for(self.session.send_ping(), timeout=30.0)
                return
            except Exception as exc:
                # 只有 "method not found" 意味着 ping 不支持；其他错误
                # （超时、传输关闭、会话过期）都是真实活性失败 → 上抛。
                if not _is_method_not_found_error(exc):
                    raise
                if not self._advertises_tools():
                    # 无 ping 无 tools → 没有更廉价的探测手段可用
                    raise
                self._ping_unsupported = True
                logger.info(
                    "MCP server '%s': does not implement the optional 'ping' "
                    "utility (-32601); using 'list_tools' for keepalive on "
                    "this connection.",
                    self.name,
                )
        # 不支持 ping 的 server 的降级探测
        await asyncio.wait_for(self.session.list_tools(), timeout=30.0)

    def _mark_session_proven(self) -> None:
        """记录当前会话已证明健康（原版 :2362）。

        keepalive 成功路径与工具调用成功路径都会调用。只有 proven 才清
        重连预算；否则 flapping 传输继续计费直到 park（#62212）。
        """
        if not self._session_proven:
            self._session_proven = True
            self._reconnect_retries = 0
            if self._was_parked:
                self._was_parked = False
                logger.warning(
                    "MCP server '%s': revived — session healthy again after "
                    "parking (state: parked → connected)",
                    self.name,
                )

    # ── 发现与注册 ────────────────────────────────────────────────────────

    async def _discover_tools(self) -> None:
        """从已连接 session 发现工具并注册（对应原版 :3177）。"""
        if self.session is None:
            return
        if not self._advertises_tools():
            logger.info(
                "MCP server '%s': does not advertise 'tools' capability",
                self.name,
            )
            self._tools = []
            self._register_discovered_tools_if_needed()
            return
        async with self._rpc_lock:
            self._tools = await _paginate_full_list(
                self.session.list_tools, "tools", self.name
            )
        self._register_discovered_tools_if_needed()

    def _register_discovered_tools_if_needed(self) -> None:
        """把发现的工具注册进全局 registry（原版 :3208）。

        只注册一次（_registered_tool_names 非空即跳过）；重连后工具可能
        已注销，允许重新发布。首次连接时 _servers 尚无本 server（注册
        由 _discover_and_register_server 显式完成），此判断防竞态。
        """
        if self._registered_tool_names:
            return
        if not self._ready.is_set():
            with _lock:
                if _servers.get(self.name) is not self:
                    return
        self._registered_tool_names = _register_server_tools(
            self.name, self, self._config
        )

    # ── 连接体（stdio / HTTP）─────────────────────────────────────────────

    async def _run_stdio(self, config: dict) -> None:
        """stdio 传输：spawn 子进程 → ClientSession → initialize → 发现 → 等待。"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not _MCP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires the 'mcp' Python SDK, "
                "but it is not installed."
            )
        command = config.get("command")
        args = config.get("args", [])
        if not command:
            raise ValueError(
                f"MCP server '{self.name}' has no 'command' in config"
            )

        safe_env = _build_safe_env(config.get("env"))
        command, safe_env = _resolve_stdio_command(command, safe_env)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=safe_env if safe_env else None,
            cwd=config.get("cwd"),
            encoding_error_handler="replace",
        )
        connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)

        # 子进程 stderr 重定向到共享日志文件（对齐原版）：SDK 默认把
        # stderr 接到用户 TTY，MCP server 的启动横幅（context7/FastMCP）
        # 会直接打到终端/输入栏。重定向后 ~/.hermes/logs/mcp-stderr.log
        # 仍可查各 server 日志。
        _write_stderr_log_header(self.name)
        _errlog = _get_mcp_stderr_log()
        async with stdio_client(server_params, errlog=_errlog) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                self.session = session
                self.initialize_result = await asyncio.wait_for(
                    session.initialize(), timeout=connect_timeout
                )
                self._ping_unsupported = False  # 新连接重置 ping 锁存
                # 对齐原版顺序：先发现（填充 self._tools），再置就绪。
                # 若先 set ready，start() 会提前返回，_discover_and_register_server
                # 显式注册时 _tools 还是空的（竞态）。
                await self._discover_tools()
                # 连接成功 = 新会话建立：清熔断计数（对齐原版 _run_stdio
                # 成功路径的 _reset_server_error；否则 server 恢复后计数
                # 仍 ≥ 阈值，下一次调用会被熔断 gate 挡在门外）。
                _reset_server_error(self.name)
                self._ready.set()
                self._error = None
                self._session_proven = False
                # 等待 shutdown / reconnect 信号（keepalive 轮询在内）
                await self._wait_for_lifecycle_event()
        self.session = None

    async def _run_http(self, config: dict) -> None:
        """HTTP/StreamableHTTP 传输（无 preflight/OAuth，Phase 2/3 补）。"""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        if not _MCP_AVAILABLE:
            raise ImportError(
                f"MCP server '{self.name}' requires the 'mcp' Python SDK, "
                "but it is not installed."
            )
        url = config.get("url")
        if not url:
            raise ValueError(f"MCP server '{self.name}' has no 'url' in config")
        headers = config.get("headers") or {}
        connect_timeout = config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT)

        async with streamable_http_client(url, headers=headers) as (
            read_stream,
            write_stream,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                self.session = session
                self.initialize_result = await asyncio.wait_for(
                    session.initialize(), timeout=connect_timeout
                )
                self._ping_unsupported = False
                # 对齐原版顺序：先发现再就绪（同上注释）
                await self._discover_tools()
                _reset_server_error(self.name)
                self._ready.set()
                self._error = None
                self._session_proven = False
                await self._wait_for_lifecycle_event()
        self.session = None

    # ── 生命周期信号等待（对应原版 :2383）────────────────────────────────

    async def _wait_for_lifecycle_event(self) -> str:
        """等待 shutdown / reconnect 信号；空闲期定期 keepalive 心跳。

        心跳周期 = ``keepalive_interval`` 配置（默认 180s，下限 5s）。
        心跳失败 → 置 reconnect 信号 → 返回 "reconnect"，外层 run()
        重建传输。这解决了 Phase 1 的死局：stdio 子进程死亡 / HTTP 会话
        过期后无人感知，连接永远处于"看起来活着"的死状态。
        """
        keepalive_interval = max(
            _MIN_KEEPALIVE_INTERVAL,
            float(self._config.get("keepalive_interval", _DEFAULT_KEEPALIVE_INTERVAL)),
        )

        shutdown_task = asyncio.create_task(self._shutdown_event.wait())
        reconnect_task = asyncio.create_task(self._reconnect_event.wait())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    {shutdown_task, reconnect_task},
                    timeout=keepalive_interval,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if done:
                    break

                # 超时未收到信号 → 心跳探测活性
                if self.session:
                    try:
                        await self._keepalive_probe()
                    except Exception as exc:
                        root = _unwrap_exception_group(exc)
                        logger.warning(
                            "MCP server '%s' keepalive failed, triggering "
                            "reconnect (state: connected → degraded): %s: %s",
                            self.name, type(root).__name__, root,
                        )
                        self._reconnect_event.set()
                        break
                    # 心跳成功 = 会话存活满一个完整周期，是真实健康证明
                    self._mark_session_proven()
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass

        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    async def _wait_for_parked_revival(self, timeout: Optional[float] = None) -> str:
        """park 状态：等待 shutdown 或 reconnect；超时自探（对应原版 :2488）。

        park 期间工具已注销，任何工具调用都到不了熔断器的半开探测或
        _signal_reconnect——没有自探的话 park 的 server 只能靠进程重启
        复活。每 ``timeout``（默认 _PARKED_RETRY_INTERVAL）自醒一次尝试
        重连，或由显式 reconnect 请求提前唤醒。
        """
        shutdown_task = asyncio.ensure_future(self._shutdown_event.wait())
        reconnect_task = asyncio.ensure_future(self._reconnect_event.wait())
        try:
            await asyncio.wait(
                {shutdown_task, reconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=timeout,
            )
        finally:
            for t in (shutdown_task, reconnect_task):
                if not t.done():
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
        if self._shutdown_event.is_set():
            return "shutdown"
        self._reconnect_event.clear()
        return "reconnect"

    # ── 主循环（对应原版 run() :3237）────────────────────────────────────

    async def run(self, config: dict) -> None:
        """长驻协程：连接 → 发现 → 等待（心跳）→ 断线重连（指数退避）→ park。

        状态机：首次连接重试（_MAX_INITIAL_CONNECT_RETRIES）→ 连接成功后
        断线重连（_MAX_RECONNECT_RETRIES，快速掉线预算 #62212）→ 重试耗尽
        park（每 _PARKED_RETRY_INTERVAL 自探复活一次）。错误分类
        （permanent/transient）与 park 自探细节对齐原版语义。
        """
        self._config = config
        self.tool_timeout = config.get("timeout", _DEFAULT_TOOL_TIMEOUT)
        initial_retries = 0
        backoff = 1.0
        while True:
            try:
                if self._is_http():
                    await self._run_http(config)
                else:
                    await self._run_stdio(config)
                # 传输正常退出：shutdown → 退出；否则是 reconnect 信号
                # （keepalive 失败 / 显式重建请求 / 熔断半开恢复）。
                if self._shutdown_event.is_set():
                    return
                # 只有 proven 会话（存活 ≥1 个 keepalive 周期或成功调用过）
                # 才重置重连预算；快速掉线会话继续计费（#62212）。
                if self._session_proven:
                    self._reconnect_retries = 0
                    backoff = 1.0
                else:
                    self._reconnect_retries += 1
                    if self._reconnect_retries > _MAX_RECONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s': %d consecutive reconnects "
                            "without a healthy session, parking; will "
                            "self-probe every %ds until it recovers "
                            "(state: degraded → parked)",
                            self.name, _MAX_RECONNECT_RETRIES,
                            _PARKED_RETRY_INTERVAL,
                        )
                        self._was_parked = True
                        self._deregister_tools()
                        self._reconnect_event.clear()
                        parked = await self._wait_for_parked_revival(
                            timeout=_PARKED_RETRY_INTERVAL
                        )
                        if parked == "shutdown":
                            return
                        logger.debug(
                            "MCP server '%s': attempting revival from parked "
                            "state (self-probe or explicit reconnect request)",
                            self.name,
                        )
                        # 每次唤醒只探一次：唤醒后立刻回到 _MAX，再次失败
                        # 直接再 park，避免无预算的热循环。
                        self._reconnect_retries = _MAX_RECONNECT_RETRIES
                        backoff = 1.0
                # 清就绪与 session：_run_* 重进后会重新填充。留着 _ready
                # 会让 handler 侧把旧 session 误当新会话提前重试。
                self._ready.clear()
                self.session = None
                continue
            except asyncio.CancelledError:
                # 任务被取消（shutdown/显式 cancel）：不当作连接失败，
                # 让取消正常传播（对应原版 #9930 处理）。
                self.session = None
                raise
            except Exception as exc:
                self.session = None
                root = _unwrap_exception_group(exc)
                # 首次连接阶段失败：重试 N 次后 park（带自探）
                if not self._ready.is_set():
                    initial_retries += 1
                    if initial_retries > _MAX_INITIAL_CONNECT_RETRIES:
                        logger.warning(
                            "MCP server '%s' failed initial connection after "
                            "%d attempts, parking until reconnect requested "
                            "(state: connecting → parked): %s",
                            self.name, _MAX_INITIAL_CONNECT_RETRIES, root,
                        )
                        self._error = exc
                        self._ready.set()
                        self._deregister_tools()
                        self._reconnect_event.clear()
                        parked = await self._wait_for_parked_revival(
                            timeout=_PARKED_RETRY_INTERVAL
                        )
                        if parked == "shutdown":
                            return
                        logger.debug(
                            "MCP server '%s': attempting revival from parked "
                            "state (self-probe or explicit reconnect request)",
                            self.name,
                        )
                        initial_retries = 0
                        backoff = 1.0
                        self._error = None
                        self._ready.clear()
                        continue
                    logger.debug(
                        "MCP server '%s' initial connection failed "
                        "(attempt %d/%d), retrying in %.0fs: %s",
                        self.name, initial_retries,
                        _MAX_INITIAL_CONNECT_RETRIES, backoff, root,
                    )
                    await asyncio.sleep(_jittered(backoff))
                    backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                    if self._shutdown_event.is_set():
                        self._error = exc
                        self._ready.set()
                        return
                    continue
                # 已连接过：断线重连（指数退避），超限后 park
                if self._shutdown_event.is_set():
                    return
                self._reconnect_retries += 1
                if self._reconnect_retries > _MAX_RECONNECT_RETRIES:
                    logger.warning(
                        "MCP server '%s' failed after %d reconnection "
                        "attempts, parking until reconnect requested "
                        "(state: degraded → parked): %s",
                        self.name, _MAX_RECONNECT_RETRIES, root,
                    )
                    self._was_parked = True
                    self._deregister_tools()
                    self._reconnect_event.clear()
                    parked = await self._wait_for_parked_revival(
                        timeout=_PARKED_RETRY_INTERVAL
                    )
                    if parked == "shutdown":
                        return
                    logger.debug(
                        "MCP server '%s': attempting revival from parked "
                        "state (self-probe or explicit reconnect request)",
                        self.name,
                    )
                    self._reconnect_retries = _MAX_RECONNECT_RETRIES
                    backoff = 1.0
                    continue
                logger.debug(
                    "MCP server '%s' connection lost (attempt %d/%d), "
                    "reconnecting in %.0fs: %s",
                    self.name, self._reconnect_retries,
                    _MAX_RECONNECT_RETRIES, backoff, root,
                )
                await asyncio.sleep(_jittered(backoff))
                backoff = min(backoff * 2, _MAX_BACKOFF_SECONDS)
                if self._shutdown_event.is_set():
                    return
            finally:
                self.session = None

    # ── 启动 / 关闭（对应原版 start/shutdown :3638/:3657）────────────────

    async def start(self, config: dict) -> None:
        """创建后台 Task 并等待就绪（或失败）。"""
        self._task = asyncio.ensure_future(self.run(config))
        try:
            await self._ready.wait()
        except asyncio.CancelledError:
            # 调用方超时取消本协程时，回收独立的 run() 任务，避免
            # 挂起的传输无人清理（对应原版 #59349）。
            if self._task and not self._task.done():
                self._task.cancel()
            raise
        if self._error:
            raise self._error

    async def shutdown(self) -> None:
        """置关闭信号并等待资源清理。"""
        self._shutdown_event.set()
        self._reconnect_event.set()
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning(
                    "MCP server '%s' shutdown timed out, cancelling task",
                    self.name,
                )
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        self._deregister_tools()
        self.session = None

    def _deregister_tools(self) -> None:
        """把本 server 的工具从全局 registry 注销（幂等，原版 :3687）。"""
        from tools.registry import registry

        for tname in list(self._registered_tool_names):
            registry.deregister(tname)
        self._registered_tool_names = []


# ── 后台事件循环（对齐原版 _ensure_mcp_loop / _run_on_mcp_loop）─────────


def _mcp_loop_exception_handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
    """MCP loop 异常处理器：记录但不崩溃 loop。"""
    logger.warning(
        "MCP event loop error: %s",
        context.get("exception") or context.get("message"),
    )


def _ensure_mcp_loop() -> None:
    """启动后台事件循环线程（幂等）。"""
    global _mcp_loop, _mcp_thread
    with _lock:
        if _mcp_loop is not None and _mcp_loop.is_running():
            return
        _mcp_loop = asyncio.new_event_loop()
        _mcp_loop.set_exception_handler(_mcp_loop_exception_handler)
        _mcp_thread = threading.Thread(
            target=_mcp_loop.run_forever,
            name="mcp-event-loop",
            daemon=True,
        )
        _mcp_thread.start()


def _run_on_mcp_loop(coro_or_factory: Any, timeout: float = 30) -> Any:
    """调度协程到 MCP loop 并阻塞等待（对应原版 :4790）。

    接受协程对象或零参工厂（避免 loop 不可用时构造协程泄漏帧）。
    轮询等待期间检查线程中断（tools.interrupt.is_interrupted），
    让 MCP 调用可被用户中断打断。
    """
    from tools.interrupt import is_interrupted

    with _lock:
        loop = _mcp_loop
    if loop is None or not loop.is_running():
        if asyncio.iscoroutine(coro_or_factory):
            coro_or_factory.close()
        raise RuntimeError("MCP event loop is not running")
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    start = time.monotonic()
    while True:
        if is_interrupted():
            future.cancel()
            raise InterruptedError("User interrupted during MCP call")
        remaining = start + timeout - time.monotonic()
        if remaining <= 0:
            future.cancel()
            raise TimeoutError(
                f"MCP call timed out after {timeout:.1f}s (configured timeout: {timeout:.1f}s)"
            )
        try:
            return future.result(timeout=min(0.1, remaining))
        except concurrent.futures.TimeoutError:
            if future.done():
                return future.result()
            continue


def _stop_mcp_loop() -> None:
    """停止后台 loop（无 server 时调用，幂等）。"""
    global _mcp_loop, _mcp_thread
    with _lock:
        loop, thread = _mcp_loop, _mcp_thread
        _mcp_loop = None
        _mcp_thread = None
    if loop is not None and loop.is_running():
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass
    if thread is not None:
        thread.join(timeout=5)


# ── 熔断 / 冷却 / 恢复（对应原版 :3764-4063）─────────────────────────────


def _record_connect_failure(server_name: str) -> None:
    """连接失败后盖指数退避冷却戳（原版 :3768）。

    连续失败次数几何增长、封顶 _CONNECT_RETRY_MAX_BACKOFF_SEC——永久
    坏掉的 server 会沉降为低频重试，而不是每次发现都 tight respawn
    （#50394 restart storm）。须在 _lock 内调用。
    """
    n = _server_connect_failures.get(server_name, 0) + 1
    _server_connect_failures[server_name] = n
    backoff = min(
        _CONNECT_RETRY_BASE_BACKOFF_SEC * (2 ** (n - 1)),
        _CONNECT_RETRY_MAX_BACKOFF_SEC,
    )
    _server_connect_retry_after[server_name] = time.monotonic() + backoff


def _clear_connect_failure(server_name: str) -> None:
    """连接成功后清冷却状态（原版 :3786）。"""
    _server_connect_failures.pop(server_name, None)
    _server_connect_retry_after.pop(server_name, None)


def _connect_cooldown_active(server_name: str) -> bool:
    """server 是否仍在重试冷却期内（原版 :3792）。"""
    deadline = _server_connect_retry_after.get(server_name)
    return deadline is not None and time.monotonic() < deadline


def _bump_server_error(server_name: str) -> None:
    """连续失败计数 +1；跨过阈值时盖熔断开启时刻（原版 :3964）。

    状态机：closed（低于阈值全放行）→ open（达到阈值短路，冷却内明确
    告诉模型别重试）→ half-open（冷却过后下一个调用作为探测）→ 成功回
    closed / 失败重新 open。
    """
    n = _server_error_counts.get(server_name, 0) + 1
    _server_error_counts[server_name] = n
    if n >= _CIRCUIT_BREAKER_THRESHOLD:
        _server_breaker_opened_at[server_name] = time.monotonic()


def _reset_server_error(server_name: str) -> None:
    """完全关闭熔断器（原版 :3977）：清计数与开启时刻。

    任何明确的成功信号（工具调用成功、重连成功）都调用。
    """
    _server_error_counts[server_name] = 0
    _server_breaker_opened_at.pop(server_name, None)


def _signal_reconnect(server: Any) -> bool:
    """线程安全地请求 server 任务重建传输（原版 :3988）。

    工具 handler 跑在调用方线程，而 server 任务和它的 _reconnect_event
    活在后台 MCP loop 上——跨线程 set asyncio.Event 必须经
    ``loop.call_soon_threadsafe``。返回 False 表示 server 没有重连机制。
    """
    event = getattr(server, "_reconnect_event", None)
    if event is None:
        return False
    loop = _mcp_loop
    if (
        isinstance(event, asyncio.Event)
        and loop is not None
        and loop.is_running()
    ):
        loop.call_soon_threadsafe(event.set)
    else:
        event.set()
    return True


def reconnect_mcp_server(server_name: str) -> bool:
    """请求一个现存 server 重建传输（对应原版 :4015，供外部刷新调用）。"""
    with _lock:
        server = _servers.get(server_name)
    if server is None:
        return False
    return _signal_reconnect(server)


def _wait_for_server_session_ready(
    srv: MCPServerTask,
    *,
    old_session: Any = None,
    timeout: float = 15.0,
) -> bool:
    """等待 server 暴露可用 session（原版 :4024）。

    handler 跑在工作线程而传输活在后台 loop：重连窗口里 ``srv.session``
    可能是 None 或仍指向旧会话。盲目重试会烧熔断计数。提供 old_session
    时要求观察到的新会话对象不同，避免把重连前的旧会话当新鲜的。
    """
    poll_interval = 0.25
    iterations = max(1, int(max(float(timeout), 0.0) / poll_interval))
    for i in range(iterations):
        session = getattr(srv, "session", None)
        ready = getattr(srv, "_ready", None)
        is_ready = True
        if ready is not None and hasattr(ready, "is_set"):
            try:
                is_ready = bool(ready.is_set())
            except Exception:
                is_ready = True
        if session is not None and session is not old_session and is_ready:
            return True
        if i < iterations - 1:
            time.sleep(poll_interval)
    return False


# ── 连接与发现编排（对齐原版 _connect_server / register_mcp_servers）────


async def _connect_server(name: str, config: dict) -> MCPServerTask:
    """创建 MCPServerTask 并启动，就绪后返回（对应原版 :5043）。"""
    server = MCPServerTask(name)
    try:
        await server.start(config)
    except asyncio.CancelledError:
        raise
    except BaseException:
        # 启动失败：回收孤儿任务（无 owner 时必须自己清，原版同款）
        try:
            await server.shutdown()
        except Exception:
            pass
        raise
    return server


async def _discover_and_register_server(name: str, config: dict) -> list:
    """连接单个 server 并注册工具（带 connect_timeout，原版 :6565 精简）。

    首次注册在这里显式完成：连接就绪后直接对 ``server._tools`` 注册，
    并写回 ``server._registered_tool_names``（server 内部的
    ``_register_discovered_tools_if_needed`` 只负责重连后的自动重注册）。
    """
    try:
        server = await asyncio.wait_for(
            _connect_server(name, config),
            timeout=config.get("connect_timeout", _DEFAULT_CONNECT_TIMEOUT),
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"MCP server '{name}' connect timed out after "
            f"{config.get('connect_timeout', _DEFAULT_CONNECT_TIMEOUT)}s"
        )
    with _lock:
        _servers[name] = server
    registered_names = _register_server_tools(name, server, config)
    server._registered_tool_names = list(registered_names)
    return registered_names


def register_mcp_servers(servers: Dict[str, dict]) -> List[str]:
    """连接显式提供的 MCP server 并注册工具（对应原版 :6631）。

    幂等：已连接/正在连接/冷却中的 server 跳过；``enabled: false`` 跳过。
    并行连接（asyncio.gather），失败记录日志（含格式化错误 + 冷却戳）
    不阻断其他 server。park/重连中的 server 会被 nudge 唤醒（#50170）。
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK not available -- skipping explicit MCP registration")
        return []
    if not servers:
        return []

    servers = _filter_suspicious_mcp_servers(servers)
    if not servers:
        return []

    with _lock:
        connecting = set(_server_connecting)
        new_servers = {
            k: v
            for k, v in servers.items()
            if k not in _servers
            and k not in connecting
            and _parse_boolish(v.get("enabled", True), default=True)
            # 冷却期内跳过：坏 server 不会在每个发现入口都重新 spawn
            # （#50394 restart storm）。成功连接后自动清冷却。
            and not _connect_cooldown_active(k)
        }
        # 已注册但 session 断开的 server（park/重连中）：工具已注销，
        # 没别的东西能碰到 _signal_reconnect——显式 nudge 唤醒，让工具
        # 尽快回来，而不是等最长 _PARKED_RETRY_INTERVAL 的自探（#50170）。
        stale_cached = [
            _servers[k]
            for k in servers
            if k in _servers and getattr(_servers[k], "session", None) is None
        ]
        _server_connecting.update(new_servers)
        for srv_name in new_servers:
            _server_connect_errors.pop(srv_name, None)
        # 并行安全 server 追踪（幂等）
        for srv_name, srv_cfg in servers.items():
            if _parse_boolish(
                srv_cfg.get("supports_parallel_tool_calls", False), default=False
            ):
                _parallel_safe_servers.add(srv_name)
            else:
                _parallel_safe_servers.discard(srv_name)
    if not new_servers:
        for srv in stale_cached:
            _signal_reconnect(srv)
        return _existing_tool_names()

    _ensure_mcp_loop()

    async def _discover_all() -> None:
        server_names = list(new_servers.keys())
        results = await asyncio.gather(
            *(_discover_and_register_server(n, cfg) for n, cfg in new_servers.items()),
            return_exceptions=True,
        )
        for name, result in zip(server_names, results):
            if isinstance(result, BaseException):
                message = _format_connect_error(result)
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors[name] = message
                    # 盖冷却戳：下次发现入口不再立刻重试这个坏 server
                    _record_connect_failure(name)
                logger.warning(
                    "Failed to connect to MCP server '%s': %s",
                    name, message,
                )
            else:
                with _lock:
                    _server_connecting.discard(name)
                    _server_connect_errors.pop(name, None)
                    _clear_connect_failure(name)
                logger.info(
                    "MCP server '%s': %d tool(s) registered",
                    name, len(result),
                )

    # 中断保护：临时清当前线程中断标志，避免旧会话残留中断取消发现
    from tools.interrupt import is_interrupted, set_interrupt

    _was_interrupted = is_interrupted()
    if _was_interrupted:
        set_interrupt(False)
    try:
        _run_on_mcp_loop(_discover_all, timeout=120)
    except (TimeoutError, InterruptedError) as exc:
        with _lock:
            stale = [n for n in new_servers if n in _server_connecting]
            if stale:
                logger.warning(
                    "MCP discovery %s while %d server(s) still connecting; "
                    "clearing stale connecting set: %s",
                    "timed out" if isinstance(exc, TimeoutError) else "interrupted",
                    len(stale), ", ".join(stale),
                )
                _server_connecting.difference_update(stale)
                for _sn in stale:
                    _server_connect_errors.setdefault(
                        _sn,
                        f"Connection attempt {'timed out' if isinstance(exc, TimeoutError) else 'interrupted'} during discovery",
                    )
        raise
    finally:
        if _was_interrupted:
            set_interrupt(True)

    return _existing_tool_names()


def discover_mcp_tools() -> List[str]:
    """入口：读配置 → 连接 → 注册工具（对应原版 :6845）。

    安全：``mcp`` 未安装 / 无配置时返回空列表。幂等。
    """
    if not _MCP_AVAILABLE:
        logger.debug("MCP SDK not available -- skipping MCP tool discovery")
        return []
    servers = _load_mcp_config()
    if not servers:
        logger.debug("No MCP servers configured")
        return []
    return register_mcp_servers(servers)


# ── 工具注册（对齐原版 _register_server_tools）──────────────────────────


def _normalize_name_filter(value: Any, label: str) -> Set[str]:
    """把 include/exclude 配置归一化成工具名模式集合（原版 :6002）。

    条目可以是精确工具名或 fnmatch glob（``*_radar_*``）。
    """
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    logger.warning("MCP config %s must be a string or list of strings; ignoring %r", label, value)
    return set()


def matches_name_filter(tool_name: str, patterns: Set[str]) -> bool:
    """tool_name 是否命中模式集（原版 :6019）。

    精确名 O(1) 命中；含 fnmatch 元字符（* ? [）的条目按大小写敏感
    glob 匹配。
    """
    if not patterns:
        return False
    if tool_name in patterns:
        return True
    return any(
        fnmatch.fnmatchcase(tool_name, p)
        for p in patterns
        if "*" in p or "?" in p or "[" in p
    )


def _existing_tool_names() -> List[str]:
    """返回当前已注册的全部 MCP 工具名。"""
    from tools.registry import registry

    names: List[str] = []
    for server in list(_servers.values()):
        names.extend(server._registered_tool_names)
    return names


def _register_server_tools(
    name: str, server: MCPServerTask, config: dict
) -> List[str]:
    """把已连接 server 的工具注册进 registry（对应原版 :6192 精简）。

    Phase 1.5 补齐了原版的注册纪律：
    - include/exclude 过滤（config 的 ``tools.include`` / ``tools.exclude``，
      支持 glob；include 优先于 exclude）；
    - 归一化碰撞 fail-closed：两个不同原名归一化成同一个注册名时，所有
      碰撞项全部跳过并记 error（不挑一个"任意"的 handler）；
    - 跨 toolset 占用保护：名字已被别的 toolset（内置工具 / 其他 MCP
      server）占用时跳过，保留原 owner；
    - 注册后校验 registry 原子所有权（多个 server 并行连接，register
      才是真正的所有权门）。
    """
    from tools.registry import registry

    registered: List[str] = []
    toolset_name = f"mcp-{name}"

    # 选择性工具加载（原版 #690 spec + glob 扩展）
    tools_filter = config.get("tools") or {}
    include_set = _normalize_name_filter(
        tools_filter.get("include"), f"mcp_servers.{name}.tools.include"
    )
    exclude_set = _normalize_name_filter(
        tools_filter.get("exclude"), f"mcp_servers.{name}.tools.exclude"
    )

    def _should_register(tool_name: str) -> bool:
        if include_set:
            return matches_name_filter(tool_name, include_set)
        if exclude_set:
            return not matches_name_filter(tool_name, exclude_set)
        return True

    candidates: List[dict] = []
    for mcp_tool in server._tools:
        if not _should_register(mcp_tool.name):
            logger.debug(
                "MCP server '%s': skipping tool '%s' (filtered by config)",
                name, mcp_tool.name,
            )
            continue
        _scan_mcp_description(
            name, mcp_tool.name,
            getattr(mcp_tool, "description", None) or "",
        )
        schema = _convert_mcp_schema(name, mcp_tool)
        candidates.append(
            {
                "registry_name": schema["name"],
                "origin": f"tool {mcp_tool.name!r}",
                "schema": schema,
                "handler": _make_tool_handler(name, mcp_tool.name, server.tool_timeout),
            }
        )

    # 归一化碰撞预检：相同注册名的多个来源全部跳过（fail-closed）
    unique_candidates: List[dict] = []
    seen_candidates: Set[tuple] = set()
    origins_by_name: Dict[str, Set[str]] = {}
    for candidate in candidates:
        key = (candidate["registry_name"], candidate["origin"])
        if key in seen_candidates:
            continue
        seen_candidates.add(key)
        unique_candidates.append(candidate)
        origins_by_name.setdefault(candidate["registry_name"], set()).add(
            candidate["origin"]
        )

    ambiguous_names = {
        registry_name: sorted(origins)
        for registry_name, origins in origins_by_name.items()
        if len(origins) > 1
    }
    for registry_name, origins in sorted(ambiguous_names.items()):
        logger.error(
            "MCP server '%s': name normalization collision for '%s' from %s; "
            "skipping every colliding entry instead of choosing an arbitrary "
            "handler",
            name, registry_name, ", ".join(origins),
        )

    for candidate in unique_candidates:
        registry_name = candidate["registry_name"]
        if registry_name in ambiguous_names:
            continue

        existing_toolset = registry.get_toolset_for_tool(registry_name)
        if existing_toolset and existing_toolset != toolset_name:
            if existing_toolset.startswith("mcp-"):
                logger.error(
                    "MCP server '%s': %s normalizes to '%s', already owned by "
                    "MCP toolset '%s' — skipping to preserve the existing owner",
                    name, candidate["origin"], registry_name, existing_toolset,
                )
            else:
                logger.warning(
                    "MCP server '%s': %s (→ '%s') collides with built-in tool "
                    "in toolset '%s' — skipping to preserve built-in",
                    name, candidate["origin"], registry_name, existing_toolset,
                )
            continue

        registry.register(
            name=registry_name,
            toolset=toolset_name,
            schema=candidate["schema"],
            handler=candidate["handler"],
            check_fn=None,
            is_async=False,
            description=candidate["schema"]["description"],
        )

        # 预检只是参考：多个 server 并行连接时 registry.register 才是
        # 原子所有权门，注册后校验实际归属。
        if registry.get_toolset_for_tool(registry_name) != toolset_name:
            logger.error(
                "MCP server '%s': registration of %s as '%s' was rejected by "
                "the registry; skipping provenance/count updates",
                name, candidate["origin"], registry_name,
            )
            continue

        registered.append(registry_name)

    if registered:
        registry.register_toolset_alias(name, toolset_name)
    return registered


# ── 工具调用 handler（对齐原版 _make_tool_handler / _get_connected_server_for_call）──


def _get_connected_server_for_call(server_name: str) -> Optional[MCPServerTask]:
    """取可调用的 server（Phase 1.5 无 lazy 分支，原版 :5209 精简）。"""
    with _lock:
        return _servers.get(server_name)


def _sanitize_error(text: str) -> str:
    """错误信息凭据脱敏（对应原版 :526，Phase 1 最小模式集）。

    覆盖常见凭据模式：GitHub PAT、OpenAI sk-、Bearer、token=/key=/
    API_KEY=/password=/secret=。
    """
    patterns = [
        r"gh[pousr]_[A-Za-z0-9_]{10,}",
        r"sk-[A-Za-z0-9_-]{16,}",
        r"Bearer\s+[A-Za-z0-9._~+/=-]{8,}",
        r"(?i)(token|key|api_key|password|secret)\s*[=:]\s*[^\s,;]+",
    ]
    result = str(text)
    for pattern in patterns:
        result = re.sub(pattern, "<redacted>", result)
    return result


def _interrupted_call_result() -> str:
    """中断时的工具返回（对应原版 :4866）。

    与其他错误路径（超时/异常/断线）一样走 tool_error 格式——调用方
    能识别 "error" 字段，不会把中断当成功结果喂给模型。
    """
    return tool_error("MCP call interrupted: user sent a new message")


def _make_tool_handler(server_name: str, tool_name: str, tool_timeout: float):
    """返回符合 registry 派发接口的同步 handler：``handler(args, **kwargs) -> str``。

    对应原版 :5238。流程：熔断 gate → 取连接 → 无 session 先短暂等待重连
    完成，仍失败则请求重建传输（_signal_reconnect）→ loop 上 call_tool
    （rpc_lock 串行）→ 聚合 content/structuredContent → 按结果更新熔断
    计数。Phase 1.5 不含 trust gate / 认证重试（Phase 2/3 补）。
    """

    def _handler(args: dict, **kwargs) -> str:
        # ① 熔断器 gate：连续失败跨过阈值后进入冷却期，明确告诉模型
        #    别重试（防 90 迭代烧预算，#10447）。冷却过后放行一次作为
        #    半开探测；成功路径会重置熔断器。
        if _server_error_counts.get(server_name, 0) >= _CIRCUIT_BREAKER_THRESHOLD:
            opened_at = _server_breaker_opened_at.get(server_name, 0.0)
            age = time.monotonic() - opened_at
            if age < _CIRCUIT_BREAKER_COOLDOWN_SEC:
                remaining = max(1, int(_CIRCUIT_BREAKER_COOLDOWN_SEC - age))
                return tool_error(
                    f"MCP server '{server_name}' is unreachable after "
                    f"{_server_error_counts[server_name]} consecutive "
                    f"failures. Auto-retry available in ~{remaining}s. "
                    f"Do NOT retry this tool yet — use alternative "
                    f"approaches or ask the user to check the MCP server."
                )
            # 冷却已过 → 作为半开探测放行

        # ② 取连接
        server = _get_connected_server_for_call(server_name)
        if not server:
            _bump_server_error(server_name)
            return tool_error(f"MCP server '{server_name}' is not connected")

        # ③ 无 live session：重连可能正在完成（传输异步换入新 session），
        #    先短暂等待，避免瞬时重连窗口烧熔断计数（#26892）。
        if not server.session:
            if _wait_for_server_session_ready(
                server, timeout=min(5.0, float(tool_timeout or 5.0)),
            ):
                pass  # 新 session 已到达，继续
            else:
                # 仍不可用：server 任务可能正在重连或已 park（如 stdio
                # 子进程死亡）。请求 server 任务重建 transport（会 respawn
                # 死亡的子进程），并返回明确提示让模型退避（#16788）。
                _bump_server_error(server_name)
                if _signal_reconnect(server):
                    return tool_error(
                        f"MCP server '{server_name}' transport is down; "
                        f"reconnect requested. Do NOT retry this tool "
                        f"immediately — give it a few seconds to come back."
                    )
                return tool_error(f"MCP server '{server_name}' is not connected")

        async def _call() -> str:
            server.mark_tool_call()
            async with server._rpc_lock:
                result = await server.session.call_tool(
                    tool_name, arguments=args
                )
            # RPC 往返完成 = 传输层可证明健康（即使工具本身 isError），
            # 清快速掉线预算（#62212）。
            _mark_proven = getattr(server, "_mark_session_proven", None)
            if _mark_proven is not None:
                _mark_proven()
            if getattr(result, "isError", False):
                error_text = ""
                for block in (getattr(result, "content", None) or []):
                    text = getattr(block, "text", None)
                    if text:
                        error_text += str(text)
                        continue
                    # EmbeddedResource 块的文本藏在 .resource.text 下
                    res_text = getattr(
                        getattr(block, "resource", None), "text", None
                    )
                    if res_text:
                        error_text += str(res_text)
                return tool_error(
                    _sanitize_error(error_text or "MCP tool returned an error")
                )
            parts: List[str] = []
            for block in (getattr(result, "content", None) or []):
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
            text_result = "\n".join(parts)
            structured = getattr(result, "structuredContent", None)
            if structured is not None:
                if text_result:
                    return json.dumps(
                        {"result": text_result, "structuredContent": structured},
                        ensure_ascii=False,
                    )
                return json.dumps({"result": structured}, ensure_ascii=False)
            return json.dumps({"result": text_result}, ensure_ascii=False)

        def _call_once():
            return _run_on_mcp_loop(_call, timeout=float(tool_timeout))

        try:
            result = _call_once()
            # ④ 按结果更新熔断计数：工具自身报错 / 异常 → bump；
            #    成功 → reset（半开探测成功即闭合）。
            try:
                parsed = json.loads(result)
                if "error" in parsed:
                    _bump_server_error(server_name)
                else:
                    _reset_server_error(server_name)
            except (json.JSONDecodeError, TypeError):
                _reset_server_error(server_name)  # 非 JSON = 成功
            return result
        except InterruptedError:
            return _interrupted_call_result()
        except TimeoutError:
            _bump_server_error(server_name)
            return tool_error(
                f"MCP tool '{tool_name}' timed out after "
                f"{float(tool_timeout):.0f}s"
            )
        except Exception as exc:
            _bump_server_error(server_name)
            return tool_error(_sanitize_error(str(exc)))

    return _handler


# ── 状态查询 / 关闭（对齐原版 get_mcp_status / shutdown_mcp_servers）────


def get_mcp_status() -> List[dict]:
    """返回各 MCP server 的连接状态（对应原版 :6947 精简）。"""
    status: List[dict] = []
    with _lock:
        for name, server in _servers.items():
            status.append(
                {
                    "name": name,
                    "connected": server.session is not None,
                    "tools": len(server._registered_tool_names),
                    "error": str(server._error) if server._error else None,
                    "connect_error": _server_connect_errors.get(name),
                }
            )
    return status


def is_mcp_tool_parallel_safe(tool_name: str) -> bool:
    """MCP 工具是否属于允许并行调用的 server（对应原版 :6929）。"""
    with _lock:
        for server_name in _parallel_safe_servers:
            if tool_name.startswith(
                f"{MCP_TOOL_NAME_PREFIX}{sanitize_mcp_name_component(server_name)}{_MCP_NAME_DELIM}"
            ):
                return True
    return False


def has_registered_mcp_tools() -> bool:
    """是否注册过 MCP 工具（对应原版 :7101）。"""
    with _lock:
        return any(
            s._registered_tool_names for s in _servers.values()
        )


def get_registered_mcp_server_names() -> Set[str]:
    """返回已注册工具的 server 名集合（agent_init 合并工具用）。"""
    with _lock:
        return {
            name
            for name, server in _servers.items()
            if server._registered_tool_names
        }


def refresh_agent_mcp_tools(
    agent,
    *,
    enabled_override=None,
    disabled_override=None,
    quiet_mode: bool = True,
) -> set:
    """按当前 registry 重建 agent 的工具快照（对应原版 :7130 精简版）。

    agent 在构建时快照一次 ``agent.tools`` 就不再读 registry；后台 MCP
    发现慢于构建时，server 连接后其工具对模型不可见。本函数是唯一共享
    的刷新入口：每回合 between-turns 刷新（turn_context）都调它。

    my-hermes 精简点（对照原版）：
    - 原版的 ``_reinject_post_build_tools``（memory-provider / context-engine
      工具）未移植，无需重注入；
    - 无 ``_agent_tools_lock`` / ``_tool_snapshot_generation``（原版防两个
      并发刷新互相覆盖）；my-hermes 刷新只发生在对话循环线程，后台线程
      只写 registry（registry 自带锁），不存在并发发布竞争。

    尊重 agent 构建时的 enabled_toolsets / disabled_toolsets（同一个
    过滤）；按工具**名** diff（数量比较会漏掉等量换入换出）。无变化时
    不触碰快照（避免无谓 churn）。返回新增工具名集合（空 = 无变化）。
    """
    from model_tools import get_tool_definitions
    from tools.registry import registry

    if enabled_override is not None or disabled_override is not None:
        enabled = (
            enabled_override
            if enabled_override is not None
            else getattr(agent, "enabled_toolsets", None)
        )
        disabled = (
            disabled_override
            if disabled_override is not None
            else getattr(agent, "disabled_toolsets", None)
        )
        agent.enabled_toolsets = enabled
        agent.disabled_toolsets = disabled
    else:
        enabled = getattr(agent, "enabled_toolsets", None)
        disabled = getattr(agent, "disabled_toolsets", None)

    new_defs = list(
        get_tool_definitions(
            enabled_toolsets=enabled,
            disabled_toolsets=disabled,
            quiet_mode=quiet_mode,
        )
        or []
    )
    new_names = {t["function"]["name"] for t in new_defs}

    current = {
        t["function"]["name"]
        for t in (getattr(agent, "tools", None) or [])
    }
    if new_names == current:
        # 无变化 → 不触碰快照（无 churn）
        return set()

    # 原子发布：tools / valid_tool_names / _tool_impls 一起换，避免
    # 并发读者看到跨属性半换（对话循环单线程，主要是防御性一致）。
    agent.tools = new_defs
    agent.valid_tool_names = new_names
    agent._tool_impls = {
        entry.name: entry.handler for entry in registry.iter_entries()
    }
    return new_names - current


def shutdown_mcp_servers() -> None:
    """关闭所有 MCP server 并停止后台 loop（对应原版 :7321 精简）。"""
    with _lock:
        servers_snapshot = list(_servers.values())
    if not servers_snapshot:
        _stop_mcp_loop()
        return

    async def _shutdown() -> None:
        results = await asyncio.gather(
            *(server.shutdown() for server in servers_snapshot),
            return_exceptions=True,
        )
        for server, result in zip(servers_snapshot, results):
            if isinstance(result, Exception):
                logger.debug(
                    "Error closing MCP server '%s': %s", server.name, result,
                )
        with _lock:
            _servers.clear()

    with _lock:
        loop = _mcp_loop
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            future.result(timeout=15)
        except BaseException as exc:
            logger.debug("Error during MCP shutdown: %s", exc)
    _stop_mcp_loop()


__all__ = [
    "MCPServerTask",
    "discover_mcp_tools",
    "register_mcp_servers",
    "shutdown_mcp_servers",
    "reconnect_mcp_server",
    "is_mcp_tool_parallel_safe",
    "get_mcp_status",
    "has_registered_mcp_tools",
    "get_registered_mcp_server_names",
    "refresh_agent_mcp_tools",
    "mcp_prefixed_tool_name",
    "sanitize_mcp_name_component",
]

