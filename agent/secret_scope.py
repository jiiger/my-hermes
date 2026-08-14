"""按 profile 作用域的凭据解析（精简移植版）。

对应原版 hermes-agent 的 agent/secret_scope.py（293 行）。多 profile 网关
（multiplexing gateway）在同一个进程里服务多个 profile，每个 profile 有
自己的 ``.env``（自己的 provider key 与平台 token），所以**不能**把它们
并进进程全局的 ``os.environ``——那会把 profile A 的密钥泄漏给 profile B
的回合，也会泄漏给每个以 ``env=dict(os.environ)`` 派生的子进程。

本模块提供一个 fail-closed（失败即报错）的 context-local 密钥作用域：

- ``set_secret_scope(mapping)`` 为当前任务安装活动 profile 的密钥
  （contextvar，经 ``copy_context()`` 与 HERMES_HOME override 一样
  传播进 agent 的工作线程）；
- ``get_secret(name)`` 从该作用域读取。当 multiplexing **开启**且未安装
  作用域时，它**抛异常**而不是静默回退到 ``os.environ``——未迁移或新增
  的调用点会在那一行大声失败，而不是泄漏另一个 profile 的值；当
  multiplexing **关闭**（默认）时，它透明地读 ``os.environ``，单 profile
  网关与所有非网关调用方的行为与从前完全一致。

my-hermes 现状：无多 profile 网关，multiplexing 恒为关闭态，本模块主要
为 hindsight 等插件提供统一的 ``get_secret`` 入口（读 ``os.environ``，
行为等价 ``os.getenv``）。完整移植以对齐原版接口，未来接入网关即可用。
"""
from __future__ import annotations

import os
import re
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Dict, Mapping, Optional


# ── multiplexing 开关 ───────────────────────────────────────────────────
# 进程级全局：网关启动时若 gateway.multiplex_profiles 为真则置位一次。
# 决定 get_secret() 在无作用域读取时是否 fail-closed。
# 用普通模块全局（非 contextvar）：描述的是部署模式，不是每任务的值。
_MULTIPLEX_ACTIVE: bool = False


def set_multiplex_active(active: bool) -> None:
    """标记进程是否作为 profile 复用器（multiplexer）运行。

    网关启动时调用一次。为 True 时，``get_secret`` 在无作用域读取时
    fail-closed（抛异常）而不是回退到 ``os.environ``。
    """
    global _MULTIPLEX_ACTIVE
    _MULTIPLEX_ACTIVE = bool(active)


def is_multiplex_active() -> bool:
    """返回进程是否作为 profile 复用器运行。"""
    return _MULTIPLEX_ACTIVE


# ── 密钥作用域 contextvar ───────────────────────────────────────────────
_SECRET_SCOPE: ContextVar[Optional[Mapping[str, str]]] = ContextVar(
    "_SECRET_SCOPE", default=None
)


class UnscopedSecretError(RuntimeError):
    """multiplex 模式下未安装作用域就读密钥时抛出。

    这是 fail-closed 信号：意味着一次凭据读取到达 ``get_secret`` 时没有
    profile 作用域在生效——在复用器里，否则会泄漏恰好残留在
    ``os.environ`` 里的某个 profile 的值。修复方式是给调用路径包上
    ``set_secret_scope(...)``（每回合 / 每适配器的 profile 作用域），
    而不是放宽豁免名单。
    """


def set_secret_scope(secrets: Optional[Mapping[str, str]]) -> Token:
    """为当前上下文安装活动 profile 的密钥映射。

    返回用于 ``reset_secret_scope`` 的 token。传 ``None`` 表示清除。
    """
    return _SECRET_SCOPE.set(secrets)


def reset_secret_scope(token: Token) -> None:
    """恢复上一个密钥作用域。"""
    _SECRET_SCOPE.reset(token)


def current_secret_scope() -> Optional[Mapping[str, str]]:
    """返回当前生效的密钥映射；未安装作用域时返回 None。"""
    return _SECRET_SCOPE.get()


# ── 真正的进程级 env vars（不是按 profile 的密钥）────────────────────────
# 这些是进程/部署级设置，不是 profile 凭据。它们名正言顺地待在
# os.environ 里，即使在 multiplex 模式下也必须继续从那里读——把她们
# 塞进 fail-closed 路径会错误崩溃。凡是匹配的变量一律无视作用域直接读
# os.environ。
#
# 成员判定按精确名或前缀（见 _is_global_env）。名单要克制：拿不准时
# 一律当 profile 密钥处理。
_GLOBAL_ENV_EXACT = frozenset({
    # Hermes 运行时 / 部署
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_GATEWAY_LOCK_DIR",
    "HERMES_MAX_ITERATIONS", "HERMES_MAX_TOKENS", "HERMES_API_TIMEOUT",
    "HERMES_REDACT_SECRETS", "HERMES_NOUS_TIMEOUT_SECONDS",
    "_HERMES_GATEWAY",
    # OS / 解释器
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "PWD", "SHELL", "TMPDIR",
    "VIRTUAL_ENV", "PYTHONPATH", "SSL_CERT_FILE",
    # Kanban 路径（按看板，而非按 profile 的密钥）
    "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_BOARD",
    # API-server LISTENER 设置——部署配置（Docker compose ``environment:``
    # 块、systemd ``Environment=``），不是 profile 密钥。作用域化的 runner
    # 重载必须继续看到它们，否则容器部署会悄悄丢掉 api_server 平台。
    # 注意：API_SERVER_KEY 故意不在这里——它确实是凭据，保持 profile 作用域。
    "API_SERVER_ENABLED", "API_SERVER_HOST", "API_SERVER_PORT",
    "API_SERVER_CORS_ORIGINS",
})
_GLOBAL_ENV_PREFIXES = (
    "HERMES_KANBAN_",
    "HERMES_TELEGRAM_",   # 调参旋钮（批处理延迟、回退开关）——不是 token
    "TERMINAL_",          # terminal/sandbox 后端设置
)


def _is_global_env(name: str) -> bool:
    """返回该环境变量是否属于真正的进程级（非 profile 密钥）变量。"""
    if name in _GLOBAL_ENV_EXACT:
        return True
    return any(name.startswith(p) for p in _GLOBAL_ENV_PREFIXES)


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """按环境变量名解析凭据，尊重当前生效的 profile 作用域。

    解析顺序：

    1. 真正的进程级变量（``_is_global_env``）一律读 ``os.environ``——
       它们是部署设置，不是 profile 密钥。
    2. 已安装密钥作用域（multiplex 回合）：从作用域读。multiplex 开启时
       作用域是权威来源——键缺失返回 ``default``，且**不**回退
       ``os.environ``，因为复用器里 ``os.environ`` 可能装着别的 profile
       的值。multiplex 关闭时，作用域未命中回退 ``os.environ``：单
       profile 部署合法地通过进程环境提供凭据（systemd ``Environment=``、
       secret-manager 包装器如 ``pass-cli run`` / ``op run``、裸 shell
       export），而作用域——例如无条件包在每次 cron 任务外——必须只是
       ``.env`` 的叠加层，不是眼罩。
    3. 未安装作用域：
       - multiplex 关闭（默认部署）：读 ``os.environ``——与所有调用方
         之前的 ``os.getenv`` 行为完全一致。
       - multiplex 开启：FAIL CLOSED。抛 ``UnscopedSecretError``，让缺失
         作用域被大声捕获，而不是泄漏跨 profile 的值。
    """
    if _is_global_env(name):
        val = os.environ.get(name)
        return val if val is not None else default

    scope = _SECRET_SCOPE.get()
    if scope is not None:
        val = scope.get(name)
        if val is not None:
            return val
        if _MULTIPLEX_ACTIVE:
            return default
        # multiplex 关闭：作用域是进程环境的叠加层，不是隔离边界——
        # 没有别的 profile 可泄漏。没有这个回退，只在进程环境里注入的
        # 凭据在任何 set_secret_scope(...) 块内（cron 调度器给每个任务
        # 都装一个）会凭空消失：cron 任务发出占位 API key 而 401，交互
        # 回合却一直正常。
        val = os.environ.get(name)
        return val if val is not None else default

    if _MULTIPLEX_ACTIVE:
        raise UnscopedSecretError(
            f"get_secret({name!r}) called with no profile secret scope active "
            f"while multiplexing is on. This credential read must run inside a "
            f"set_secret_scope(...) block (the per-turn / per-adapter profile "
            f"scope). Reading os.environ here would risk leaking another "
            f"profile's value. See docs/design/multiplexing-gateway.md "
            f"(Workstream A)."
        )

    val = os.environ.get(name)
    return val if val is not None else default


def _strip_inline_comment(value: str) -> str:
    """从原始 ``.env`` 值里剥掉 dotenv 风格的行内注释。

    与 python-dotenv (1.2.2) 语义一致（经验证）：

    - 带引号的值：扫描配对的闭合引号（双引号感知 ``\\`` 转义，因为
      ``save_env_value`` 写出 ``\\"`` / ``\\\\`` 转义）。闭合引号之前
      的内容全部保留；其后剩余的 ``# ...`` 丢弃，故
      ``KEY="has # inside" # trailing`` 得 ``has # inside``。非注释的
      尾随垃圾不动原值（宽松，不像 dotenv 直接硬报错）。
    - 不带引号的值：只在**前面有空白**的 ``#`` 处截断，所以
      ``KEY=foo#bar`` 保留 ``foo#bar``，而 ``KEY=value # comment`` 得
      ``value``。以 ``#`` 开头的值（``KEY=#leading``）原样保留。
    """
    value = value.strip()
    if not value:
        return value
    quote = value[0]
    if quote in ("'", '"'):
        i = 1
        while i < len(value):
            ch = value[i]
            if quote == '"' and ch == "\\":
                i += 2  # 跳过被转义的字符
                continue
            if ch == quote:
                remainder = value[i + 1:].lstrip()
                if remainder.startswith("#"):
                    return value[: i + 1]
                return value
            i += 1
        return value  # 未闭合引号：原样保留
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(env_path: Path) -> Dict[str, str]:
    """把 ``.env`` 文件解析成纯 dict，**不**触碰 ``os.environ``。

    用于把 profile 的密钥载入隔离映射，交给 ``set_secret_scope``。
    只解析 Hermes 自己写出的那撮 KEY=VALUE 子集（``export`` 前缀、
    ``#`` 注释——整行与 dotenv 兼容的行内注释、与写入端 ``\\"`` /
    ``\\\\`` 转义对称的引号配对——与 ``hermes_cli.config._parse_env_value``
    同语义），但绝不改动进程环境——隔离正是本函数的意义。

    编码用 ``utf-8-sig``：Windows 记事本 / PowerShell
    ``Set-Content -Encoding UTF8`` 写出的前导 UTF-8 BOM 不会把第一个键
    变成 ``\\ufeffNAME``，导致作用域内 ``get_secret('NAME')`` 落空。
    """
    secrets: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8-sig")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return secrets

    # 用 Hermes 官方解析器解析值：save_env_value 在双引号内转义
    # " 和 \，所有其他读取方（load_env、python-dotenv）反向还原这些
    # 转义。只剥外层引号会破坏含 " 或 \ 的凭据——交互时正常，作用域
    # （cron / multiplex）解析时坏掉。
    from hermes_cli.config import _parse_env_value

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        secrets[key] = _parse_env_value(_strip_inline_comment(value))

    return secrets


def build_profile_secret_scope(hermes_home: Path) -> Dict[str, str]:
    """从 profile 的 ``<home>/.env`` 构建其密钥映射。

    返回全新 dict（可安全交给 ``set_secret_scope``）。真正的进程级变量
    刻意不拷贝进来——``get_secret`` 直接读 ``os.environ``，作用域只装
    profile 密钥。
    """
    home = Path(hermes_home)
    secrets = load_env_file(home / ".env")

    try:
        from hermes_cli.env_loader import get_secret_source_values
        external_secrets = get_secret_source_values(home)
    except Exception:
        external_secrets = {}

    for key, value in external_secrets.items():
        if _is_global_env(key):
            continue
        secrets[key] = value

    return secrets
