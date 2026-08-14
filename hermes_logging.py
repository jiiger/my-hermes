"""集中式日志初始化（对齐原版 ``hermes_logging.py`` 的 CLI 最小集）。

原版为 Hermes Agent 提供统一日志入口 ``setup_logging()``，CLI 与网关在
启动早期调用。所有日志文件位于 ``~/.hermes/logs/``（profile 感知，经
:func:`hermes_constants.get_hermes_home`）：

    agent.log   — INFO+，所有 agent/工具/会话活动（主日志）
    errors.log  — WARNING+，仅错误和警告（快速定位）

两个文件均使用 ``RotatingFileHandler`` 轮转，``errors.log`` 容量上限
固定为 2MB×2，``agent.log`` 默认 5MB×3（可经 config.yaml ``logging.*``
覆盖）。

相对原版（hermes_logging.py，800 行）的裁剪项：

- gateway/gui 日志与 ``_ComponentFilter``（my-hermes 无 gateway/gui）；
- 异步队列机制（_ManagedRotatingFileHandler 外部轮转检测、
  _NonFormattingQueueHandler、flush_log_queue/drain_log_queue 等约 300
  行）→ 改为直接 ``logger.addHandler(RotatingFileHandler)``：my-hermes
  是简单 CLI，日志量小，无需把轮转锁移到后台队列线程；
- session context（set_session_context / record factory）→ 日志不含
  ``[session_id]`` 标签；
- RedactingFormatter（原版依赖 agent/redact.py 共 1427 行脱敏系统，
  my-hermes 未移植）→ 改用标准 ``logging.Formatter``，日志不脱敏；
- setup_verbose_logging、_safe_stderr、Windows 锁超时辅助。

依赖：仅 Python 标准库 ``logging``。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

# 模块级初始化标记：setup_logging 幂等（重复调用 no-op，除非 force=True）。
_logging_initialized = False

# 日志行格式。原版带 %(session_tag)s（session context 注入），my-hermes
# 已裁剪 session context，故去掉该字段。
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# 第三方噪音 logger：统一压到 WARNING，避免刷屏 agent.log / 终端。
_NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
)


def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """配置 my-hermes 日志子系统（签名对齐原版 hermes_logging.setup_logging）。

    可安全多次调用——第二次调用为 no-op，除非 ``force=True``。

    Parameters
    ----------
    hermes_home
        Hermes home 目录覆盖。缺省回退到 ``get_hermes_home()``。
    log_level
        ``agent.log`` 文件 handler 的最低级别（``"DEBUG"``/``"INFO"``/
        ``"WARNING"``）。默认 ``"INFO"`` 或 config.yaml ``logging.level``。
    max_size_mb
        单日志文件轮转前的大小上限（MB）。默认 5 或 config.yaml
        ``logging.max_size_mb``。
    backup_count
        保留的轮转备份数。默认 3 或 config.yaml ``logging.backup_count``。
    mode
        调用方上下文：``"cli"``/``"gateway"``/``"gui"``/``"cron"``。
        my-hermes 仅使用 ``"cli"``——gateway/gui 日志分支已裁剪，该参数
        仅保留签名兼容。
    force
        即使已初始化也重新执行。

    Returns
    -------
    Path
        日志写入目录 ``logs/``。
    """
    global _logging_initialized
    home = hermes_home or get_hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 读 config.yaml 默认值（best-effort——启动早期 config 可能尚未加载）。
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # 原版用 agent.redact.RedactingFormatter 给日志脱敏；my-hermes 未移植
    # 该 1427 行模块，改用标准 Formatter（见模块 docstring 裁剪项说明）。
    formatter = logging.Formatter(_LOG_FORMAT)

    root = logging.getLogger()

    # --- agent.log（INFO+）——主活动日志 --------------------------------
    _add_rotating_handler(
        root,
        log_dir / "agent.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backups,
        formatter=formatter,
    )

    # --- errors.log（WARNING+）——快速定位错误/警告 ----------------------
    _add_rotating_handler(
        root,
        log_dir / "errors.log",
        level=logging.WARNING,
        max_bytes=2 * 1024 * 1024,
        backup_count=2,
        formatter=formatter,
    )

    if _logging_initialized and not force:
        return log_dir

    # 保证 root logger 级别足够低，让上面的文件 handler 能触发。
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # 抑制第三方噪音 logger。
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
    return log_dir


def _add_rotating_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> None:
    """给 *logger* 添加 ``RotatingFileHandler``，同路径已存在则跳过（幂等）。

    原版通过异步队列（_register_queued_handler）挂 handler，把轮转锁等待
    移出调用线程；my-hermes 为简单 CLI，直接 addHandler 即可（裁剪项，
    见模块 docstring）。
    """
    resolved = path.resolve()
    for existing in logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # 已挂过同一文件

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _read_logging_config():
    """Best-effort 读 config.yaml 的 ``logging.*``。

    Returns
    -------
    tuple
        ``(level, max_size_mb, backup_count)``——任一可为 ``None``。
    """
    try:
        from hermes_cli.config import read_raw_config

        cfg = read_raw_config() or {}
    except Exception:
        return (None, None, None)
    if cfg:
        log_cfg = cfg.get("logging", {})
        if isinstance(log_cfg, dict):
            return (
                log_cfg.get("level"),
                log_cfg.get("max_size_mb"),
                log_cfg.get("backup_count"),
            )
    return (None, None, None)
