"""后台 MCP 发现（my-hermes 精简移植版）。

对应原版 hermes_cli/mcp_startup.py（265 行）。核心思路：MCP 连接/发现
（可能耗时数十秒）放到 daemon 后台线程，agent 构建前只 ``join`` 一个
有界短等待（``mcp_discovery_timeout``，默认 1.5s），错过窗口的 server
由每回合的 between-turns 刷新（tools.mcp_tool.refresh_agent_mcp_tools）
自动补进工具快照——启动永远不被 MCP 阻塞。

与原版的裁剪：
- 去掉 ``agent_plugins`` 探测（my-hermes 无插件 MCP）；
- 去掉 ``mcp_oauth`` 交互抑制（my-hermes 无 OAuth，Phase 2/3 预留），
  直接调用 discover_mcp_tools；
- 保留 hermes_home_override 捕获/恢复（hermes_constants 已有对应函数，
  多 profile 进程里后台线程才能发现正确的 mcp_servers）。
"""

from __future__ import annotations

import threading
from typing import Optional

_mcp_discovery_lock = threading.Lock()
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None


def _has_configured_mcp_servers() -> bool:
    """廉价配置探测：没有 mcp_servers 配置的用户避免导入 MCP 栈。"""
    try:
        from hermes_cli.config import read_raw_config

        raw_config = read_raw_config() or {}
        mcp_servers = raw_config.get("mcp_servers")
        if isinstance(mcp_servers, dict) and len(mcp_servers) > 0:
            return True
    except Exception:
        # 探测失败时保守返回 True：后台照常尝试，启动仍不会阻塞。
        return True
    return False


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """为本进程启动一个共享的后台 MCP 发现线程（幂等）。

    若首次后台发现退出时没有任何 server 连上（如启动取消 / OOM 重启），
    后续调用允许重试，而不是永久把进程钉在"已启动但零工具"的状态。
    """
    global _mcp_discovery_started, _mcp_discovery_thread

    with _mcp_discovery_lock:
        if _mcp_discovery_started:
            thread = _mcp_discovery_thread
            if thread is not None and thread.is_alive():
                return
            try:
                from tools.mcp_tool import get_mcp_status

                status = get_mcp_status() or []
                if any(entry.get("connected") for entry in status):
                    return
            except Exception:
                return
            logger.warning(
                "Background MCP discovery previously exited with no connected "
                "servers; retrying discovery thread"
            )
            _mcp_discovery_started = False
            _mcp_discovery_thread = None

        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        # 捕获调用方上下文里的 HERMES_HOME override（多 profile 进程）并在
        # 后台线程内恢复——ContextVars 不会传播进裸线程（原版 #67605）。
        try:
            from hermes_constants import get_hermes_home_override

            home_override = get_hermes_home_override()
        except Exception:
            home_override = None

        def _discover() -> None:
            token = None
            try:
                from hermes_constants import set_hermes_home_override

                token = set_hermes_home_override(home_override)
            except Exception:
                token = None
            try:
                from tools.mcp_tool import discover_mcp_tools

                discover_mcp_tools()
                try:
                    from tools.mcp_tool import get_mcp_status

                    status = get_mcp_status() or []
                    if not any(entry.get("connected") for entry in status):
                        logger.warning(
                            "Background MCP discovery completed with zero connected servers"
                        )
                except Exception:
                    logger.debug(
                        "Failed to inspect MCP status after background discovery",
                        exc_info=True,
                    )
            except Exception:
                logger.debug(
                    "Background MCP tool discovery failed", exc_info=True
                )
            finally:
                if token is not None:
                    try:
                        from hermes_constants import reset_hermes_home_override

                        reset_hermes_home_override(token)
                    except Exception:
                        pass
                with _mcp_discovery_lock:
                    global _mcp_discovery_thread, _mcp_discovery_started
                    _mcp_discovery_thread = None

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        _mcp_discovery_thread = thread
        thread.start()


def _resolve_discovery_timeout(
    explicit: "float | None", *, single_query: bool = False
) -> float:
    """解析 MCP 发现等待上限：显式参数 > 配置 > 默认。

    读 config.yaml 的 ``mcp_discovery_timeout`` / ``mcp_single_query_discovery_timeout``
    （缺省 1.5s / 15s）。懒加载且 fail-safe：配置缺失/非法时回退短安全值，
    保证启动永远不会挂死。
    """
    if explicit is not None:
        return explicit
    key = (
        "mcp_single_query_discovery_timeout"
        if single_query
        else "mcp_discovery_timeout"
    )
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get(key, fallback))
        try:
            raw = (load_config() or {}).get(key, default)
            val = float(raw)
            return val if val > 0 else default
        except Exception:
            return default
    except Exception:
        return fallback


def wait_for_mcp_discovery(
    timeout: "float | None" = None, *, single_query: bool = False
) -> None:
    """等待后台发现（有界）：``thread.join(timeout)`` 在发现完成瞬间返回。

    无 MCP 配置或 server 很快的用户 ~0s 付出；上限只挡住死 server 冻结
    启动。错过窗口的 server 由 between-turns 刷新自动补进快照。
    """
    thread = _mcp_discovery_thread
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def mcp_discovery_in_flight() -> bool:
    """后台发现线程是否仍在运行。"""
    thread = _mcp_discovery_thread
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """阻塞等待后台发现结束（至多 *timeout*）。

    返回 True 表示发现已完成（线程不存在或已退出），False 表示超时仍在跑。
    """
    thread = _mcp_discovery_thread
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def ensure_mcp_discovery_before_agent_build(
    *,
    logger,
    timeout: "float | None" = None,
    single_query: bool = False,
    thread_name: str = "cli-mcp-discovery",
) -> None:
    """agent 构建前给已配置的 MCP 工具一个有界的注册机会。

    ``wait_for_mcp_discovery`` 只 join 已存在的线程；直接到达 agent 构建
    的入口（my-hermes 的 standalone main）需要本函数自足：需要时先启动
    发现，再等到配置上限。失败吞掉——坏配置绝不中止 agent 构建。
    """
    try:
        start_background_mcp_discovery(
            logger=logger,
            thread_name=thread_name,
        )
        wait_for_mcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug(
            "MCP discovery readiness check failed before agent build",
            exc_info=True,
        )
