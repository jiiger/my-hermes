"""环境自检（hermes-agent doctor）—— my-hermes 轻量版。

对应原版 hermes_cli/doctor.py（3005 行）的精简移植：只保留 my-hermes
相关的检查项，输出 ✅/⚠️/❌ 分段报告，不做自动修复（原版 --fix /
--ack 未移植）。

检查项全部复用 my-hermes 既有模块（config / mcp_security / mcp_tool /
model_tools / hermes_state），无网络探测，纯本地诊断。
"""

import importlib.util
import shutil
import sys
from pathlib import Path

_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_RESET = "\033[0m"


def _ok(text: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  {_GREEN}✅{_RESET} {text}{suffix}")


def _warn(text: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  {_YELLOW}⚠️ {_RESET}{text}{suffix}")


def _fail(text: str, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"  {_RED}❌{_RESET} {text}{suffix}")


def _section(title: str) -> None:
    print(f"\n{_CYAN}── {title}{_RESET}")


def _check_dependencies() -> list:
    """必需依赖探测。返回缺失项列表。"""
    deps = {
        "mcp": "MCP SDK（MCP server 支持）",
        "httpx": "HTTP 客户端",
        "yaml": "PyYAML（配置解析）",
        "dotenv": "python-dotenv（.env 加载）",
    }
    missing = []
    for mod, label in deps.items():
        if importlib.util.find_spec(mod) is None:
            missing.append(f"{mod}（{label}）")
    return missing


def _check_external_tools() -> list:
    """MCP/终端依赖的外部工具。返回缺失项列表。"""
    missing = []
    for name in ("git", "node", "npx", "uvx"):
        if shutil.which(name) is None:
            missing.append(name)
    return missing


def run_doctor() -> int:
    """运行全部检查。返回失败项计数（0 = 全部正常）。"""
    failures = 0
    print()
    print(f"{_CYAN}┌──────────────────────────────────────────────┐{_RESET}")
    print(f"{_CYAN}│          🩺 hermes-agent Doctor              │{_RESET}")
    print(f"{_CYAN}└──────────────────────────────────────────────┘{_RESET}")

    # ── Python 环境 ─────────────────────────────────────────────
    _section("Python 环境")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 11):
        _ok(f"Python {py_ver}（要求 ≥ 3.11）")
    else:
        failures += 1
        _fail(f"Python {py_ver}（要求 ≥ 3.11）")

    # ── 依赖 ─────────────────────────────────────────────────────
    _section("必需依赖")
    missing_deps = _check_dependencies()
    if missing_deps:
        failures += len(missing_deps)
        for dep in missing_deps:
            _fail(f"缺少依赖：{dep}（`uv sync` 或 `pip install` 安装）")
    else:
        _ok("mcp / httpx / yaml / dotenv 均已安装")

    # ── 配置与凭据 ───────────────────────────────────────────────
    _section("配置与凭据")
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        _ok("config.yaml 可解析", f"{len(cfg)} 个配置键")
    except Exception as exc:
        failures += 1
        _fail("config.yaml 解析失败", str(exc))

    try:
        from hermes_cli.cli import _resolve_runtime

        api_key, base_url, model, provider = _resolve_runtime()
        if api_key or base_url:
            _ok(
                "API 凭据已配置",
                f"provider={provider or '?'} model={model or '?'}"
                + ("" if api_key else "（仅 base_url，无 key）"),
            )
        else:
            failures += 1
            _fail("未找到 API 凭据（环境变量 / .env / config.yaml）")
    except Exception as exc:
        failures += 1
        _fail("凭据解析异常", str(exc))

    # ── MCP 配置安全 + 连接状态 ──────────────────────────────────
    _section("MCP")
    try:
        from hermes_cli.config import load_config as _lc
        from hermes_cli.mcp_security import validate_mcp_server_entry

        servers = (_lc() or {}).get("mcp_servers") or {}
        if not servers:
            _ok("未配置 mcp_servers")
        else:
            suspicious = 0
            for name, entry in servers.items():
                issues = validate_mcp_server_entry(name, entry)
                if issues:
                    suspicious += 1
                    _fail(f"server '{name}' 命中安全拦截", "; ".join(issues))
            if suspicious == 0:
                _ok(f"{len(servers)} 个 MCP server 配置通过安全校验")
    except Exception as exc:
        failures += 1
        _fail("MCP 安全校验异常", str(exc))

    try:
        from tools.mcp_tool import get_mcp_status

        status = get_mcp_status() or []
        if not status:
            _warn("MCP 未连接（正常：启动时后台发现，可运行一次对话后再查）")
        else:
            for s in status:
                if s.get("connected"):
                    _ok(f"server '{s['name']}' 已连接", f"{s['tools']} 个工具")
                else:
                    _warn(
                        f"server '{s['name']}' 未连接",
                        s.get("connect_error") or s.get("error") or "后台重连中",
                    )
    except Exception:
        pass  # mcp_tool 未导入时跳过（无 MCP 使用）

    # ── 外部工具 ─────────────────────────────────────────────────
    _section("外部工具")
    missing_tools = _check_external_tools()
    if missing_tools:
        _warn(f"以下工具缺失（影响对应 MCP/终端功能）：{', '.join(missing_tools)}")
    else:
        _ok("git / node / npx / uvx 均可用")

    # ── 工具注册 ─────────────────────────────────────────────────
    _section("工具注册")
    try:
        import model_tools

        toolsets = model_tools.get_available_toolsets()
        total = sum(len(v["tools"]) for v in toolsets.values())
        _ok(f"已注册 {total} 个工具", f"工具集：{', '.join(sorted(toolsets))}")
    except Exception as exc:
        failures += 1
        _fail("工具注册查询失败", str(exc))

    # ── 会话数据库 ───────────────────────────────────────────────
    _section("会话数据库")
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        _ok("SessionDB 可打开")
        try:
            db.close()
        except Exception:
            pass
    except Exception as exc:
        failures += 1
        _fail("SessionDB 不可用", str(exc))

    # ── 汇总 ─────────────────────────────────────────────────────
    print()
    if failures:
        print(
            f"  {_RED}✗ {failures} 项检查未通过{_RESET}"
            "（详见上方 ❌/⚠️ 条目）"
        )
    else:
        print(f"  {_GREEN}✓ 全部检查通过{_RESET}")
    print()
    return failures
