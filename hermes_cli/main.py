"""my-hermes CLI 入口（对齐原版 hermes_cli/main.py 分发骨架）。

原版 main.py（12690 行）的核心骨架：argparse 子命令体系 +
``set_defaults(func=cmd_xxx)`` 分发 + 启动编排 + 收尾。my-hermes 精简为
三个子命令：

- ``chat``（默认）：交互 / 单次对话（会话实现复用 hermes_cli/cli.py）；
- ``version``：打印版本；
- ``doctor``：环境自检（hermes_cli/doctor.py）。

与 run_agent.main 的关系：run_agent.py 是纯库（AIAgent 类 + 极简入口），
本模块是安装后的正式控制台入口（pyproject [project.scripts]）。
"""

import argparse
import sys


def cmd_chat(args) -> None:
    """交互 / 单次对话（会话实现复用 cli.main，避免逻辑重复）。"""
    from hermes_cli.cli import main as _cli_main

    # cli.main(argv) 内部：凭据解析 → SessionDB → AIAgent → banner →
    # once/interactive → finally agent.close()（含 MCP shutdown 收尾）。
    _cli_main(list(getattr(args, "query", None) or []))


def cmd_version(args) -> None:
    """打印版本：优先已安装包元数据，开发环境回退读 pyproject.toml。"""
    try:
        from importlib.metadata import version

        print(f"hermes-agent {version('my-hermes')}")
        return
    except Exception:
        pass
    try:
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        with open(root / "pyproject.toml", "rb") as fh:
            data = tomllib.load(fh)
        print(f"hermes-agent {data['project']['version']} (dev)")
    except Exception:
        print("hermes-agent (dev)")


def cmd_doctor(args) -> None:
    """环境自检（轻量版，无网络探测）。"""
    from hermes_cli.doctor import run_doctor

    sys.exit(0 if run_doctor() == 0 else 1)


def build_top_level_parser() -> argparse.ArgumentParser:
    """构造顶层 argparse（对齐原版 build_top_level_parser 骨架）。"""
    parser = argparse.ArgumentParser(
        prog="hermes-agent",
        description="my-hermes agent 命令行入口",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    chat_parser = subparsers.add_parser(
        "chat", help="交互对话（默认：无参数直接进入）"
    )
    chat_parser.add_argument(
        "query",
        nargs="*",
        help="单次查询内容（不传则进入交互模式）",
    )
    chat_parser.set_defaults(func=cmd_chat)

    version_parser = subparsers.add_parser("version", help="打印版本")
    version_parser.set_defaults(func=cmd_version)

    doctor_parser = subparsers.add_parser("doctor", help="环境自检")
    doctor_parser.set_defaults(func=cmd_doctor)

    return parser


def main(argv=None) -> None:
    """唯一入口：解析 argv → 分发子命令（对齐原版 main() 语义）。"""
    argv = list(sys.argv[1:] if argv is None else argv)

    # 默认 chat：第一个 token 不是已知子命令 / 帮助选项时，视为 query
    # 前置 "chat"（对齐原版 ``hermes "问题"`` 直接对话的语义）。
    if not argv or argv[0] not in {"chat", "version", "doctor", "-h", "--help"}:
        argv = ["chat"] + argv

    parser = build_top_level_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        # 无 func（例如只有 chat 子命令但分发异常兜底）→ 显示帮助
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
