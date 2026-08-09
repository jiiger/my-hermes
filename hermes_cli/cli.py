"""my-hermes 简单 CLI。

用法：
    python cli.py                      # 交互模式（多轮对话，自动保留上下文）
    python cli.py "你的问题"            # 单次查询模式

凭据从环境变量读取：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL

交互模式命令：
    /new        开启新会话（清空上下文）
    /quit       退出（或 Ctrl+C / Ctrl+D）
"""

import os
import sys
from pathlib import Path

# 保证从任意目录运行都能 import 项目模块（项目根目录；editable 安装的
# finder 不覆盖新增的 tools/ 包，故显式把项目根加进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 加载项目根目录的 .env（凭据配置：OPENAI_API_KEY / DEEPSEEK_API_KEY /
# OPENAI_BASE_URL / OPENAI_MODEL）。python-dotenv 已随依赖安装。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / "../.env")

from run_agent import AIAgent  # noqa: E402

try:
    from prompt_toolkit import prompt
    from prompt_toolkit.history import FileHistory

    _HAS_PTK = True
except ImportError:
    _HAS_PTK = False

try:
    from rich.console import Console

    _console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

HISTORY_FILE = str(Path.home() / ".my_hermes_history")


def _print_banner(agent: AIAgent) -> None:
    """打印启动横幅（rich 面板；没有 rich 时退化为纯文本）。"""
    lines = [
        "my-hermes — 简单对话 CLI",
        f"model: {agent.model or '(未设置)'}",
        "输入问题开始对话；/quit 退出，/new 开启新会话",
    ]
    if _HAS_RICH:
        _console.print(f"[bold cyan]{lines[0]}[/bold cyan]")
        _console.print(f"[dim]{lines[1]}[/dim]")
        _console.print(f"[dim]{lines[2]}[/dim]")
    else:
        print("=" * 50)
        print("\n".join(lines))
        print("=" * 50)


def _ask(text: str) -> str:
    """读取一行输入：有 prompt_toolkit 用带历史记录的提示符，否则用 input。"""
    if _HAS_PTK:
        return prompt(
            text,
            history=FileHistory(HISTORY_FILE),
        )
    return input(text)


def _stream_print(text: str) -> None:
    """流式回调：逐段打印（打字机效果，不换行）。"""
    print(text, end="", flush=True)


def interactive(agent: AIAgent) -> None:
    """交互模式：多轮对话，自动把上一轮 messages 传给下一轮。"""
    history = None  # 当前会话的完整消息列表（None = 新会话）
    while True:
        try:
            user_input = _ask("你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "q", "quit", "exit"):
            break
        if user_input.lower() == "/new":
            history = None
            print("（已开启新会话）\n")
            continue

        # 对话：把上一轮的 messages 作为 conversation_history 传入；
        # stream_callback 让最终回答逐段打印（打字机效果），
        # 失败路径不走流式（重试耗尽，无响应可流）
        print("🤖 ", end="", flush=True)
        result = agent.run_conversation(
            user_input,
            conversation_history=history,
            stream_callback=_stream_print,
        )
        print()  # 流式输出结束后换行

        if result["failed"]:
            print(f"❌ {result['final_response']}")

        history = result["messages"]


def once(agent: AIAgent, query: str) -> None:
    """单次查询模式：跑完一轮就退出。"""
    result = agent.run_conversation(query, stream_callback=_stream_print)
    print()  # 流式输出结束后换行
    if result["failed"]:
        print(f"❌ {result['final_response']}")
        sys.exit(1)


def main() -> None:
    # 凭据：环境变量 > .env 文件（load_dotenv 已在模块顶部执行）> 交互输入
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    base_url = os.getenv("OPENAI_BASE_URL") or ""
    model = os.getenv("OPENAI_MODEL") or ""

    if not api_key and not base_url:
        print(
            "未找到凭据。请先设置环境变量：\n"
            "  export OPENAI_API_KEY=sk-xxx\n"
            "  export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 以 DeepSeek 为例\n"
            "  export OPENAI_MODEL=deepseek-chat\n"
        )
        sys.exit(1)

    try:
        agent = AIAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=(os.getenv("OPENAI_PROVIDER") or ""),
        )
    except RuntimeError as exc:
        print(f"❌ Agent 初始化失败: {exc}")
        sys.exit(1)

    _print_banner(agent)

    # 带位置参数 → 单次查询；否则交互模式
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        once(agent, " ".join(sys.argv[1:]))
    else:
        interactive(agent)


if __name__ == "__main__":
    main()
