"""my-hermes 简单 CLI。

用法：
    python cli.py                      # 交互模式（多轮对话，自动保留上下文）
    python cli.py "你的问题"            # 单次查询模式

参数解析优先级（对齐原版 hermes 的"config.yaml 为中心"）：
    1. config.yaml：model.default / model.provider / model.base_url；
       api_key 按 providers.<provider>.key_env（环境变量名）→
       providers.<provider>.api_key（硬编码）→ <PROVIDER>_API_KEY 环境变量
       依次解析（.env 里只放 API key，模型/端点配置都在 config.yaml）
    2. 向后兼容兜底：OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_PROVIDER /
       OPENAI_API_KEY / DEEPSEEK_API_KEY / OPENROUTER_API_KEY 环境变量

交互模式命令：
    /new        开启新会话（清空上下文）
    /quit       退出（或 Ctrl+C / Ctrl+D）
"""

import argparse
import os
import select
import sys
import threading
from pathlib import Path
from typing import List, Optional

# 保证从任意目录运行都能 import 项目模块（项目根目录；editable 安装的
# finder 不覆盖新增的 tools/ 包，故显式把项目根加进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 加载项目根目录的 .env（凭据配置：OPENAI_API_KEY / DEEPSEEK_API_KEY /
# OPENAI_BASE_URL / OPENAI_MODEL）。python-dotenv 已随依赖安装。
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent / "../.env")

from hermes_constants import get_hermes_home  # noqa: E402
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


def _read_stdin_available() -> str:
    """非阻塞读取 stdin 当前可用的输入（对话期间打断用，原版输入线程思路）。

    仅在有数据时被调用（select 已确认可读）；读取失败返回空串。
    """
    try:
        data = os.read(sys.stdin.fileno(), 4096)
    except (OSError, ValueError):
        return ""
    return data.decode("utf-8", errors="replace").strip()


def _api_key_env_for_provider(provider: str) -> str:
    """按原版约定把 provider 名映射为 api_key 环境变量名。

    规则：大写 + 连字符转下划线 + ``_API_KEY``（对齐原版 runtime_provider
    按 hostname 推断 key 变量的思路，如 opencode-go → OPENCODE_GO_API_KEY、
    deepseek → DEEPSEEK_API_KEY）。
    """
    if not provider:
        return ""
    return provider.upper().replace("-", "_") + "_API_KEY"


def _resolve_runtime_from_config() -> tuple[str, str, str, str]:
    """从 config.yaml 读取运行参数 (model, provider, base_url, api_key)。

    对齐原版"config.yaml 为中心"的解析（runtime_provider 的简化版）：
    - model.default（别名 model.model）→ model
    - model.provider → provider
    - model.base_url → base_url
    - api_key 依次尝试：
      1. providers.<provider>.key_env（别名 api_key_env）指向的环境变量
      2. providers.<provider>.api_key（config 硬编码，key_env 缺失时兜底）
      3. 按 provider 推断 <PROVIDER>_API_KEY 环境变量（config 无 providers
         段时，如 opencode-go → OPENCODE_GO_API_KEY）
    读取失败/无配置时返回四个空字符串。
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        model_cfg = cfg.get("model") or {}
        model = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
        provider = str(model_cfg.get("provider") or "").strip().lower()
        base_url = str(model_cfg.get("base_url") or "").strip()

        api_key = ""
        providers_cfg = cfg.get("providers")
        entry = providers_cfg.get(provider) if isinstance(providers_cfg, dict) else None
        if isinstance(entry, dict):
            key_env = str(
                entry.get("key_env") or entry.get("api_key_env") or ""
            ).strip()
            if key_env:
                api_key = os.getenv(key_env) or ""
            if not api_key:
                api_key = str(entry.get("api_key") or "").strip()
        if not api_key:
            api_key = os.getenv(_api_key_env_for_provider(provider)) or ""
        return model, provider, base_url, api_key
    except Exception:
        return "", "", "", ""


def _resolve_runtime() -> tuple[str, str, str, str]:
    """解析运行参数 (api_key, base_url, model, provider)。

    对齐原版优先级：
    - model / provider / base_url：config.yaml（model 段）优先，
      OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_PROVIDER 环境变量兜底
      （my-hermes 旧 .env 习惯的向后兼容）；
    - api_key：config 解析结果（key_env → inline → <PROVIDER>_API_KEY）
      优先，OPENAI_API_KEY / DEEPSEEK_API_KEY / OPENROUTER_API_KEY 兜底。
    附带加载 HERMES_HOME/.env（用户级凭据文件，key_env 与
    <PROVIDER>_API_KEY 的值常来自这里）。
    """
    # 补加载用户级 .env（HERMES_HOME/.env，若存在）。项目根 .env 已在
    # 模块顶部加载；用户级文件按原版语义优先（override=True）。
    user_env = get_hermes_home() / ".env"
    if user_env.exists():
        load_dotenv(user_env, override=True)

    cfg_model, cfg_provider, cfg_base_url, cfg_api_key = _resolve_runtime_from_config()
    model = cfg_model or os.getenv("OPENAI_MODEL") or ""
    provider = cfg_provider or os.getenv("OPENAI_PROVIDER") or ""
    base_url = cfg_base_url or os.getenv("OPENAI_BASE_URL") or ""
    api_key = cfg_api_key or (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    ) or ""
    return api_key, base_url, model, provider


def interactive(agent: AIAgent) -> None:
    """交互模式：多轮对话，自动把上一轮 messages 传给下一轮。

    对话在后台线程执行、主线程监听键盘（对齐原版"输入线程调
    interrupt"的模式）：输入新内容或按 Ctrl+C 会调用
    ``agent.request_interrupt()`` 优雅中断当前轮（流式/工具执行停止），
    新输入立即作为下一轮问题；被中断轮的不完整消息不回填对话历史。
    """
    history = None  # 当前会话的完整消息列表（None = 新会话）
    pending_input = None  # 打断时读到的新输入，作为下一轮问题
    while True:
        if pending_input is None:
            try:
                user_input = _ask("你 > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 再见")
                break
        else:
            user_input = pending_input
            pending_input = None

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
        before_history = history
        result_holder: dict = {}
        interrupted = {"flag": False}

        def _run():
            result_holder["result"] = agent.run_conversation(
                user_input,
                conversation_history=before_history,
                stream_callback=_stream_print,
            )

        worker = threading.Thread(target=_run, daemon=True, name="cli-conversation")
        worker.start()
        # 主线程监听键盘：有新输入或 Ctrl+C → 请求优雅中断当前轮
        while worker.is_alive():
            try:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    text = _read_stdin_available()
                    if text:
                        agent.request_interrupt()
                        interrupted["flag"] = True
                        pending_input = text
                        break
            except KeyboardInterrupt:
                # Ctrl+C：第一次请求优雅中断，不退出（输入阶段仍由 _ask 处理退出）
                agent.request_interrupt()
                interrupted["flag"] = True
                break
            except (OSError, ValueError):
                # 平台不支持 stdin select（如 Windows）→ 退化为纯 Ctrl+C 打断
                pass
        worker.join()
        print()  # 流式输出结束后换行

        result = result_holder.get("result")
        if interrupted["flag"]:
            # 被中断轮的不完整消息不回填历史，避免污染下一轮上下文
            history = before_history
            print("⚡ 已中断当前回复。")
        elif result and result.get("failed"):
            print(f"❌ {result['final_response']}")
            history = result.get("messages") or history
        else:
            history = (result or {}).get("messages") or history


def once(agent: AIAgent, query: str) -> None:
    """单次查询模式：跑完一轮就退出。"""
    result = agent.run_conversation(query, stream_callback=_stream_print)
    print()  # 流式输出结束后换行
    if result["failed"]:
        print(f"❌ {result['final_response']}")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    # argparse 解析（对应原版 hermes_cli/main.py 的 argparse 体系）：
    # --help 在任何凭据检查之前由 argparse 处理并 SystemExit(0)，
    # 因此帮助信息不依赖 API 凭据。
    parser = argparse.ArgumentParser(
        prog="hermes-agent",
        description="my-hermes 简单对话 CLI",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="单次查询内容（不传则进入交互模式）",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # 参数解析：环境变量/.env > config.yaml（_resolve_runtime 内部实现）
    api_key, base_url, model, provider = _resolve_runtime()

    if not api_key and not base_url:
        print(
            "未找到凭据。请先设置环境变量：\n"
            "  export OPENAI_API_KEY=sk-xxx\n"
            "  export OPENAI_BASE_URL=https://api.deepseek.com/v1   # 以 DeepSeek 为例\n"
            "  export OPENAI_MODEL=deepseek-chat\n"
            "或在 ~/.hermes/config.yaml 配置 model 段与 providers.<name> 段\n"
        )
        sys.exit(1)

    try:
        agent = AIAgent(
            api_key=api_key,
            base_url=base_url,
            model=model,
            provider=provider,
        )
    except RuntimeError as exc:
        print(f"❌ Agent 初始化失败: {exc}")
        sys.exit(1)

    _print_banner(agent)

    # 带位置参数 → 单次查询；否则交互模式
    if args.query:
        once(agent, " ".join(args.query))
    else:
        interactive(agent)


if __name__ == "__main__":
    main()
