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
import time
from pathlib import Path
from typing import Callable, List, Optional

from agent.interrupt_compat import request_hard_interrupt

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


# =========================================================================
# 状态栏（对齐原版 cli.py 的 _build_status_bar_text / _get_status_bar_fragments）
#
# 原版在 prompt_toolkit TUI 底部渲染一行实时状态栏，典型形态：
#   ⚕ deepseek-v4-flash │ ctx -- │ [░░░░░░░░░░] -- │ 4s │ ⏲ 0s
# my-hermes 没有完整 TUI，拆成两部分：
#   - 输入阶段：prompt_toolkit 的 bottom_toolbar（冻结态 ⏲ / ✓ idle）；
#   - 回复期间：轻量状态栏线程实时刷新（live ⏱），开始流式输出即清除。
# =========================================================================


def _format_duration_compact(seconds: float) -> str:
    """紧凑时长格式（对齐原版 agent/usage_pricing.py:format_duration_compact）。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    if hours < 24:
        remaining_min = int(minutes % 60)
        return f"{int(hours)}h {remaining_min}m" if remaining_min else f"{int(hours)}h"
    days = hours / 24
    return f"{days:.1f}d"


def _format_token_count_compact(value: int) -> str:
    """紧凑 token 数格式（对齐原版 agent/usage_pricing.py:format_token_count_compact）。"""
    abs_value = abs(int(value))
    if abs_value < 1_000:
        return str(int(value))

    sign = "-" if value < 0 else ""
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = abs_value / threshold
            if scaled < 10:
                text = f"{scaled:.2f}"
            elif scaled < 100:
                text = f"{scaled:.1f}"
            else:
                text = f"{scaled:.0f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return f"{sign}{text}{suffix}"

    return f"{value:,}"


def _format_context_length(tokens: int) -> str:
    """上下文窗口长度格式（对齐原版 hermes_cli/banner.py:_format_context_length）。"""
    if tokens >= 1_000_000:
        val = tokens / 1_000_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}M"
        return f"{val:.1f}M"
    elif tokens >= 1_000:
        val = tokens / 1_000
        rounded = round(val)
        if abs(val - rounded) < 0.05:
            return f"{rounded}K"
        return f"{val:.1f}K"
    return str(tokens)


def _build_context_bar(percent_used: Optional[int], width: int = 10) -> str:
    """上下文占用进度条（对齐原版 cli.py:_build_context_bar）。"""
    safe_percent = max(0, min(100, percent_used or 0))
    filled = round((safe_percent / 100) * width)
    return f"[{('█' * filled) + ('░' * max(0, width - filled))}]"


def _format_prompt_elapsed(
    prompt_start_time: Optional[float],
    prompt_duration: float,
    live: bool = False,
) -> str:
    """当前轮耗时标签（对齐原版 cli.py:_format_prompt_elapsed）。

    进行中（live）用 ⏱，冻结/未开始用 ⏲；始终返回字符串，未开始时显示
    ``⏲ 0s``。
    """
    if prompt_start_time is None and prompt_duration == 0.0:
        return "⏲ 0s"
    elapsed = time.time() - prompt_start_time if prompt_start_time is not None else prompt_duration
    elapsed = max(0.0, elapsed)

    days = int(elapsed // 86400)
    remaining = elapsed % 86400
    hours = int(remaining // 3600)
    remaining = remaining % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)

    if days > 0:
        time_str = f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s" if seconds else f"{hours}h {minutes}m"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s" if seconds else f"{minutes}m"
    else:
        time_str = f"{int(elapsed)}s"

    emoji = "⏱" if live else "⏲"
    return f"{emoji} {time_str}"


def _format_idle_since(last_finished_at: Optional[float], turn_live: bool) -> str:
    """距上一次回复结束的时长标签（对齐原版 cli.py:_format_idle_since）。

    回复进行中或从未结束过回复时返回空串；否则 ``✓ 42s`` / ``✓ 3m``。
    """
    if turn_live or last_finished_at is None:
        return ""
    idle = max(0.0, time.time() - last_finished_at)
    return f"✓ {_format_duration_compact(idle)}"


def _context_snapshot(agent: AIAgent) -> tuple[Optional[int], Optional[int]]:
    """从 agent.context_compressor 读取上下文占用 ``(tokens, context_length)``。

    无 compressor、无有效 context_length、或尚未记录到用量（tokens 为 0）
    时返回 ``(None, None)``（状态栏显示 ``ctx --`` / ``--``，对齐原版无
    数据时的形态；避免出现 ``0/200K 0%`` 这种误导性恒值）。
    """
    comp = getattr(agent, "context_compressor", None)
    if comp is None:
        return None, None
    length = getattr(comp, "context_length", 0) or 0
    if not length:
        return None, None
    tokens = getattr(comp, "last_prompt_tokens", 0) or 0
    if tokens <= 0:
        return None, None
    return tokens, length


def _build_status_bar_text(
    model: str,
    *,
    session_start: float,
    prompt_start_time: Optional[float] = None,
    prompt_duration: float = 0.0,
    last_turn_finished_at: Optional[float] = None,
    context_tokens: Optional[int] = None,
    context_length: Optional[int] = None,
) -> str:
    """构建一行状态栏文本（对齐原版 cli.py:_build_status_bar_text 宽格式）。

    形态：``⚕ {model} │ {ctx} │ [{bar}] {pct} │ {duration} │ {elapsed} │ {idle}``。
    """
    # 模型短名：去掉 provider 前缀，超长截断（对齐原版）
    model_short = model.split("/")[-1] if "/" in model else model
    if model_short.endswith(".gguf"):
        model_short = model_short[:-5]
    if len(model_short) > 26:
        model_short = f"{model_short[:23]}..."

    # 上下文标签：有 context_length 时 ``used/total``，否则 ``ctx --``
    if context_length:
        ctx_total = _format_context_length(context_length)
        ctx_used = _format_token_count_compact(context_tokens or 0)
        context_label = f"{ctx_used}/{ctx_total}"
    else:
        context_label = "ctx --"

    # 进度条 + 百分比：无数据时全空条 + ``--``
    percent = None
    if context_length and context_tokens is not None:
        percent = round((context_tokens / context_length) * 100)
    percent_label = f"{percent}%" if percent is not None else "--"
    bar = _build_context_bar(percent)

    duration_label = _format_duration_compact(time.time() - session_start)
    prompt_label = _format_prompt_elapsed(
        prompt_start_time, prompt_duration, live=prompt_start_time is not None
    )

    parts = [
        f"⚕ {model_short}",
        context_label,
        f"{bar} {percent_label}",
        duration_label,
        prompt_label,
    ]
    idle_label = _format_idle_since(
        last_turn_finished_at, turn_live=prompt_start_time is not None
    )
    if idle_label:
        parts.append(idle_label)
    return " │ ".join(parts)


class _StatusBarThread:
    """回复等待期间在终端独占一行实时刷新的状态栏线程。

    渲染约定（对齐原版 TUI 底部工具栏的效果，但保持纯 print 架构）：
    - 状态栏独占一行：渲染时 ``text + \\n``，刷新时先 ``\\x1b[1A`` 回移
      一行覆盖状态栏行，因此光标始终停在状态栏下方、其他输出不会与
      状态栏粘连；
    - 其他输出到来时调用 ``print_above(text)``：先清掉状态栏行再输出
      （仿原版 KawaiiSpinner.print_above），状态栏在下一帧于光标处重绘；
    - 开始流式输出时调用 ``stop()`` 清除并停止，避免与打字机输出互相
      干扰。

    非 TTY（管道/日志）时不渲染、不写转义序列，透明退化为无状态栏。
    """

    def __init__(
        self,
        text_fn: Callable[[], str],
        interval: float = 0.2,
        out=None,
    ):
        self._text_fn = text_fn
        self._interval = max(0.05, interval)
        self._out = out or sys.stdout
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._has_bar_line = False  # 当前是否有一行状态栏（供清行/回移判断）

    @property
    def _is_tty(self) -> bool:
        """输出是否为真实终端（非 TTY 跳过动画，避免转义序列进管道）。"""
        try:
            return hasattr(self._out, "isatty") and self._out.isatty()
        except (ValueError, OSError):
            return False

    def start(self) -> None:
        """启动渲染线程（已在运行则忽略）。"""
        if not self._is_tty:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._has_bar_line = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cli-status-bar"
        )
        self._thread.start()

    def stop(self, clear: bool = True) -> None:
        """停止渲染线程并（默认）清除最后一行。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 0.2)
            self._thread = None
        if clear:
            self.clear()

    def clear(self) -> None:
        """清除状态栏行（若有），光标回到状态栏行行首。"""
        with self._lock:
            try:
                if self._has_bar_line and self._is_tty:
                    self._out.write("\x1b[1A\r\x1b[K")
                    self._has_bar_line = False
                self._out.flush()
            except (OSError, ValueError):
                pass

    def print_above(self, text: str) -> None:
        """在状态栏上方输出一行（其他输出到来时调用，避免粘连）。"""
        with self._lock:
            try:
                if self._has_bar_line and self._is_tty:
                    self._out.write("\x1b[1A\r\x1b[K")
                    self._has_bar_line = False
                self._out.write(text)
                self._out.flush()
            except (OSError, ValueError):
                pass

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._render()
            self._stop_event.wait(self._interval)

    def _render(self) -> None:
        with self._lock:
            try:
                # 已有状态栏行 → 光标在其下方一行，回移覆盖；否则在当前行新建
                if self._has_bar_line:
                    self._out.write("\x1b[1A\r\x1b[K" + self._text_fn() + "\n")
                else:
                    self._out.write("\r\x1b[K" + self._text_fn() + "\n")
                self._has_bar_line = True
                self._out.flush()
            except (OSError, ValueError):
                pass


class _StatusBarStdoutProxy:
    """状态栏运行期间的 sys.stdout 代理。

    非空输出先经状态栏 ``print_above`` 清掉状态栏行，再写入真实 stdout，
    保证其他输出（如 agent 的 ``💬 Starting conversation``）从新行开始、
    不与状态栏行粘连。其余属性委托给真实 stdout。
    """

    def __init__(self, real, bar: _StatusBarThread):
        self._real = real
        self._bar = bar

    def write(self, text: str) -> int:
        if text:
            self._bar.print_above(text)
        return len(text)

    def flush(self) -> None:
        self._bar.print_above("")

    def __getattr__(self, name: str):
        return getattr(self._real, name)


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


def _ask(text: str, bottom_toolbar: Optional[Callable[[], str]] = None) -> str:
    """读取一行输入：有 prompt_toolkit 用带历史记录/状态栏的提示符，否则用 input。"""
    if _HAS_PTK:
        return prompt(
            text,
            history=FileHistory(HISTORY_FILE),
            bottom_toolbar=bottom_toolbar,
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
    interrupt"的模式）：输入新内容会调用 ``agent.interrupt(text)``，
    按 Ctrl+C 会调用 ``request_hard_interrupt(agent)`` 硬中断当前轮
    （流式/工具执行停止），
    新输入立即作为下一轮问题；被中断轮的不完整消息不回填对话历史。

    状态栏（对齐原版 cli.py 底部工具栏）：
    - 输入阶段：prompt_toolkit bottom_toolbar 显示冻结态状态栏
      （⏲ 上轮耗时 / ✓ 空闲 / 会话时长 / ctx）；
    - 回复期间：轻量状态栏线程实时刷新（⏱ 当前轮耗时递增），
      开始流式输出时清除，避免与打字机输出互相干扰。
    """
    history = None  # 当前会话的完整消息列表（None = 新会话）
    pending_input = None  # 打断时读到的新输入，作为下一轮问题
    session_start = time.time()  # 会话开始时间（状态栏显示会话时长）
    last_turn_finished_at = None  # 上一轮回复结束时间（输入阶段显示 ✓ idle）
    last_prompt_duration = 0.0  # 上一轮耗时（输入阶段显示 ⏲ 冻结值）

    def _toolbar() -> str:
        """输入阶段底部状态栏：冻结态（⏲ 上轮耗时 / ✓ 空闲）。"""
        ctx_tokens, ctx_length = _context_snapshot(agent)
        return _build_status_bar_text(
            agent.model or "",
            session_start=session_start,
            prompt_start_time=None,
            prompt_duration=last_prompt_duration,
            last_turn_finished_at=last_turn_finished_at,
            context_tokens=ctx_tokens,
            context_length=ctx_length,
        )

    while True:
        if pending_input is None:
            try:
                user_input = _ask("你 > ", bottom_toolbar=_toolbar).strip()
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
            # 新会话：轮换 session_id 并通知外部记忆 provider
            # （对齐原版 /new 语义：commit_session_boundary_async 保证
            # on_session_end 先于 on_session_switch，串行执行）。
            _mm = getattr(agent, "_memory_manager", None)
            if _mm is not None:
                from datetime import datetime
                import uuid as _uuid

                _old_sid = agent.session_id
                agent.session_id = (
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_uuid.uuid4().hex[:6]}"
                )
                try:
                    _mm.commit_session_boundary_async(
                        history or [],
                        new_session_id=agent.session_id,
                        parent_session_id=_old_sid or "",
                        reason="new_session",
                    )
                except Exception:
                    pass
            history = None
            print("（已开启新会话）\n")
            continue

        # 对话：把上一轮的 messages 作为 conversation_history 传入；
        # stream_callback 让最终回答逐段打印（打字机效果），
        # 失败路径不走流式（重试耗尽，无响应可流）
        before_history = history
        result_holder: dict = {}
        interrupted = {"flag": False}
        prompt_start = time.time()

        # 回复等待期间：底部状态栏实时刷新（⏱ 递增，🤖 前缀并入状态栏行，
        # 避免 \r 清行把助手标记覆盖）；开始流式输出即清除
        status_bar = _StatusBarThread(
            lambda: "🤖 " + _build_status_bar_text(
                agent.model or "",
                session_start=session_start,
                prompt_start_time=prompt_start,
                prompt_duration=0.0,
                last_turn_finished_at=None,
                # 实时读取 compressor：每次 API 调用后 usage 更新立即反映
                context_tokens=_context_snapshot(agent)[0],
                context_length=_context_snapshot(agent)[1],
            )
        )
        # 状态栏运行期间包一层 stdout 代理：其他输出（如 💬 Starting
        # conversation）先清掉状态栏行再从新行写，避免与状态栏行粘连
        _orig_stdout = sys.stdout
        sys.stdout = _StatusBarStdoutProxy(_orig_stdout, status_bar)
        status_bar.start()

        first_stream = {"fired": False}

        def _stream_with_status(text: str) -> None:
            """流式回调：首次输出前清除状态栏，避免两行互相干扰。"""
            if not first_stream["fired"]:
                first_stream["fired"] = True
                status_bar.stop(clear=True)
                sys.stdout = _orig_stdout  # 流式逐字走原 stdout，不再经代理
                print("🤖 ", end="", flush=True)  # 状态栏已清除，重新打印助手标记
            _stream_print(text)

        def _run():
            result_holder["result"] = agent.run_conversation(
                user_input,
                conversation_history=before_history,
                stream_callback=_stream_with_status,
            )

        worker = threading.Thread(target=_run, daemon=True, name="cli-conversation")
        worker.start()
        # 主线程监听键盘：有新输入或 Ctrl+C → 请求优雅中断当前轮
        while worker.is_alive():
            try:
                if select.select([sys.stdin], [], [], 0.2)[0]:
                    text = _read_stdin_available()
                    if text:
                        # 对齐原版 cli.py:14329：新输入打断时把消息传给
                        # interrupt()，agent 侧记录 _interrupt_message
                        agent.interrupt(text)
                        interrupted["flag"] = True
                        pending_input = text
                        break
            except KeyboardInterrupt:
                # Ctrl+C：第一次请求硬中断，不退出（对齐原版 cli.py:16137；
                # 输入阶段仍由 _ask 处理退出）
                request_hard_interrupt(agent)
                interrupted["flag"] = True
                break
            except (OSError, ValueError):
                # 平台不支持 stdin select（如 Windows）→ 退化为纯 Ctrl+C 打断
                pass
        worker.join()
        status_bar.stop(clear=True)  # 回复结束（无论是否流式）都清除状态栏
        sys.stdout = _orig_stdout  # 恢复 stdout（非流式路径可能尚未恢复）
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

        # 记录本轮耗时与结束时间，供输入阶段状态栏显示 ⏲ / ✓
        last_prompt_duration = max(0.0, time.time() - prompt_start)
        last_turn_finished_at = time.time()


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
