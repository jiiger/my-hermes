"""显示层 —— 仅保留 KawaiiSpinner（精简移植版）。

对应原版 hermes-agent 的 agent/display.py（约 1500 行）中的
KawaiiSpinner 类（原版 :1054-1276）。

砍掉原版其余全部内容：redact（参数脱敏）、skin_engine（皮肤系统）、
skill_manager_tool、diff 渲染、_cute_tool_message / _detect_tool_failure
等工具状态行助手、以及 prompt_toolkit 的 patch_stdout 分支。

因此相对原版 KawaiiSpinner 的改动：
- get_waiting_faces / get_thinking_faces / get_thinking_verbs 砍掉对
  _get_skin() 的皮肤查询，直接返回类常量（原版在无皮肤时同样回退常量）；
- _animate 砍掉皮肤翅膀（wings）与 prompt_toolkit StdoutProxy 分支；
- _is_patch_stdout_proxy 不再需要（无 prompt_toolkit）。
"""

import sys
import threading
import time


class KawaiiSpinner:
    """CLI 反馈用带 kawaii 表情的动画 spinner（工具执行期间显示）。"""

    SPINNERS = {
        'dots': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'],
        'bounce': ['⠁', '⠂', '⠄', '⡀', '⢀', '⠠', '⠐', '⠈'],
        'grow': ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█', '▇', '▆', '▅', '▄', '▃', '▂'],
        'arrows': ['←', '↖', '↑', '↗', '→', '↘', '↓', '↙'],
        'star': ['✶', '✷', '✸', '✹', '✺', '✹', '✸', '✷'],
        'moon': ['🌑', '🌒', '🌓', '🌔', '🌕', '🌖', '🌗', '🌘'],
        'pulse': ['◜', '◠', '◝', '◞', '◡', '◟'],
        'brain': ['🧠', '💭', '💡', '✨', '💫', '🌟', '💡', '💭'],
        'sparkle': ['⁺', '˚', '*', '✧', '✦', '✧', '*', '˚'],
    }

    KAWAII_WAITING = [
        "(｡◕‿◕｡)", "(◕‿◕✿)", "٩(◕‿◕｡)۶", "(✿◠‿◠)", "( ˘▽˘)っ",
        "♪(´ε` )", "(◕ᴗ◕✿)", "ヾ(＾∇＾)", "(≧◡≦)", "(★ω★)",
    ]

    KAWAII_THINKING = [
        "(｡•́︿•̀｡)", "(◔_◔)", "(¬‿¬)", "( •_•)>⌐■-■", "(⌐■_■)",
        "(´･_･`)", "◉_◉", "(°ロ°)", "( ˘⌣˘)♡", "ヽ(>∀<☆)☆",
        "٩(๑❛ᴗ❛๑)۶", "(⊙_⊙)", "(¬_¬)", "( ͡° ͜ʖ ͡°)", "ಠ_ಠ",
    ]

    THINKING_VERBS = [
        "pondering", "contemplating", "musing", "cogitating", "ruminating",
        "deliberating", "mulling", "reflecting", "processing", "reasoning",
        "analyzing", "computing", "synthesizing", "formulating", "brainstorming",
    ]

    @classmethod
    def get_waiting_faces(cls) -> list:
        """返回等待表情列表（精简版无皮肤系统，直接返回常量）。"""
        return cls.KAWAII_WAITING

    @classmethod
    def get_thinking_faces(cls) -> list:
        """返回思考表情列表（精简版无皮肤系统，直接返回常量）。"""
        return cls.KAWAII_THINKING

    @classmethod
    def get_thinking_verbs(cls) -> list:
        """返回思考动词列表（精简版无皮肤系统，直接返回常量）。"""
        return cls.THINKING_VERBS

    def __init__(self, message: str = "", spinner_type: str = 'dots', print_fn=None):
        self.message = message
        self.spinner_frames = self.SPINNERS.get(spinner_type, self.SPINNERS['dots'])
        self.running = False
        self.thread = None
        self.frame_idx = 0
        self.start_time = None
        self.last_line_len = 0
        # 可选可调用对象，把所有输出路由到它（如静默后台 agent 用 no-op）。
        # 设置后完全绕过 self._out，让覆盖了 _print_fn 的 agent 保持静默。
        self._print_fn = print_fn
        # 现在就捕获 stdout，避免之后子 agent 的 redirect_stdout(devnull)
        # 把 sys.stdout 换成黑洞。
        self._out = sys.stdout

    def _write(self, text: str, end: str = '\n', flush: bool = False):
        """写到 spinner 创建时捕获的 stdout。

        构造时提供了 print_fn 则所有输出都走它 —— 允许调用方用 no-op
        lambda 静默掉 spinner。
        """
        if self._print_fn is not None:
            try:
                self._print_fn(text)
            except Exception:
                pass
            return
        try:
            self._out.write(text + end)
            if flush:
                self._out.flush()
        except (ValueError, OSError):
            pass

    @property
    def _is_tty(self) -> bool:
        """检查输出是否为真实终端，对已关闭流安全。"""
        try:
            return hasattr(self._out, 'isatty') and self._out.isatty()
        except (ValueError, OSError):
            return False

    def _animate(self):
        # stdout 不是真实终端（Docker、systemd、管道）时完全跳过动画 ——
        # 会产生海量日志。只打一次开始，让 stop() 打完成。
        if not self._is_tty:
            self._write(f"  [tool] {self.message}", flush=True)
            while self.running:
                time.sleep(0.5)
            return

        while self.running:
            frame = self.spinner_frames[self.frame_idx % len(self.spinner_frames)]
            elapsed = time.time() - self.start_time
            line = f"  {frame} {self.message} ({elapsed:.1f}s)"
            pad = max(self.last_line_len - len(line), 0)
            self._write(f"\r{line}{' ' * pad}", end='', flush=True)
            self.last_line_len = len(line)
            self.frame_idx += 1
            time.sleep(0.12)

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def update_text(self, new_message: str):
        self.message = new_message

    def print_above(self, text: str):
        """在 spinner 上方打印一行而不打断动画。

        清掉当前 spinner 行、打印文本，让下一个动画 tick 在下一行重绘。
        线程安全：用创建时捕获的 stdout 引用（self._out）。
        """
        if not self.running:
            self._write(f"  {text}", flush=True)
            return
        # 用空格（而非 \033[K）清 spinner 行，避免 prompt_toolkit
        # patch_stdout 活跃时产生乱码转义码 —— 与 stop() 同法。
        blanks = ' ' * max(self.last_line_len + 5, 40)
        self._write(f"\r{blanks}\r  {text}", flush=True)

    def stop(self, final_message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

        is_tty = self._is_tty
        if is_tty:
            # 用空格而非 \033[K 清 spinner 行，避免 prompt_toolkit
            # patch_stdout 活跃时产生乱码转义码。
            blanks = ' ' * max(self.last_line_len + 5, 40)
            self._write(f"\r{blanks}\r", end='', flush=True)
        if final_message:
            elapsed = f" ({time.time() - self.start_time:.1f}s)" if self.start_time else ""
            if is_tty:
                self._write(f"  {final_message}", flush=True)
            else:
                self._write(f"  [done] {final_message}{elapsed}", flush=True)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
