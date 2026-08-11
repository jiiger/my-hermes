"""上下文引擎抽象基类（对应原版 agent/context_engine.py）。

上下文引擎决定对话接近模型 token 上限时如何管理上下文：内置的
ContextCompressor 是默认实现。第三方引擎可通过插件替换，但 my-hermes
精简版只保留内置 compressor（配置 ``context.engine`` 恒为 "compressor"）。

引擎职责：
  - 决定何时触发压缩（should_compress）
  - 执行压缩（摘要 / 修剪等，compress）
  - 从 API 响应追踪 token 用量（update_from_response）

生命周期：
  1. 引擎实例化并挂到 agent.context_compressor
  2. update_from_response() 在每次 API 响应后调用
  3. should_compress() 在每轮后检查
  4. compress() 在 should_compress() 为 True 时调用
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ContextEngine(ABC):
    """所有上下文引擎必须实现的基类。"""

    # -- 身份 ------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """短标识（如 'compressor'）。"""

    # -- token 状态（run_agent / conversation_loop 直接读取）-------------
    # 引擎必须维护这些字段。

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # -- 压缩参数 --------------------------------------------------------
    # protect_first_n：始终原样保留的非 system head 消息条数（不含总是
    # 隐式保护的 system prompt）。默认 3 保持 "system + 前 3 条非 system"
    # 的 head 形状。

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- 核心接口 --------------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """根据 API 响应更新追踪的 token 用量。

        每次 LLM 调用后调用，usage 为归一化 dict，含旧键
        ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``。
        """

    @abstractmethod
    def should_compress(self, prompt_tokens: int = None) -> bool:
        """返回本轮是否应触发压缩。"""

    def should_compress_info(self, prompt_tokens: int = None) -> "tuple[bool, str | None]":
        """返回 ``(should_compress, reason)``。

        基类实现向后兼容：只实现 should_compress 的引擎得到
        ``(should_compress(prompt_tokens), None)``。具体引擎可覆写以给出
        人类可读的阻断原因（如摘要 LLM 冷却）。
        """
        return self.should_compress(prompt_tokens), None

    @abstractmethod
    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """压缩消息列表并返回新列表。

        主入口。引擎收到完整消息列表，返回（可能更短的）合法
        OpenAI 格式消息序列。my-hermes 精简版忽略 memory_context。
        """

    # -- 可选：工具结果预修剪 -------------------------------------------

    def prune_tool_results_only(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int | None = None,
    ) -> tuple[List[Dict[str, Any]], int]:
        """不调 LLM，确定性修剪旧工具结果载荷。

        返回 ``(messages, n_pruned)``。默认安全 no-op。
        """
        return messages, 0

    # -- 可选：预检（每轮 API 调用前）------------------------------------

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """API 调用前的粗略检查（尚无真实 token 数）。默认跳过。"""
        return False

    def should_defer_preflight_to_real_usage(self, rough_tokens: int) -> bool:
        """预检应信任最近真实用量时返回 True。默认 False。"""
        return False

    # -- 可选：手动 /compress 预检 ---------------------------------------

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """检查 messages 中是否有可压缩内容。

        默认返回 True（总是尝试）。引擎可覆写：当 transcript 仍在保护区内
        时返回 False。
        """
        return True

    # -- 可选：会话生命周期 ----------------------------------------------

    def on_session_reset(self) -> None:
        """/new 或 /reset 时调用，重置会话状态。"""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    # -- 可选：状态展示 --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """返回状态 dict（显示/日志用）。"""
        last_prompt = self.last_prompt_tokens if self.last_prompt_tokens > 0 else 0
        return {
            "last_prompt_tokens": last_prompt,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, last_prompt / self.context_length * 100)
                if self.context_length else 0
            ),
            "compression_count": self.compression_count,
        }
