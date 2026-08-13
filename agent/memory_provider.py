"""可插拔记忆提供方（MemoryProvider）抽象基类（精简移植版）。

对应原版 hermes-agent 的 agent/memory_provider.py（357 行）。记忆提供方
让 agent 具备跨会话持久召回；MemoryManager 强制"最多一个外部提供方"
以防水涨船高的工具 schema 与互相冲突的记忆后端。

外部提供方（Honcho / Hindsight / Mem0 等）经 plugins/memory/<name>/ 插件
加载，由 config.yaml 的 memory.provider 键激活。同一时刻只运行一个外部
提供方。

生命周期（由 MemoryManager 调用，run_agent 接线）：
  initialize()          — 连接、建资源、预热
  system_prompt_block()  — 系统提示里的静态文本
  prefetch(query)        — 每回合前的后台召回
  sync_turn(user, asst)  — 每回合后的异步写入
  get_tool_schemas()     — 暴露给模型的工具 schema
  handle_tool_call()     — 工具调用派发
  shutdown()             — 干净退出

可选钩子（override 选择启用）：
  on_turn_start / on_session_end / on_session_switch / on_pre_compress /
  on_memory_write / on_delegation

精简版改动（相对原版）：
- 砍掉 get_config_schema / save_config（hermes memory setup 命令用，
  my-hermes 无此命令）与 backup_paths（hermes backup 命令用）；
- 其余接口签名逐字一致。
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# 无语义信号的提示词——琐碎确认、问候、斜杠命令、空输入。
# 核心的每回合 prefetch 门（agent/turn_context.py）与 provider 侧分类器
# （plugins/memory/honcho）共享此单一事实来源，两者永不失步。
# 该替换被锚定，后面只能跟空白或标点，所以只是以琐碎词开头的词
# （"k8s"、"yolo"、"note"、"hindsight"）不会误匹配，而带尾随标点的变体
# （"hi!"、"hey."、"thanks :)"、"done???"）会。
TRIVIAL_PROMPT_RE = re.compile(
    r"^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|"
    r"hi|hey|hello|yo|sup|"
    r"continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)"
    r'[\s!?.:;,"'
    + "'"
    + r"~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$",
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    """提示词是否琐碎到不值得触发记忆召回。

    空/纯空白输入、斜杠命令、裸问候或确认（可带尾随标点）都算琐碎。
    调用方用它跳过无语义信号回合的 provider prefetch/注入——省一次阻塞
    网络往返，也防止陈旧的用户-模型上下文带偏一字回复。
    """
    if not text:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    """记忆提供方抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """短标识（如 'builtin'、'honcho'、'hindsight'）。"""

    # -- 核心生命周期（实现这些） -------------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """本提供方是否已配置、有凭据、可用。

        agent 初始化时调用以决定是否激活。不应发起网络调用——只查配置
        与已安装依赖。
        """

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """按会话初始化。

        启动时调用一次。可建资源（库、表）、建立连接、启动后台线程等。

        kwargs 恒含：
          - hermes_home (str)：激活的 HERMES_HOME 目录。用它做按 profile
            隔离的存储，而不是硬编码 ``~/.hermes``。
          - platform (str)："cli"、"telegram"、"discord"、"cron" 等。

        kwargs 可能含：
          - agent_context (str)："primary"、"subagent"、"cron"、"flush"。
            非 primary 上下文应跳过写入（cron 系统提示会污染用户画像）。
          - agent_identity (str)：profile 名（如 "coder"），做按 profile
            的提供方身份隔离。
          - agent_workspace (str)：共享工作区名（如 "hermes"）。
          - parent_session_id (str)：子 agent 的父会话 id。
          - user_id / user_id_alt / user_name / chat_id 等平台用户标识。
        """

    def system_prompt_block(self) -> str:
        """返回要进系统提示的文本。

        系统提示组装时调用。返回空串表示跳过。这是**静态** provider 信息
        （说明、状态）。召回上下文由 prefetch() 单独注入。
        """
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """为即将到来的回合召回相关上下文。

        每次 API 调用前调用。返回要作为上下文注入的格式化文本，无相关
        内容返回空串。实现应快——实际召回放后台线程，这里返回缓存结果。

        session_id 提供给并发会话（网关群聊、缓存 agent）；不需要按会话
        隔离的提供方可忽略。
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """排队一次后台召回，供**下一**回合使用。

        每回合完成后调用。结果由下一回合的 prefetch() 消费。默认为空操作
        ——做后台预取的提供方应 override。
        """

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """把完成的回合持久化到后端。

        每回合后调用。应非阻塞——后端有延迟就排队后台处理。

        ``messages`` 是回合完成时的 OpenAI 风格消息列表，含 assistant
        工具调用与工具结果。不需要原始回合上下文的提供方可忽略。
        """

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回本提供方暴露的工具 schema。

        每个 schema 遵循 OpenAI 函数调用格式：
        {"name": "...", "description": "...", "parameters": {...}}

        无工具（仅上下文型）返回空列表。
        """

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """处理本提供方某个工具的工具调用。

        必须返回 JSON 字符串（工具结果）。只对 get_tool_schemas() 返回的
        工具名调用。
        """
        raise NotImplementedError(
            f"Provider {self.name} does not handle tool {tool_name}"
        )

    def shutdown(self) -> None:
        """干净关闭——冲刷队列、关闭连接。"""

    # -- 可选钩子（override 选择启用） ---------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """每回合开始以用户消息调用。

        用于回合计数、作用域管理、周期维护。

        kwargs 可能含：remaining_tokens、model、platform、tool_count。
        """

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """会话结束时调用（显式退出或超时）。

        用于会话末事实提取、总结等。messages 是完整对话历史。

        不是每回合后调用——只在真实会话边界（CLI 退出、/reset、网关会话
        过期）。
        """

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """agent 进程内切换 session_id 时调用。

        触发于 ``/resume``、``/branch``、``/reset``、``/new``（CLI）、网关
        对应操作、以及上下文压缩——任何在不拆掉提供方的情况下重绑
        ``AIAgent.session_id`` 的路径。

        在 initialize() 里缓存了按会话状态的提供方（``_session_id``、
        ``_document_id``、累积回合缓冲、计数器）应在此更新/重置，让后续
        写入落到正确会话记录。

        参数
        ----------
        new_session_id: 刚切换到的 session_id。
        parent_session_id: 有意义的旧 session_id——``/branch``（fork 血统）、
            上下文压缩（延续血统）、``/resume``（离开的会话）会设置。
            无血统时为空串。
        reset: 真正的新对话（非续接）为 True。由 ``/reset`` / ``/new`` 触发。
            提供方应冲刷累计的按会话缓冲（``_session_turns`` 等）。``/resume``
            / ``/branch`` / 压缩为 False——逻辑对话在新 id 下延续。
        rewound: session_id 未变但转录被截断为 True；缓存按回合文档状态的
            提供方应失效。

        默认为空操作以向后兼容。
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """上下文压缩丢弃旧消息前调用。

        用于从即将被压缩的消息里提取洞见。messages 是将被摘要/丢弃的列表。

        返回要进压缩摘要 prompt 的文本，让压缩器保留 provider 提取的洞见。
        无贡献返回空串（向后兼容默认）。
        """
        return ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """子 agent 完成时在**父** agent 上调用。

        父的记忆提供方拿到 task+result 对，作为"委派了什么、回来了什么"
        的观察。子 agent 本身无 provider 会话（skip_memory=True）。

        task: 委派提示词
        result: 子 agent 的最终回复
        child_session_id: 子 agent 的 session_id
        """

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """内置 memory 工具写入条目时调用。

        action: 'add'、'replace' 或 'remove'
        target: 'memory' 或 'user'
        content: 条目内容
        metadata: 结构化溯源（可用时）。常见键：``write_origin``、
            ``execution_context``、``session_id``、``parent_session_id``、
            ``platform``、``tool_name``。

        用于把内置记忆写入镜像到自己的后端。
        """
