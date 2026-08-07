from typing import Any, Dict, List, Optional


def run_conversation(
    agent,
    user_message: Any,
    system_message: str = None,
    conversation_history: List[Dict[str, Any]] = None,
    task_id: str = None,
    stream_callback: Optional[callable] = None,
    persist_user_message: Optional[Any] = None,
    persist_user_timestamp: Optional[float] = None,
    persist_user_display_kind: Optional[str] = None,
    persist_user_display_metadata: Optional[Dict[str, Any]] = None,
    moa_config: Optional[dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    完成一个完整的conversation

    参数：
    user_message（str）：用户的消息/问题
    system_message（str）：自定义系统消息（可选，如果已提供，则覆盖临时系统提示）
    conversation_history（List[Dict]）：之前的对话消息（可选）
    task_id（str）：此任务的唯一标识符，用于在并发任务之间隔离虚拟机（VM）（可选，如果未提供，则自动生成）
    stream_callback：流式处理期间，每次文本增量更新时调用的可选回调函数。在完整响应之前，由TTS管道用于启动音频生成。当设置为“无”（默认）时，API调用将使用标准的非流式路径。
    persist_user_message：当user_message包含仅限API的合成前缀时，可选的干净用户消息，以存储在transcripts/history中。
    persist_user_timestamp：可选的平台事件时间戳，用于作为元数据存储在该持久化用户消息中。
    persist_user_display_kind：合成用户轮次的可选展示类型（如“auto_continue”、“model_switch”等）。仅显示：转录界面将该行呈现为时间线事件，而非用户气泡，同时模型仍会原样接收消息。
    persist_user_display_metadata：该事件的可选载荷（例如，一个委托的任务计数）。或者排队等待后续预取工作。


    返回：字典：包含最终回复和消息历史的完整对话结果
    """

    # TODO moa_config配置

    agent._last_compaction_in_place = False
    agent._last_compression_attempt_recorded = False
    agent._last_compression_attempt_in_place = None

    