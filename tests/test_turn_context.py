"""turn_context 序言（build_turn_context）冒烟测试。"""

from agent import conversation_loop as cl
from agent.turn_context import build_turn_context


class FakeAgent:
    def __init__(self, quiet: bool = True):
        self.quiet_mode = quiet
        self._cached_system_prompt = None
        self.session_id = "sess-1"


def _build(agent, *args, **kwargs):
    """带真实回调的 build_turn_context 调用封装。"""
    kwargs.setdefault(
        "restore_or_build_system_prompt", cl._restore_or_build_system_prompt
    )
    kwargs.setdefault("install_safe_stdio", cl._install_safe_stdio)
    kwargs.setdefault("sanitize_surrogates", cl._sanitize_surrogates)
    kwargs.setdefault(
        "summarize_user_message_for_log", cl._summarize_user_message_for_log
    )
    return build_turn_context(agent, *args, **kwargs)


def test_build_turn_context_assembles_messages():
    agent = FakeAgent()
    ctx = _build(
        agent,
        "你好",
        "SYSTEM",
        [{"role": "assistant", "content": "前一轮回复"}],
        "task-1",
        None,
        None,
    )
    assert ctx.user_message == "你好"
    assert ctx.messages == [
        {"role": "assistant", "content": "前一轮回复"},
        {"role": "user", "content": "你好"},
    ]
    assert ctx.current_turn_user_idx == 1
    assert ctx.active_system_prompt == "SYSTEM"
    assert ctx.effective_task_id == "task-1"
    assert ctx.turn_id
    assert agent._cached_system_prompt == "SYSTEM"


def test_build_turn_context_sanitizes_surrogates_and_defaults():
    agent = FakeAgent()
    ctx = _build(agent, "坏\uD800字符", None, None, None, None, None)
    # 孤立代理对字符被替换为 U+FFFD
    assert ctx.user_message == "坏\ufffd字符"
    # system_message 为 None 且 agent 无 _build_system_prompt → 内置默认提示
    assert agent._cached_system_prompt == "You are a helpful assistant."
    # task_id 为空 → 自动生成
    assert ctx.effective_task_id
    # 无历史 → messages 只有用户消息
    assert ctx.messages == [{"role": "user", "content": "坏\ufffd字符"}]
    assert ctx.current_turn_user_idx == 0


def test_restore_or_build_reuses_cache():
    agent = FakeAgent()
    agent._cached_system_prompt = "已缓存提示"
    cl._restore_or_build_system_prompt(agent, "新提示", None)
    assert agent._cached_system_prompt == "已缓存提示"  # 缓存优先，不被覆盖


def test_display_kind_metadata_stamped():
    agent = FakeAgent()
    ctx = _build(
        agent,
        "继续",
        None,
        None,
        None,
        None,
        None,
        persist_user_display_kind="auto_continue",
        persist_user_display_metadata={"count": 2},
    )
    assert ctx.messages[-1]["display_kind"] == "auto_continue"
    assert ctx.messages[-1]["display_metadata"] == {"count": 2}
