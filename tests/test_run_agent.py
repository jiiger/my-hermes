"""run_agent.AIAgent.run_conversation 转发器冒烟测试。

不依赖真实 LLM API：用桩替换 agent.conversation_loop.run_conversation，
验证转发器的 relay 租约/轮次生命周期与参数传递（无递归、无 TypeError）。
"""

import agent.conversation_loop as cl
from agent import relay_runtime
from run_agent import AIAgent


def _make_agent(session_id: str) -> AIAgent:
    # 跳过 __init__（会走 init_agent/API 初始化），只构造空壳并补齐必需属性
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = "cli"
    agent.model = "test-model"
    agent._relay_pending_turn_id = None
    return agent


def test_run_conversation_forwards_without_recursion(monkeypatch):
    agent = _make_agent("test-fwd-session")

    captured = {}

    def fake_run_conversation(
        fwd_agent,
        user_message,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        persist_user_timestamp=None,
        persist_user_display_kind=None,
        persist_user_display_metadata=None,
        moa_config=None,
    ):
        captured["agent"] = fwd_agent
        captured["user_message"] = user_message
        captured["task_id"] = task_id
        captured["persist_user_timestamp"] = persist_user_timestamp
        return {"final_response": "ok", "messages": [], "interrupted": False}

    monkeypatch.setattr(cl, "run_conversation", fake_run_conversation)

    result = agent.run_conversation(
        "你好", task_id="task-1", persist_user_timestamp=123.45
    )

    # 转发目标收到正确参数（而不是递归调用自身）
    assert captured["agent"] is agent
    assert captured["user_message"] == "你好"
    assert captured["task_id"] == "task-1"
    assert captured["persist_user_timestamp"] == 123.45
    assert result == {"final_response": "ok", "messages": [], "interrupted": False}

    # relay 租约/轮次已走完并复位
    assert agent._relay_pending_turn_id is None
    assert relay_runtime.current_turn() is None


def test_run_conversation_outcome_mapping(monkeypatch):
    agent = _make_agent("test-outcome-session")

    def fake_interrupted(
        fwd_agent,
        user_message,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        persist_user_timestamp=None,
        persist_user_display_kind=None,
        persist_user_display_metadata=None,
        moa_config=None,
    ):
        return {"final_response": "", "messages": [], "interrupted": True}

    monkeypatch.setattr(cl, "run_conversation", fake_interrupted)

    agent.run_conversation("停一下", task_id="task-2")
    # interrupted=True 的终态结果不应抛异常，relay 轮次正常收尾
    assert relay_runtime.current_turn() is None
