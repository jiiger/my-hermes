"""conversation_loop 主循环冒烟测试：用假 client 模拟完整对话流程。"""

import pytest

from agent import conversation_loop as cl
from agent.iteration_budget import IterationBudget
from run_agent import AIAgent


class FakeToolCall:
    def __init__(self, name, arguments, tc_id):
        self.id = tc_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [FakeChoice(FakeMessage(content, tool_calls), finish_reason)]


class FakeClient:
    """按脚本返回响应的假 client（模拟 OpenAI SDK 的 client.chat.completions.create）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []  # 记录每次调用的 kwargs
        self.chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "Completions", (), {"create": self._create}
                )()
            },
        )()

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise RuntimeError("脚本用尽")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(client, *, max_iterations=5, tool_impls=None, quiet=True):
    agent = AIAgent.__new__(AIAgent)
    agent.quiet_mode = quiet
    agent.model = "test-model"
    agent.provider = "test"
    agent._base_url_lower = ""
    agent.client = client
    agent.max_iterations = max_iterations
    agent.iteration_budget = IterationBudget(max_iterations)
    agent._budget_grace_call = False
    agent._interrupt_requested = False
    agent.tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]
    agent.valid_tool_names = {"add"}
    agent._tool_impls = tool_impls or {}
    agent._api_max_retries = 3
    agent.ephemeral_system_prompt = None
    agent.prefill_messages = []
    agent._cached_system_prompt = "SYS"
    agent.session_id = "test-session"
    agent.platform = "cli"
    return agent


def test_tool_call_chain_then_final_answer():
    """一轮工具调用 + 一轮最终回答的完整链路。"""
    client = FakeClient(
        [
            FakeResponse(
                tool_calls=[
                    FakeToolCall("add", '{"x":1,"y":2}', "tc-1"),
                ]
            ),
            FakeResponse(content="结果是3", finish_reason="stop"),
        ]
    )
    agent = make_agent(client, tool_impls={"add": lambda x, y: x + y})
    result = cl.run_conversation(agent, "1+2=?")
    assert result["final_response"] == "结果是3"
    assert result["interrupted"] is False and result["failed"] is False
    assert result["api_call_count"] == 2
    assert result["turn_exit_reason"] == "completed"
    # messages 应包含: user + assistant(tool_calls) + tool + assistant(最终)
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert result["messages"][2]["content"] == "3"  # 工具结果（字符串化）
    # API 收到的是剥离内部字段的 api_messages，且带系统提示
    first_call = client.calls[0]
    assert first_call["model"] == "test-model"
    assert first_call["messages"][0] == {"role": "system", "content": "SYS"}
    assert "tools" in first_call


def test_api_failure_exhausts_retries():
    """API 连续失败 max_retries 次 → failed=True，错误信息作为最终回复。"""
    client = FakeClient([RuntimeError("连接超时")] * 3)
    agent = make_agent(client)
    result = cl.run_conversation(agent, "你好")
    assert result["failed"] is True
    assert result["turn_exit_reason"] == "api_failed"
    assert "连接超时" in result["final_response"]
    assert len(client.calls) == 3  # 重试了 3 次


def test_budget_exhausted_stops_loop():
    """预算 1 次：第一轮工具调用后预算耗尽，循环正常停止。"""
    client = FakeClient(
        [
            FakeResponse(
                tool_calls=[FakeToolCall("add", '{"x":1,"y":2}', "tc-1")]
            ),
            FakeResponse(content="不该被调用"),
        ]
    )
    agent = make_agent(client, max_iterations=1,
                       tool_impls={"add": lambda x, y: x + y})
    result = cl.run_conversation(agent, "1+2=?")
    assert result["turn_exit_reason"] == "budget_exhausted"
    assert result["failed"] is False
    assert result["final_response"] is None
    assert len(client.calls) == 1  # 只调了一次 API


def test_interrupt_stops_loop():
    """中断标志置位后，循环第一轮就退出。"""
    client = FakeClient([FakeResponse(content="不该被调用")])
    agent = make_agent(client)
    agent.request_interrupt()
    result = cl.run_conversation(agent, "你好")
    assert result["interrupted"] is True
    assert result["turn_exit_reason"] == "interrupted_by_user"
    assert len(client.calls) == 0  # 一次 API 都没调


def test_unregistered_tool_returns_error_message():
    """模型调用了未注册的工具 → 工具结果写错误信息，循环继续。"""
    client = FakeClient(
        [
            FakeResponse(
                tool_calls=[FakeToolCall("不存在的工具", "{}", "tc-1")]
            ),
            FakeResponse(content="我知道了", finish_reason="stop"),
        ]
    )
    agent = make_agent(client)  # 不注册任何工具实现
    result = cl.run_conversation(agent, "帮我看看")
    assert result["final_response"] == "我知道了"
    assert "未注册的工具" in result["messages"][2]["content"]
    assert result["api_call_count"] == 2


def test_grace_call_does_not_consume_budget():
    """_budget_grace_call 为 True 时，允许在预算耗尽后多调一次。"""
    client = FakeClient([FakeResponse(content="最终答案", finish_reason="stop")])
    agent = make_agent(client, max_iterations=0)
    agent._budget_grace_call = True  # 预算为 0 但允许宽限一次
    result = cl.run_conversation(agent, "你好")
    assert result["final_response"] == "最终答案"
    assert result["api_call_count"] == 1
    assert agent._budget_grace_call is False  # 宽限标记被消费
