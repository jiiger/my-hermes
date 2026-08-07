"""主循环前置依赖：迭代预算 / 中断 / 工具执行（agent_init + run_agent）冒烟测试。"""

import pytest

from agent import agent_init
from agent.iteration_budget import IterationBudget
from run_agent import AIAgent


class FakeToolCall:
    def __init__(self, name, args, tc_id):
        self.function = type("F", (), {"name": name, "arguments": args})()
        self.id = tc_id


class FakeAssistantMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class FakeAgent:
    """只提供 init_agent 需要的桩方法，避免创建真实 OpenAI 客户端。"""

    def __init__(self):
        self.quiet_mode = True
        self._tool_impls = {}

    def _create_openai_client(self, client_kwargs, *, reason, shared):
        return None  # 桩：不创建真实客户端


@pytest.fixture(autouse=True)
def _no_ssl_verify(monkeypatch):
    # init_agent 内部会执行真实 CA bundle 校验，测试环境打桩跳过
    monkeypatch.setattr(
        "agent.ssl_guard.verify_ca_bundle_with_fallback", lambda: None
    )


def test_iteration_budget_consume_refund_remaining():
    budget = IterationBudget(3)
    assert budget.remaining == 3
    assert budget.consume()
    assert budget.consume()
    assert budget.consume()
    assert not budget.consume()  # 预算耗尽
    assert budget.remaining == 0
    budget.refund()
    assert budget.remaining == 1


def test_init_agent_sets_loop_prerequisites():
    agent = FakeAgent()
    agent_init.init_agent(agent, max_iterations=5)
    assert agent.max_iterations == 5
    assert isinstance(agent.iteration_budget, IterationBudget)
    assert agent.iteration_budget.remaining == 5
    assert agent._budget_grace_call is False
    assert agent._interrupt_requested is False
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._tool_impls == {}


def test_request_interrupt():
    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent.request_interrupt()
    assert agent._interrupt_requested is True


def test_execute_tool_calls_registered_and_unregistered():
    agent = AIAgent.__new__(AIAgent)
    agent._tool_impls = {
        "echo": lambda text: f"回声:{text}",
        "boom": lambda: (_ for _ in ()).throw(RuntimeError("炸了")),
    }
    msg = FakeAssistantMessage(
        [
            FakeToolCall("echo", '{"text": "hi"}', "tc-1"),
            FakeToolCall("nope", "{}", "tc-2"),
            FakeToolCall("boom", "{}", "tc-3"),
        ]
    )
    messages = []
    agent._execute_tool_calls(msg, messages, "t-1", 1)
    assert messages == [
        {"role": "tool", "tool_call_id": "tc-1", "name": "echo",
         "content": "回声:hi"},
        {"role": "tool", "tool_call_id": "tc-2", "name": "nope",
         "content": "错误: 未注册的工具 nope"},
        {"role": "tool", "tool_call_id": "tc-3", "name": "boom",
         "content": "工具执行异常: RuntimeError: 炸了"},
    ]


def test_execute_tool_calls_bad_json_defaults_to_empty_args():
    agent = AIAgent.__new__(AIAgent)
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return "ok"

    agent._tool_impls = {"capture": capture}
    msg = FakeAssistantMessage([FakeToolCall("capture", "不是JSON{{{", "tc-9")])
    messages = []
    agent._execute_tool_calls(msg, messages, "t-2")
    assert seen == {}  # 参数解析失败 → 按空参执行
    assert messages[0]["content"] == "ok"
