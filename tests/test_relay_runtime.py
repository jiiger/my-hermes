"""agent/relay_runtime 核心功能冒烟测试（无外部依赖，可离线运行）。"""

import pytest

from agent import relay_runtime as r

try:
    import nemo_relay  # noqa: F401

    _HAS_NEMO = True
except ImportError:
    _HAS_NEMO = False


def _fake_relay():
    """构造一个最小假 relay：模拟 scope.push/pop/event 与订阅者冲刷。"""

    class FakeRelay:
        class ScopeType:
            Agent = "agent"
            Function = "function"

        def __init__(self):
            self.stack = []
            self.pushed = []
            self.popped = []

        def get_scope_stack(self):
            return self.stack

        def _push(
            self, name, stype, handle=None, data=None, input=None, metadata=None
        ):
            h = ("handle", len(self.stack))
            self.stack.append(name)
            self.pushed.append(name)
            return h

        def _pop(self, handle, output=None, metadata=None):
            self.popped.append(handle)

        def _event(self, name, handle=None, data=None, metadata=None):
            pass

        subscribers = type("Subs", (), {"flush": staticmethod(lambda: None)})()

    fake = FakeRelay()
    fake.scope = type(
        "S",
        (),
        {
            "push": fake._push,
            "pop": fake._pop,
            "event": fake._event,
            "ScopeType": FakeRelay.ScopeType,
        },
    )()
    return fake


class TestCoordinatorNoop:
    """空壳主机 + 协调器：租约/轮次全生命周期（曾因缺方法崩溃的路径）。"""

    def test_full_cycle(self):
        lease = r.ConversationLease(
            profile_key="test-noop",
            session_id="s1",
            platform="cli",
            host=r.NoopRelayRuntime("test-noop", "forced"),
            session=None,
        )
        turn = r.SESSION_COORDINATOR.begin_turn(lease, turn_id="t1", task_id="k")
        assert r.current_turn() is turn
        r.SESSION_COORDINATOR.end_turn(turn, outcome="completed")
        assert r.current_turn() is None
        assert r.active_turn("s1") is None
        r.SESSION_COORDINATOR.release_conversation(lease)
        assert lease.released
        assert not r.SESSION_COORDINATOR._active_turns

    def test_double_end_turn_is_safe(self):
        lease = r.ConversationLease(
            profile_key="test-noop",
            session_id="s2",
            platform="cli",
            host=r.NoopRelayRuntime("test-noop", "forced"),
            session=None,
        )
        turn = r.SESSION_COORDINATOR.begin_turn(lease, turn_id="t2", task_id="k")
        r.SESSION_COORDINATOR.end_turn(turn, outcome="completed")
        r.SESSION_COORDINATOR.end_turn(turn, outcome="completed")  # 幂等，不应抛异常


class TestRelayRuntime:
    """真实 RelayRuntime 方法（注入假 relay，确定性验证）。"""

    def test_session_lifecycle(self):
        rt = r.RelayRuntime(relay=_fake_relay(), profile_key="test-real")
        s = rt.ensure_session({"session_id": "sess-A"})
        assert s is not None and s.handle == ("handle", 0)
        assert rt.ensure_session({"session_id": "sess-A"}) is s  # 幂等

        assert rt.run_in_session(s, lambda x: x * 2, 21) == 42
        assert rt.get_session("sess-A") is s
        assert rt.get_session("nope") is None

        rt.close_session({"session_id": "sess-A"})
        assert rt.get_session("sess-A") is None
        rt.shutdown()  # 幂等，不应抛异常

    def test_subagent_register_unregister(self):
        rt = r.RelayRuntime(relay=_fake_relay(), profile_key="test-real")
        child = rt.register_subagent(
            {"parent_session_id": "sess-A", "child_session_id": "child-1"}
        )
        assert child is not None
        assert rt._subagent_parents["child-1"] == "sess-A"
        rt.unregister_subagent({"child_session_id": "child-1"})
        assert rt.get_session("child-1") is None
        assert "child-1" not in rt._subagent_parents


@pytest.mark.skipif(not _HAS_NEMO, reason="nemo_relay 未安装")
def test_real_nemo_relay_lifecycle():
    """真实 nemo_relay：会话 scope 可推入/关闭。"""
    rt = r.RelayRuntime(profile_key="test-nemo")
    s = rt.ensure_session({"session_id": "nemo-1"})
    assert s is not None and s.handle is not None
    rt.close_session({"session_id": "nemo-1"})
    rt.shutdown()
