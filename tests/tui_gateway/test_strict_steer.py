"""Strict steering through the real RPC dispatcher and tool checkpoint."""
import concurrent.futures
import threading
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
from run_agent import AIAgent
from tui_gateway import server
from tui_gateway.strict_steer import StrictSteerRun


@pytest.fixture
def runtime(monkeypatch):
    frames = []
    transport = SimpleNamespace(write=lambda frame: frames.append(frame) or True)
    agent = object.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = threading.Lock()
    owner = {"session_key": "stored", "history_lock": threading.Lock(),
             "agent": agent, "history": [], "running": True, "transport": transport}
    monkeypatch.setattr(server, "_sessions", {"ui": owner})
    server._start_inflight_turn(owner, "original")
    run = owner["_strict_steer_run"]
    agent._strict_steer_checkpoint = lambda messages, count: run.checkpoint(
        messages, count, agent._drain_pending_steer)

    def call(method="session.steer.submit", via=transport, **params):
        args = {"session_id": "ui", **params}
        if method == "session.steer.submit":
            args = {"run_id": run.run_id, "intent_id": "intent", "text": "strict guidance", **args}
        return server.dispatch({"id": 1, "method": method, "params": args}, transport=via)

    return owner, agent, run, frames, call


def batch():
    return [{"role": "assistant", "tool_calls": [{"id": "tool"}]},
            {"role": "tool", "tool_call_id": "tool", "content": "real result"}]


def test_first_tool_checkpoint_and_lost_ack_retry(runtime):
    owner, agent, run, frames, call = runtime
    assert call()["result"]["status"] == "accepted"
    assert call()["result"]["status"] == "accepted"
    messages = batch()
    tool_before = dict(messages[-1])
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert messages[-2] == tool_before
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].count("strict guidance") == 1
    assert call()["result"]["status"] == "delivered"
    assert call("session.steer.state")["result"]["receipts"][0]["status"] == "delivered"
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert len(messages) == 3
    assert agent._drain_pending_steer() is None
    statuses = [f["params"]["payload"]["status"] for f in frames
                if f.get("method") == "event" and f["params"]["type"] == "steer.update"]
    assert statuses == ["accepted", "delivered"]


def test_startup_then_tool_checkpoint(runtime):
    owner, agent, run, frames, call = runtime
    owner["agent"] = None
    assert call()["result"]["status"] == "accepted"
    owner["agent"] = agent
    messages = batch()
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert run.snapshot()["receipts"][0]["status"] == "delivered"


@pytest.mark.parametrize("terminal", ["clear", "error"])
def test_no_tool_terminal_never_falls_back(runtime, terminal):
    owner, agent, run, frames, call = runtime
    call()
    old_history = batch() + [{"role": "assistant", "content": "old answer"}]
    apply_pending_steer_to_tool_results(agent, old_history, 0)
    assert len(old_history) == 3
    if terminal == "clear":
        server._clear_inflight_turn(owner)
    else:
        server._fail_inflight_turn(owner, "startup failed")
    assert call()["result"]["status"] == "missed"
    assert "error" in call(intent_id="late")
    assert agent._drain_pending_steer() is None
    assert not owner.get("queued_prompts") and not owner.get("queued_prompt")


def test_strict_and_legacy_share_one_new_user_row(runtime):
    owner, agent, run, frames, call = runtime
    call()
    agent.steer("legacy guidance")
    messages = batch()
    apply_pending_steer_to_tool_results(agent, messages, 1)
    assert len(messages) == 3
    assert "strict guidance" in messages[-1]["content"]
    assert "legacy guidance" in messages[-1]["content"]


def test_reconnect_and_foreign_identity_fail_closed(runtime):
    owner, agent, run, frames, call = runtime
    call()
    foreign = SimpleNamespace(write=lambda frame: True)
    assert call(via=foreign)["error"]["code"] == 4001
    assert call(session_id="missing")["error"]["code"] == 4001
    assert call(run_id="wrong")["error"]["code"] == 4009
    assert call(text="changed")["error"]["code"] == 4009
    owner["transport"] = foreign  # authenticated reattachment, same live session
    assert call(via=foreign)["result"]["status"] == "accepted"
    assert call()["error"]["code"] == 4001
    server._start_inflight_turn(owner, "new turn")
    assert call(via=foreign)["error"]["code"] == 4009


def test_concurrent_duplicate_and_finish_are_atomic():
    run = StrictSteerRun()
    barrier = threading.Barrier(16)
    def submit(_):
        barrier.wait(timeout=5)
        return run.submit("id", "guidance")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        replies = list(pool.map(submit, range(16)))
    assert all(r["status"] == "accepted" for r in replies)
    messages = batch()
    barrier = threading.Barrier(2)
    def checkpoint():
        barrier.wait(timeout=5)
        run.checkpoint(messages, 1, lambda: None)
    def finish():
        barrier.wait(timeout=5)
        run.finish()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda fn: fn(), [checkpoint, finish]))
    receipt = run.submit("id", "guidance")
    assert receipt["status"] in {"delivered", "missed"}
    assert (len(messages) == 3) == (receipt["status"] == "delivered")


def test_lost_event_keeps_receipt_and_no_duplicate():
    run = StrictSteerRun()
    def disconnected(_):
        raise OSError("disconnected")
    run.emit = disconnected
    assert run.submit("id", "guidance")["status"] == "accepted"
    messages = batch()
    run.checkpoint(messages, 1, lambda: None)
    assert run.submit("id", "guidance")["status"] == "delivered"
    assert len(messages) == 3


def test_capacity_rejects_without_evicting_dedup_receipts():
    run = StrictSteerRun()
    for i in range(run.MAX_INTENTS):
        run.submit(str(i), "guidance")
    with pytest.raises(ValueError, match="capacity"):
        run.submit("overflow", "guidance")
    assert run.submit("0", "guidance")["status"] == "accepted"


def test_wire_validation_and_compute_host_fail_closed(runtime, monkeypatch):
    owner, agent, run, frames, call = runtime
    assert call(text="x" * 65537)["error"]["code"] == 4000
    assert call(intent_id="x" * 129)["error"]["code"] == 4000
    assert call(text=" ")["error"]["code"] == 4002
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda session: True)
    assert call()["error"]["code"] == 4010
    assert run.snapshot()["receipts"] == []


def test_retired_generation_does_not_publish_to_reused_runtime_id(runtime):
    owner, agent, run, frames, call = runtime
    call()
    frames.clear()
    server._sessions["ui"] = {**owner, "_strict_steer_run": StrictSteerRun()}
    run.finish()
    assert run.snapshot()["receipts"][0]["status"] == "missed"
    assert frames == []
