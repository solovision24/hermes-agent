"""Protected-instruction gate wiring to the hermes.approval.v1 parent-process IPC.

An MC chat child carries HERMES_APPROVAL_IPC_FD + binding env; the gate must
answer approvals on that channel (approve -> write lands, deny/timeout/close ->
fail closed with a truthful message), and non-MC processes must keep today's
gateway/CLI surfaces and the no-human fail-closed arm.
"""

import json
import socket
import threading

import pytest

from hermes_cli import approval_ipc
from tools import file_tools_write_guards as guards


@pytest.fixture
def mc_child_env(monkeypatch):
    approval_ipc._reset_channel_cache_for_tests()
    parent, child = socket.socketpair()
    monkeypatch.setenv("HERMES_APPROVAL_IPC_FD", str(child.fileno()))
    monkeypatch.setenv("HERMES_APPROVAL_IPC_PROFILE", "dev")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_SESSION", "session-1")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_RUN", "run-1")
    yield parent
    approval_ipc._reset_channel_cache_for_tests()
    parent.close()
    child.close()


def _respond(parent, choice):
    def parent_side():
        fileobj = parent.makefile("rb")
        frame = json.loads(fileobj.readline().decode("utf-8"))
        assert frame["type"] == "approval.pending"
        assert frame["approval_class"] == "protected_instruction_write"
        assert frame["choices"] == ["approve", "deny"]
        parent.sendall((json.dumps({
            "protocol": "hermes.approval.v1", "version": 1, "type": "approval.decision",
            "approval_id": frame["approval_id"], "profile": frame["profile"],
            "session": frame["session"], "run": frame["run"], "choice": choice}) + "\n").encode())
    thread = threading.Thread(target=parent_side)
    thread.start()
    return thread


def test_approve_lands_the_write(mc_child_env):
    thread = _respond(mc_child_env, "approve")
    assert guards._request_protected_instruction_approval(["AGENTS.md"]) is None
    thread.join(timeout=5)


def test_deny_fails_closed_truthfully(mc_child_env):
    thread = _respond(mc_child_env, "deny")
    result = guards._request_protected_instruction_approval(["AGENTS.md"])
    thread.join(timeout=5)
    assert result is not None
    assert "was denied by the user" in result
    assert "Do NOT retry" in result


def test_timeout_fails_closed_truthfully(mc_child_env, monkeypatch):
    """Parent stays silent: the gate must fail closed, not treat silence as consent.

    The gate calls ``request_approval`` without a deadline (the 900s default), so
    shorten the deadline at the module seam the gate looks up at call time; the
    guard's own call path is unchanged.
    """
    real_request = approval_ipc.request_approval
    monkeypatch.setattr(
        approval_ipc, "request_approval",
        lambda **kwargs: real_request(**{**kwargs, "timeout_seconds": 0.3}))
    result = guards._request_protected_instruction_approval(["SOUL.md"])
    assert result is not None
    assert "timed out without a user response" in result
    assert "Do NOT retry" in result


def test_channel_close_fails_closed_truthfully(mc_child_env):
    """Parent closes the channel (chat cancelled/killed): no consent, no CLI fallback."""
    mc_child_env.close()
    result = guards._request_protected_instruction_approval(["AGENTS.md"])
    assert result is not None
    assert "approval channel closed before a decision arrived" in result
    assert "Do NOT retry" in result


def test_no_channel_keeps_no_human_fail_closed(monkeypatch):
    approval_ipc._reset_channel_cache_for_tests()
    for var in ("HERMES_APPROVAL_IPC_FD", "HERMES_APPROVAL_IPC_PROFILE",
                "HERMES_APPROVAL_IPC_SESSION", "HERMES_APPROVAL_IPC_RUN"):
        monkeypatch.delenv(var, raising=False)
    result = guards._request_protected_instruction_approval([".cursorrules"])
    assert result is not None
    assert "no interactive user or gateway is present" in result
