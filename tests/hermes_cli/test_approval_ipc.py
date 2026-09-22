"""Wire-contract tests for hermes_cli.approval_ipc (hermes.approval.v1 runtime half).

Mission Control's registry (src-tauri/src/approval_ipc.rs) validates the probe
payload exactly and validates every pending frame: exact protocol/version/type,
choices exactly ["approve","deny"], binding echo, expiry in (created, created+1h],
size caps, non-empty fields without NULs. These tests pin the Hermes side of
that contract with real socketpairs — no mocks of the transport.
"""

import json
import os
import socket
import sys
import threading

import pytest

from hermes_cli import approval_ipc


@pytest.fixture
def mc_child_env(monkeypatch):
    """Emulate MC's spawn handoff: socketpair + binding env vars on the child."""
    approval_ipc._reset_channel_cache_for_tests()
    parent, child = socket.socketpair()
    monkeypatch.setenv("HERMES_APPROVAL_IPC_FD", str(child.fileno()))
    monkeypatch.setenv("HERMES_APPROVAL_IPC_PROFILE", "dev")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_SESSION", "session-1")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_RUN", "run-1")
    yield parent, child
    approval_ipc._reset_channel_cache_for_tests()
    parent.close()
    child.close()


def _read_frame(fileobj):
    line = fileobj.readline()
    assert line.endswith(b"\n")
    return json.loads(line.decode("utf-8"))


def _decision_frame(approval_id, choice, profile="dev", session="session-1", run="run-1"):
    return {
        "protocol": "hermes.approval.v1",
        "version": 1,
        "type": "approval.decision",
        "approval_id": approval_id,
        "profile": profile,
        "session": session,
        "run": run,
        "choice": choice,
    }


def _assert_frame_matches_mc_contract(frame):
    """Mirror approval_ipc.rs::register_pending_shared's validation."""
    assert frame["protocol"] == "hermes.approval.v1"
    assert frame["version"] == 1
    assert frame["type"] == "approval.pending"
    assert frame["choices"] == ["approve", "deny"]  # exact — scope widening fails closed
    assert frame["profile"] == "dev"
    assert frame["session"] == "session-1"
    assert frame["run"] == "run-1"
    assert 0 < frame["expires_at"] - frame["created_at"] <= 3600
    for field, cap in (("approval_id", 256), ("label", 512),
                       ("description", 2000), ("approval_class", 128)):
        value = frame[field]
        assert isinstance(value, str) and value.strip()
        assert len(value) <= cap
        assert "\0" not in value


def test_probe_payload_matches_mc_validator_exactly(monkeypatch):
    """MC's validate_runtime_probe rejects anything but the exact shape."""
    probe = approval_ipc.probe()
    assert probe["protocol"] == "hermes.approval.v1"
    assert probe["version"] == 1
    assert probe["feature"] == "parent_process_approval_ipc"
    assert probe["choices"] == ["approve", "deny"]
    assert probe["available"] is False  # probe subprocess carries no channel env
    assert probe["source"] == "source_chat_required"
    # The exact command MC runs must succeed and print JSON.
    assert json.loads(os.popen(
        f'"{sys.executable}" -c "import json; from hermes_cli.approval_ipc import probe; '
        'print(json.dumps(probe()))"').read())["protocol"] == "hermes.approval.v1"


def test_probe_available_with_live_channel_env(mc_child_env):
    probe = approval_ipc.probe()
    assert probe["available"] is True
    assert probe["source"] == "parent_process_approval_ipc"


@pytest.mark.parametrize("missing", ["HERMES_APPROVAL_IPC_PROFILE",
                                       "HERMES_APPROVAL_IPC_SESSION",
                                       "HERMES_APPROVAL_IPC_RUN"])
def test_incomplete_binding_env_is_unavailable(monkeypatch, missing):
    approval_ipc._reset_channel_cache_for_tests()
    monkeypatch.setenv("HERMES_APPROVAL_IPC_FD", "1")  # stdout: open but unbound
    monkeypatch.setenv("HERMES_APPROVAL_IPC_PROFILE", "dev")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_SESSION", "session-1")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_RUN", "run-1")
    monkeypatch.delenv(missing)
    assert approval_ipc.probe()["available"] is False
    assert approval_ipc.request_approval(label="x", description="y") == \
        approval_ipc.RESULT_UNAVAILABLE


def test_closed_fd_is_unavailable(monkeypatch):
    approval_ipc._reset_channel_cache_for_tests()
    monkeypatch.setenv("HERMES_APPROVAL_IPC_FD", "198")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_PROFILE", "dev")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_SESSION", "session-1")
    monkeypatch.setenv("HERMES_APPROVAL_IPC_RUN", "run-1")
    # 198 is not open in this process.
    assert approval_ipc.probe()["available"] is False
    assert approval_ipc.request_approval(label="x", description="y") == \
        approval_ipc.RESULT_UNAVAILABLE


def test_approve_round_trip_emits_mc_contract_frame(mc_child_env):
    parent, _child = mc_child_env
    received = {}

    def parent_side():
        fileobj = parent.makefile("rb")
        frame = _read_frame(fileobj)
        received["frame"] = frame
        parent.sendall((json.dumps(_decision_frame(
            frame["approval_id"], "approve")) + "\n").encode("utf-8"))

    thread = threading.Thread(target=parent_side)
    thread.start()
    result = approval_ipc.request_approval(label="write to AGENTS.md", description="protected write")
    thread.join(timeout=5)
    assert result == approval_ipc.RESULT_APPROVE
    _assert_frame_matches_mc_contract(received["frame"])
    assert received["frame"]["approval_class"] == "protected_instruction_write"
    assert received["frame"]["label"] == "write to AGENTS.md"


def test_deny_round_trip(mc_child_env):
    parent, _child = mc_child_env

    def parent_side():
        fileobj = parent.makefile("rb")
        frame = _read_frame(fileobj)
        parent.sendall((json.dumps(_decision_frame(
            frame["approval_id"], "deny")) + "\n").encode("utf-8"))

    thread = threading.Thread(target=parent_side)
    thread.start()
    result = approval_ipc.request_approval(label="x", description="y")
    thread.join(timeout=5)
    assert result == approval_ipc.RESULT_DENY


def test_sequential_requests_share_one_channel(mc_child_env):
    """MC's reader loop serves many sequential approvals on one socketpair."""
    parent, _child = mc_child_env
    done = threading.Event()

    def parent_side():
        fileobj = parent.makefile("rb")
        for expected_choice in ("approve", "deny"):
            frame = _read_frame(fileobj)
            parent.sendall((json.dumps(_decision_frame(
                frame["approval_id"], expected_choice)) + "\n").encode("utf-8"))
        done.set()

    thread = threading.Thread(target=parent_side)
    thread.start()
    assert approval_ipc.request_approval(label="one", description="first") == \
        approval_ipc.RESULT_APPROVE
    assert approval_ipc.request_approval(label="two", description="second") == \
        approval_ipc.RESULT_DENY
    thread.join(timeout=5)
    assert done.is_set()


def test_timeout_is_not_consent(mc_child_env):
    parent, _child = mc_child_env
    # Parent stays silent: deadline must fire locally.
    result = approval_ipc.request_approval(label="x", description="y", timeout_seconds=0.3)
    assert result == approval_ipc.RESULT_TIMEOUT


def test_channel_close_is_not_consent(mc_child_env):
    parent, _child = mc_child_env
    parent.close()
    result = approval_ipc.request_approval(label="x", description="y", timeout_seconds=5)
    assert result == approval_ipc.RESULT_CLOSED


def test_stale_foreign_decision_is_skipped_not_obeyed(mc_child_env):
    """A late decision for an earlier approval id must not resolve a later request."""
    parent, _child = mc_child_env

    def parent_side():
        fileobj = parent.makefile("rb")
        frame = _read_frame(fileobj)
        # Stale: wrong approval id (simulates a late resolution of a prior
        # request whose local deadline already fired).
        parent.sendall((json.dumps(_decision_frame(
            "somebody-elses-approval", "approve")) + "\n").encode("utf-8"))
        # Real decision for this request.
        parent.sendall((json.dumps(_decision_frame(
            frame["approval_id"], "deny")) + "\n").encode("utf-8"))

    thread = threading.Thread(target=parent_side)
    thread.start()
    result = approval_ipc.request_approval(label="x", description="y")
    thread.join(timeout=5)
    assert result == approval_ipc.RESULT_DENY


def test_binding_mismatch_decision_fails_closed(mc_child_env):
    parent, _child = mc_child_env

    def parent_side():
        fileobj = parent.makefile("rb")
        frame = _read_frame(fileobj)
        parent.sendall((json.dumps(_decision_frame(
            frame["approval_id"], "approve", run="forged-run")) + "\n").encode("utf-8"))

    thread = threading.Thread(target=parent_side)
    thread.start()
    result = approval_ipc.request_approval(label="x", description="y")
    thread.join(timeout=5)
    assert result == approval_ipc.RESULT_CLOSED


def test_garbage_on_channel_fails_closed(mc_child_env):
    parent, _child = mc_child_env
    parent.sendall(b"this is not json\n")
    result = approval_ipc.request_approval(label="x", description="y", timeout_seconds=5)
    assert result == approval_ipc.RESULT_CLOSED


def test_overlong_fields_are_clipped_to_mc_caps(mc_child_env):
    parent, _child = mc_child_env
    received = {}

    def parent_side():
        fileobj = parent.makefile("rb")
        frame = _read_frame(fileobj)
        received["frame"] = frame
        parent.sendall((json.dumps(_decision_frame(
            frame["approval_id"], "approve")) + "\n").encode("utf-8"))

    thread = threading.Thread(target=parent_side)
    thread.start()
    result = approval_ipc.request_approval(
        label="x" * 5000, description="y" * 9000, approval_class="z" * 500)
    thread.join(timeout=5)
    assert result == approval_ipc.RESULT_APPROVE
    _assert_frame_matches_mc_contract(received["frame"])
