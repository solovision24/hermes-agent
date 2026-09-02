"""Behavior tests for durable merge/deploy closeout through the worker tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import coding_worker_lifecycle as lifecycle
from hermes_cli import kanban_db as kb


def test_closeout_tool_is_worker_gated_and_registered(monkeypatch):
    from tools import kanban_tools as kt
    from tools.registry import registry

    assert registry.get_toolset_for_tool("kanban_closeout") == "kanban"
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert kt._check_kanban_mode() is False


@pytest.fixture
def closeout_worker(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn, title="review and merge", assignee="reviewer",
            initial_status="review",
        )
        task = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert task is not None and task.claim_lock
        lifecycle.allocate_workspace(
            conn, task_id, tmp_path / "checkout",
            lease_token=task.claim_lock, now=100, ttl_seconds=300,
        )
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task_id


def test_merge_begin_is_durable_and_duplicate_suppressed(closeout_worker):
    from tools import kanban_tools as kt

    request = {
        "action": "begin",
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
        "timeout_seconds": 60,
        "max_attempts": 3,
    }
    first = json.loads(kt._handle_closeout(request))
    second = json.loads(kt._handle_closeout(request))

    assert first["ok"] is True
    assert first["started"] is True
    assert second["ok"] is True
    assert second["started"] is False
    assert second["phase"] == "merge_wait"
    assert second["wait_deadline"] == first["wait_deadline"]


def test_provider_interruption_resumes_with_bounded_durable_attempts(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    clock = iter((100, 101, 102, 103, 104))
    monkeypatch.setattr(kt.time, "time", lambda: next(clock))
    begin = {
        "action": "begin",
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
        "timeout_seconds": 60,
        "max_attempts": 2,
    }
    assert json.loads(kt._handle_closeout(begin))["started"] is True

    interrupted = json.loads(kt._handle_closeout({
        **begin,
        "action": "observe",
        "observation": "connection_error",
        "error": "provider stream interrupted",
    }))
    assert interrupted["phase"] == "merge_wait"
    assert interrupted["attempts"] == 1

    resumed = json.loads(kt._handle_closeout(begin))
    assert resumed["started"] is False
    assert resumed["attempts"] == 2
    assert resumed["wait_deadline"] == 160

    exhausted = json.loads(kt._handle_closeout({
        **begin,
        "action": "observe",
        "observation": "connection_error",
        "error": "provider unavailable again",
    }))
    assert exhausted["phase"] == "timed_out"
    assert exhausted["attempts"] == 2
    assert "attempt limit" in exhausted["last_error"]


def test_restarts_without_observations_still_exhaust_the_durable_budget(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt.time, "time", lambda: 100)
    request = {
        "action": "begin",
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
        "timeout_seconds": 60,
        "max_attempts": 2,
    }
    first = json.loads(kt._handle_closeout(request))
    recovery = json.loads(kt._handle_closeout(request))
    exhausted = json.loads(kt._handle_closeout(request))

    assert first["started"] is True and first["attempts"] == 1
    assert recovery["started"] is False and recovery["attempts"] == 2
    assert exhausted["started"] is False
    assert exhausted["phase"] == "timed_out"
    assert exhausted["attempts"] == 2
    assert "attempt limit" in exhausted["last_error"]


def test_completed_merge_closes_card_and_replay_returns_same_receipt(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt.time, "time", lambda: 100)
    operation = {
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
    }
    assert json.loads(kt._handle_closeout({
        **operation, "action": "begin", "timeout_seconds": 60,
    }))["started"] is True
    completion = {
        **operation,
        "action": "observe",
        "observation": "complete",
        "receipt": {"merge_sha": "b" * 40},
        "summary": "PR merged after focused review",
    }
    first = json.loads(kt._handle_closeout(completion))
    replay = json.loads(kt._handle_closeout(completion))

    assert first["ok"] is True
    assert first["duplicate"] is False
    assert first["receipt"] == {"merge_sha": "b" * 40}
    assert replay["ok"] is True
    assert replay["duplicate"] is True
    assert replay["receipt"] == first["receipt"]
    conn = kb.connect()
    try:
        assert kb.get_task(conn, closeout_worker).status == "done"
        assert kb.latest_run(conn, closeout_worker).outcome == "completed"
    finally:
        conn.close()


def test_deploy_begin_fails_closed_while_any_chat_is_active(
    closeout_worker, monkeypatch,
):
    from hermes_cli import active_sessions
    from tools import kanban_tools as kt

    monkeypatch.setattr(
        active_sessions,
        "active_session_registry_snapshot",
        lambda: {"telegram:chat-1": {"session_id": "active"}},
    )
    blocked = json.loads(kt._handle_closeout({
        "action": "begin",
        "kind": "deploy",
        "operation_key": "coolify:production:deploy-17",
        "timeout_seconds": 120,
    }))

    assert blocked.get("ok") is not True
    assert "active chat" in blocked["error"]
    conn = kb.connect()
    try:
        state = lifecycle.get_state(conn, closeout_worker)
        assert state is not None
        assert state.wait_kind is None
        assert state.phase == "allocated"
        assert "active chat" in (state.last_error or "")
    finally:
        conn.close()


def test_deadline_timeout_is_durable_and_terminal_observation_is_idempotent(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    clock = iter((100, 111, 112))
    monkeypatch.setattr(kt.time, "time", lambda: next(clock))
    operation = {
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
    }
    assert json.loads(kt._handle_closeout({
        **operation, "action": "begin", "timeout_seconds": 10,
        "max_attempts": 5,
    }))["started"] is True
    observation = {
        **operation,
        "action": "observe",
        "observation": "pending",
    }
    timed_out = json.loads(kt._handle_closeout(observation))
    replay = json.loads(kt._handle_closeout(observation))

    assert timed_out["phase"] == "timed_out"
    assert "deadline" in timed_out["last_error"]
    assert replay["ok"] is True
    assert replay["duplicate"] is True
    assert replay["phase"] == "timed_out"
    assert replay["attempts"] == timed_out["attempts"] == 1


def test_restart_after_deadline_times_out_before_provider_reconciliation(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    clock = iter((100, 111))
    monkeypatch.setattr(kt.time, "time", lambda: next(clock))
    request = {
        "action": "begin",
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
        "timeout_seconds": 10,
        "max_attempts": 5,
    }
    assert json.loads(kt._handle_closeout(request))["started"] is True
    resumed = json.loads(kt._handle_closeout(request))

    assert resumed["started"] is False
    assert resumed["phase"] == "timed_out"
    assert "deadline" in resumed["last_error"]
    assert resumed["attempts"] == 1


def test_closeout_limits_are_bounded_by_the_lifecycle_api(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    monkeypatch.setattr(kt.time, "time", lambda: 100)
    begun = json.loads(kt._handle_closeout({
        "action": "begin",
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
        "timeout_seconds": 10**9,
        "max_attempts": 10**9,
    }))

    assert begun["wait_deadline"] == 3700
    assert begun["max_attempts"] == 10


def test_late_completion_cannot_close_a_timed_out_operation(
    closeout_worker, monkeypatch,
):
    from tools import kanban_tools as kt

    clock = iter((100, 111, 112))
    monkeypatch.setattr(kt.time, "time", lambda: next(clock))
    operation = {
        "kind": "merge",
        "operation_key": "github:acme/repo:17:merge",
    }
    assert json.loads(kt._handle_closeout({
        **operation, "action": "begin", "timeout_seconds": 10,
    }))["started"] is True
    assert json.loads(kt._handle_closeout({
        **operation, "action": "observe", "observation": "pending",
    }))["phase"] == "timed_out"

    late = json.loads(kt._handle_closeout({
        **operation,
        "action": "observe",
        "observation": "complete",
        "receipt": {"merge_sha": "b" * 40},
        "summary": "too late",
    }))
    assert late.get("ok") is not True
    assert "timed out" in late["error"]
    conn = kb.connect()
    try:
        assert kb.get_task(conn, closeout_worker).status == "running"
    finally:
        conn.close()


def test_observation_cannot_relabel_the_reserved_provider_operation(
    closeout_worker,
):
    from tools import kanban_tools as kt

    operation_key = "github:acme/repo:17:merge"
    assert json.loads(kt._handle_closeout({
        "action": "begin",
        "kind": "merge",
        "operation_key": operation_key,
    }))["started"] is True
    mismatch = json.loads(kt._handle_closeout({
        "action": "observe",
        "kind": "deploy",
        "operation_key": operation_key,
        "observation": "pending",
    }))
    assert mismatch.get("ok") is not True
    assert "kind" in mismatch["error"]
    conn = kb.connect()
    try:
        state = lifecycle.get_state(conn, closeout_worker)
        assert state is not None and state.wait_attempts == 1
        assert state.wait_kind == "merge"
    finally:
        conn.close()
