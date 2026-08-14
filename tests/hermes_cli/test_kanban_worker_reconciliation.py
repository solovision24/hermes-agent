from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db = kb.connect()
    try:
        yield db
    finally:
        db.close()


def _event(conn, task_id: str, kind: str) -> dict:
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind=? ORDER BY id DESC LIMIT 1", (task_id, kind)).fetchone()
    assert row is not None
    return json.loads(row["payload"])


def test_completion_clears_active_pid_but_preserves_run_history(conn) -> None:
    task_id = kb.create_task(conn, title="complete", assignee="dev")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, 4242)
    assert kb.complete_task(conn, task_id, summary="verified", expected_run_id=claimed.current_run_id)
    task, run = kb.get_task(conn, task_id), kb.latest_run(conn, task_id)
    assert task is not None and run is not None
    assert task.status == "done" and task.worker_pid is None
    assert task.lifecycle_state == "completed" and task.last_worker_pid == 4242
    assert run.worker_pid == 4242 and run.ended_at is not None


def test_dead_pid_transition_is_complete_and_sanitized(conn, monkeypatch: pytest.MonkeyPatch) -> None:
    task_id = kb.create_task(conn, title="crash", assignee="dev", max_retries=3)
    claimed = kb.claim_task(conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:test")
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, 5252)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_resolve_crash_grace_seconds", lambda: 0)
    kb._record_worker_exit(5252, 7 << 8)
    assert kb.detect_crashed_workers(conn) == [task_id]
    task = kb.get_task(conn, task_id)
    assert task is not None
    event = _event(conn, task_id, "crashed")
    assert task.status == "ready" and task.lifecycle_state == "retry_scheduled"
    assert task.worker_pid is None and task.last_worker_pid == 5252
    assert task.last_exit_kind == "nonzero_exit" and task.last_exit_code == 7
    assert task.next_retry_at is not None
    assert event["run_id"] == claimed.current_run_id and event["prior_pid"] == 5252
    assert event["exit_kind"] == "nonzero_exit" and event["exit_code"] == 7
    assert event["error_category"] == "worker_crash"
    assert event["resulting_task_state"] == "ready" and event["retry_count"] == 1
    assert event["next_retry_at"] == task.next_retry_at
    assert event["recovery_owner"] == "dispatcher" and event["remediation"]


def test_retry_exhaustion_is_terminal_failed_not_human_blocked(conn) -> None:
    task_id = kb.create_task(conn, title="spawn", assignee="dev", max_retries=1)
    kb.claim_task(conn, task_id)
    assert kb._record_spawn_failure(conn, task_id, "provider permanently unavailable", failure_limit=1)
    task = kb.get_task(conn, task_id)
    assert task is not None
    event = _event(conn, task_id, "gave_up")
    assert task.status == "failed" and task.lifecycle_state == "terminal_failed"
    assert task.recovery_owner == "orion" and task.recovery_action and task.block_kind is None
    assert event["resulting_task_state"] == "failed" and event["error_category"] == "spawn_failure"


def test_retry_backoff_is_visible_and_dispatch_guarded(conn, monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_800_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    task_id = kb.create_task(conn, title="retry", assignee="dev", max_retries=3)
    kb.claim_task(conn, task_id)
    assert not kb._record_spawn_failure(conn, task_id, "temporary transport failure")
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert task.next_retry_at and task.next_retry_at > now
    assert kb.check_respawn_guard(conn, task_id) == "retry_backoff"
    monkeypatch.setattr(kb.time, "time", lambda: int(task.next_retry_at) + 1)
    assert kb.check_respawn_guard(conn, task_id) is None


def test_explicit_block_remains_human_blocked(conn) -> None:
    task_id = kb.create_task(conn, title="decision", assignee="dev")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.block_task(conn, task_id, reason="OAuth consent is legally required", kind="capability", expected_run_id=claimed.current_run_id)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"
    assert task.lifecycle_state == "human_blocked" and task.recovery_owner == "human"


def test_transient_block_uses_machine_retry_not_human_block(conn) -> None:
    task_id = kb.create_task(conn, title="flaky network", assignee="dev", max_retries=3)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.block_task(conn, task_id, reason="temporary network failure", kind="transient", expected_run_id=claimed.current_run_id)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert task.lifecycle_state == "retry_scheduled" and task.recovery_owner == "dispatcher"


def test_failure_evidence_redacts_secret_like_values(conn) -> None:
    task_id = kb.create_task(conn, title="redact", assignee="dev", max_retries=1)
    kb.claim_task(conn, task_id)
    raw_secret = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    kb._record_spawn_failure(conn, task_id, f"authentication failed for {raw_secret}", failure_limit=1)
    task = kb.get_task(conn, task_id)
    assert task is not None
    serialized = json.dumps(_event(conn, task_id, "gave_up"))
    assert raw_secret not in (task.last_failure_error or "") and raw_secret not in serialized


def test_legacy_machine_block_migrates_without_reclassifying_human_block(conn) -> None:
    machine_id = kb.create_task(conn, title="legacy machine", assignee="dev")
    human_id = kb.create_task(conn, title="legacy human", assignee="dev")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='blocked', lifecycle_state=NULL WHERE id IN (?, ?)", (machine_id, human_id))
        kb._append_event(conn, machine_id, "gave_up", {"error": "legacy failure"})
        kb._append_event(conn, human_id, "blocked", {"reason": "OAuth consent"})
    kb._migrate_add_optional_columns(conn)
    machine, human = kb.get_task(conn, machine_id), kb.get_task(conn, human_id)
    assert machine is not None and human is not None
    assert machine.status == "failed" and machine.lifecycle_state == "terminal_failed"
    assert human.status == "blocked" and human.lifecycle_state == "human_blocked"
