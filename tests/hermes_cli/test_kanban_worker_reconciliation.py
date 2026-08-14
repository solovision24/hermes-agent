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


def _irreducible_human_gate() -> dict:
    return {
        "category": "identity_or_oauth_consent",
        "exhausted_paths": [
            {"stage": stage, "evidence": f"verified {stage} cannot resolve this consent"}
            for stage in kb.AUTONOMOUS_RECOVERY_STAGES
        ],
        "exact_ask": "Complete the provider OAuth consent screen.",
        "proposed_default": "Keep the task paused without changing providers.",
    }


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


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        ("spawn_failed", "worker startup failed"),
        ("spawn_failed", "provider quota exhausted"),
        ("crashed", "repository worktree is unavailable"),
        ("timed_out", "CI verification timed out"),
    ],
)
def test_machine_failure_exhaustion_is_agent_owned_not_human_blocked(
    conn, outcome: str, message: str
) -> None:
    task_id = kb.create_task(conn, title=message, assignee="dev", max_retries=1)
    kb.claim_task(conn, task_id)
    assert kb._record_task_failure(
        conn,
        task_id,
        message,
        outcome=outcome,
        failure_limit=1,
        release_claim=True,
        end_run=True,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    event = _event(conn, task_id, "gave_up")
    assert task.status == "failed" and task.lifecycle_state == "terminal_failed"
    assert task.recovery_owner == "orion" and task.recovery_action
    assert task.block_kind is None and task.recovery_owner != "human"
    assert event["resulting_task_state"] == "failed"
    assert event["recovery_owner"] == "orion"
    assert event["autonomous_recovery_stages"] == list(kb.AUTONOMOUS_RECOVERY_STAGES)


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
    assert kb.block_task(
        conn,
        task_id,
        reason="OAuth consent is legally required",
        kind="capability",
        human_gate=_irreducible_human_gate(),
        expected_run_id=claimed.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "blocked"
    assert task.lifecycle_state == "human_blocked" and task.recovery_owner == "human"
    blocked = _event(conn, task_id, "blocked")
    assert blocked["human_gate"]["category"] == "identity_or_oauth_consent"
    assert blocked["human_gate"]["exact_ask"]
    assert blocked["human_gate"]["proposed_default"]


@pytest.mark.parametrize(
    ("kind", "expected_reason"),
    [
        (None, "human_gate_kind_required"),
        ("needs_input", "missing_structured_human_gate"),
        ("capability", "missing_structured_human_gate"),
    ],
)
def test_unstructured_human_gate_is_rejected_to_agent_owned_recovery(
    conn, kind, expected_reason
) -> None:
    task_id = kb.create_task(conn, title="ordinary recovery", assignee="dev")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.block_task(
        conn,
        task_id,
        reason="tests are failing; need approval",
        kind=kind,
        expected_run_id=claimed.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "failed" and task.lifecycle_state == "terminal_failed"
    assert task.recovery_owner == "orion" and task.block_kind is None
    rejected = _event(conn, task_id, "human_gate_rejected")
    assert rejected["reason"] == expected_reason


def test_human_gate_category_must_be_irreducible_and_paths_ordered(conn) -> None:
    task_id = kb.create_task(conn, title="bad escalation", assignee="dev")
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    gate = _irreducible_human_gate()
    gate["category"] = "code_review"
    gate["exhausted_paths"] = list(reversed(gate["exhausted_paths"]))
    assert kb.block_task(
        conn,
        task_id,
        reason="maintainer should review",
        kind="needs_input",
        human_gate=gate,
        expected_run_id=claimed.current_run_id,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "failed"
    rejected = _event(conn, task_id, "human_gate_rejected")
    assert rejected["reason"] in {"category_not_irreducible", "autonomous_paths_not_exhausted"}


def test_transient_block_uses_machine_retry_not_human_block(conn) -> None:
    task_id = kb.create_task(conn, title="flaky network", assignee="dev", max_retries=3)
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    assert kb.block_task(conn, task_id, reason="temporary network failure", kind="transient", expected_run_id=claimed.current_run_id)
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "ready"
    assert task.lifecycle_state == "retry_scheduled" and task.recovery_owner == "dispatcher"


def test_stale_run_cannot_terminal_fail_replacement_run(conn) -> None:
    task_id = kb.create_task(conn, title="ownership race", assignee="dev")
    stale = kb.claim_task(conn, task_id, claimer="dev:stale")
    assert stale is not None and stale.current_run_id is not None
    assert kb.reclaim_task(conn, task_id, reason="replace stale worker")
    replacement = kb.claim_task(conn, task_id, claimer="dev:replacement")
    assert replacement is not None and replacement.current_run_id is not None

    recorded = kb._record_task_failure(
        conn,
        task_id,
        "stale worker attempted terminal failure",
        outcome="human_gate_misuse",
        force_trip=True,
        release_claim=True,
        end_run=True,
        expected_run_id=stale.current_run_id,
    )

    task = kb.get_task(conn, task_id)
    assert recorded is None
    assert task is not None and task.status == "running"
    assert task.current_run_id == replacement.current_run_id
    assert not [event for event in kb.list_events(conn, task_id) if event.kind == "gave_up"]


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
