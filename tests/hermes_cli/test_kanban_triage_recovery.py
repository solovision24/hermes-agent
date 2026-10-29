"""Preserve-spec recovery uses the real CLI, board and claim path, without an LLM."""
from __future__ import annotations

import argparse
import dataclasses

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def run_cli(task_id, reason="Resolved external blocker; preserve approved scope"):
    root = argparse.ArgumentParser()
    cli.build_parser(root.add_subparsers(dest="cmd"))
    return cli.kanban_command(root.parse_args([
        "kanban", "resume-triage", task_id, "--reason", reason, "--json",
    ]))


def snapshot(conn, task_id):
    return {
        "task": dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()),
        "parents": [tuple(r) for r in conn.execute(
            "SELECT * FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
        )],
        "attachments": [dataclasses.asdict(a) for a in kb.list_attachments(conn, task_id)],
        "runs": [dict(r) for r in conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
        )],
    }


@pytest.mark.parametrize("has_parent", [False, True])
def test_resume_preserves_identity_audit_and_parent_gate(board, tmp_path, monkeypatch, has_parent):
    from hermes_cli import kanban_specify

    def forbidden(*a, **kw):
        pytest.fail("preserve-spec recovery must never call the specifier/LLM")

    monkeypatch.setattr(kanban_specify, "_call_aux", forbidden)
    monkeypatch.setattr(cli, "_profile_author", lambda: "operator")
    parent = kb.create_task(board, title="Existing acceptance owner", triage=True) if has_parent else None
    tid = kb.create_task(
        board, title="Exact C1: α  scope", body="Acceptance\r\n- [ ] Full producer proof\n\n",
        assignee="chip", triage=True, parents=[parent] if parent else [],
        skills=["hermes-agent"], workspace_kind="worktree", workspace_path=str(tmp_path / "existing"),
        branch_name="agent/chip/existing", completion_contract="local-only",
    )
    blob = tmp_path / "original-proof.bin"
    blob.write_bytes(b"original attachment\x00\xff")
    aid = kb.add_attachment(board, tid, filename=blob.name, stored_path=str(blob), size=blob.stat().st_size)
    # Recovery must retain breaker history, not manufacture a clean attempt.
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET block_recurrences = 2, consecutive_failures = 3 WHERE id = ?", (tid,))
    before = snapshot(board, tid)
    assert run_cli(tid) == 0
    after = snapshot(board, tid)
    assert after["task"].pop("status") == ("todo" if parent else "ready")
    before["task"].pop("status")
    assert after == before
    assert kb.get_attachment(board, aid).stored_path == str(blob)
    assert blob.read_bytes() == b"original attachment\x00\xff"
    events = [e for e in kb.list_events(board, tid) if e.kind == "triage_resumed"]
    assert len(events) == 1
    assert events[0].payload == {
        "author": "operator", "reason": "Resolved external blocker; preserve approved scope",
        "from_status": "triage", "to_status": "todo", "preserve_spec": True,
    }
    assert kb.list_comments(board, tid)[-1].author == "operator"
    if parent:
        assert kb.claim_task(board, tid) is None
        assert kb.archive_task(board, parent)
        assert kb.get_task(board, tid).status == "ready"
    claimed = kb.claim_task(board, tid)
    assert claimed is not None and claimed.id == tid and claimed.assignee == "chip"
    assert kb.get_task(board, tid).status == "running"


@pytest.mark.parametrize("obstruction", [
    "wrong_state", "missing", "claim_lock", "claim_expires", "worker_pid",
    "worker_started_at", "current_run_id", "open_run", "running_ended_run", "blank_reason",
])
def test_resume_refuses_without_mutation(board, obstruction):
    from hermes_cli.kanban_triage import resume_triage_task

    tid = kb.create_task(board, title="Unchanged", triage=True)
    with kb.write_txn(board):
        if obstruction == "wrong_state":
            board.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        elif obstruction in {"claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id"}:
            board.execute(f"UPDATE tasks SET {obstruction} = ? WHERE id = ?", (123, tid))
        elif obstruction in {"open_run", "running_ended_run"}:
            board.execute(
                "INSERT INTO task_runs (task_id, status, started_at, ended_at) VALUES (?, ?, 1, ?)",
                (tid, "blocked" if obstruction == "open_run" else "running",
                 None if obstruction == "open_run" else 2),
            )
    before = snapshot(board, tid)
    events = kb.list_events(board, tid)
    with pytest.raises(ValueError):
        resume_triage_task(board, "t_missing" if obstruction == "missing" else tid,
                           author="operator", reason="  " if obstruction == "blank_reason" else "Changed capability")
    assert snapshot(board, tid) == before
    assert kb.list_events(board, tid) == events
    assert kb.list_comments(board, tid) == []
    if obstruction in {"wrong_state", "blank_reason", "claim_lock", "open_run"}:
        assert run_cli(tid, reason=" " if obstruction == "blank_reason" else "Changed capability") == 1
        assert snapshot(board, tid) == before
        assert kb.list_events(board, tid) == events


def test_resume_audit_failure_rolls_back(board, monkeypatch):
    from hermes_cli.kanban_triage import resume_triage_task

    tid = kb.create_task(board, title="Exact", triage=True)
    before = snapshot(board, tid)
    events = kb.list_events(board, tid)
    def fail(*a, **kw):
        raise RuntimeError("audit storage unavailable")
    monkeypatch.setattr(kb, "_append_event", fail)
    with pytest.raises(RuntimeError, match="audit storage"):
        resume_triage_task(board, tid, author="operator", reason="Resolved blocker")
    assert snapshot(board, tid) == before
    assert kb.list_events(board, tid) == events
    assert kb.list_comments(board, tid) == []


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_resume_honors_delegated_child_fence(board, entry):
    from agent.delegation_context import delegated_child_context
    from hermes_cli.kanban_triage import resume_triage_task

    tid = kb.create_task(board, title="Exact", triage=True)
    before = snapshot(board, tid)
    with delegated_child_context():
        if entry == "api":
            with pytest.raises(PermissionError):
                resume_triage_task(board, tid, author="operator", reason="Changed capability")
        else:
            assert run_cli(tid) == 1
    assert snapshot(board, tid) == before
    assert kb.list_comments(board, tid) == []
