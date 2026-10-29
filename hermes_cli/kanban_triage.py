"""Explicit triage continuation without rewriting the existing specification."""
from __future__ import annotations

import sqlite3
import time

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


_claim_fields = ("claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id")


def resume_triage_task(
    conn: sqlite3.Connection, task_id: str, *, author: str, reason: str,
) -> bool:
    """Resume one idle triage card, preserving every non-status task field.

    This is an operator disposition, not a retry/claim/review or specifier. It
    retains breaker history and existing review-resume routing. Parent gating
    runs through the same readiness path as ``specify_triage_task``. Uncertain
    claim/run state fails closed; recovery never reclaims or signals a worker.
    """
    if not author or not author.strip():
        raise ValueError("triage recovery requires an author")
    if not reason or not reason.strip():
        raise ValueError("triage recovery requires a nonblank reason")
    with kb.write_txn(conn):
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if task is None or task["status"] != "triage":
            raise ValueError("triage recovery requires an existing task in triage")
        if any(task[field] is not None for field in _claim_fields):
            raise ValueError("triage recovery refuses outstanding claim/worker/run state")
        if conn.execute(
            "SELECT 1 FROM task_runs WHERE task_id = ? "
            "AND (ended_at IS NULL OR status = 'running') LIMIT 1", (task_id,),
        ).fetchone():
            raise ValueError("triage recovery refuses an open run")
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,))
        kb._insert_comment(
            conn, task_id, author.strip(),
            "Resumed from triage without spec changes: " + reason.strip(), int(time.time()),
        )
        kb._append_event(conn, task_id, "triage_resumed", {
            "author": author.strip(), "reason": reason.strip(),
            "from_status": "triage", "to_status": "todo", "preserve_spec": True,
        })
    kb.recompute_ready(conn)
    return True


def cmd_resume_triage(args) -> int:
    from hermes_cli.kanban import _profile_author
    from hermes_cli.kanban_output import _json_out

    with kbc.connect_closing() as conn:
        resume_triage_task(conn, args.task_id, author=_profile_author(), reason=args.reason)
        task = kb.get_task(conn, args.task_id)
    data = {"task_id": args.task_id, "ok": True, "status": task.status, "preserve_spec": True}
    if not _json_out(args, data):
        print(f"Resumed {args.task_id} from triage; current status: {task.status} (spec unchanged)")
    return 0
