"""Review-claim requeue preservation.

A card in ``review`` status is claimed by ``claim_review_task`` (review ->
running) and dispatched with the mandatory ``sdlc-review`` skill.  Every
requeue path (TTL expiry, heartbeat staleness, runtime timeout, crash,
manual reclaim, spawn failure, auto-block + unblock) must return the card
to the ``review`` column so the review dispatcher re-claims it — never to
``ready``, where the ready loop would spawn it as an ordinary worker
without the review skill.
"""

from __future__ import annotations

import time

import pytest

from hermes_cli import kanban_db as kb


def _review_running_task(conn) -> str:
    """Create a task, move it to review, and claim it for review."""
    task_id = kb.create_task(conn, title="review me", assignee="worker")
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (task_id,))
    conn.commit()
    claimed = kb.claim_review_task(conn, task_id)
    assert claimed is not None
    return task_id


def _make_claim_stale(conn, task_id: str) -> None:
    """Expire the claim so release_stale_claims reclaims it."""
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET claim_expires=?, worker_pid=? WHERE id=?",
        (now - 100, 999999, task_id),
    )
    conn.commit()


def _make_heartbeat_stale(conn, task_id: str) -> None:
    """Age the run start so detect_stale_running reclaims it."""
    conn.execute(
        "UPDATE task_runs SET started_at=? WHERE id = "
        "(SELECT current_run_id FROM tasks WHERE id=?)",
        (int(time.time()) - 100, task_id),
    )
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at=NULL, worker_pid=? WHERE id=?",
        (999999, task_id),
    )
    conn.commit()


def _make_runtime_expired(conn, task_id: str) -> None:
    """Age the run past max_runtime_seconds for enforce_max_runtime."""
    conn.execute(
        "UPDATE task_runs SET started_at=? WHERE id = "
        "(SELECT current_run_id FROM tasks WHERE id=?)",
        (int(time.time()) - 100, task_id),
    )
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (999999, task_id))
    conn.commit()


def _make_crashed(conn, task_id: str) -> None:
    """Set a dead pid + old start so detect_crashed_workers reclaims it."""
    conn.execute(
        "UPDATE tasks SET worker_pid=?, started_at=? WHERE id=?",
        (999999, int(time.time()) - 300, task_id),
    )
    conn.commit()


def _status(conn, task_id: str) -> str:
    return conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()["status"]


@pytest.mark.parametrize(
    "setup, requeue",
    [
        ("ttl", lambda conn, tid: kb.release_stale_claims(conn)),
        ("heartbeat", lambda conn, tid: kb.detect_stale_running(conn, stale_timeout_seconds=1)),
        ("runtime", lambda conn, tid: kb.enforce_max_runtime(conn)),
        ("crash", lambda conn, tid: kb.detect_crashed_workers(conn)),
        ("manual", lambda conn, tid: kb.reclaim_task(conn, tid, reason="test")),
    ],
    ids=["ttl-expiry", "heartbeat-stale", "runtime-timeout", "crashed", "manual-reclaim"],
)
def test_review_claim_requeues_to_review(tmp_path, monkeypatch, setup, requeue):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        # detect_crashed_workers has a launch-window grace; disable it so the
        # dead-pid fixture is reclaimed immediately (documented test knob).
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
        task_id = _review_running_task(conn)
        assert _status(conn, task_id) == "running"
        if setup == "ttl":
            _make_claim_stale(conn, task_id)
        elif setup == "heartbeat":
            _make_heartbeat_stale(conn, task_id)
        elif setup == "runtime":
            # enforce_max_runtime only considers tasks with max_runtime_seconds set.
            conn.execute("UPDATE tasks SET max_runtime_seconds=1 WHERE id=?", (task_id,))
            conn.commit()
            _make_runtime_expired(conn, task_id)
        elif setup == "crash":
            _make_crashed(conn, task_id)
        # manual needs no prep beyond the claim.
        requeue(conn, task_id)
        assert _status(conn, task_id) == "review", (
            f"{setup}: reviewer requeue must preserve review status"
        )
    finally:
        conn.close()


def test_normal_running_claim_still_requeues_to_ready(tmp_path):
    """Control: a plain (non-review) running card still requeues to ready."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="plain", assignee="worker")
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL WHERE id=?", (task_id,)
        )
        conn.commit()
        assert kb.claim_task(conn, task_id) is not None
        assert _status(conn, task_id) == "running"
        _make_claim_stale(conn, task_id)
        kb.release_stale_claims(conn)
        assert _status(conn, task_id) == "ready"
    finally:
        conn.close()


def test_review_claim_spawn_failure_requeues_to_review(tmp_path):
    """A review claim whose spawn fails must return to review, not ready.

    The review dispatch loop records spawn failures via
    ``_record_spawn_failure`` (workspace-resolution failure or spawn
    raise).  Below the breaker threshold the card must land back in the
    ``review`` column so the review dispatcher re-claims it with the
    mandatory ``sdlc-review`` skill — the ready loop would spawn it as an
    ordinary worker with no review skill.
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = _review_running_task(conn)
        assert _status(conn, task_id) == "running"
        auto = kb._record_spawn_failure(
            conn, task_id, "spawn boom", failure_limit=5
        )
        assert auto is False
        assert _status(conn, task_id) == "review", (
            "spawn-failure: reviewer requeue must preserve review status"
        )
    finally:
        conn.close()


def test_normal_running_spawn_failure_requeues_to_ready(tmp_path):
    """Control: a plain running card's spawn failure still requeues ready."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="plain", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        assert _status(conn, task_id) == "running"
        auto = kb._record_spawn_failure(
            conn, task_id, "spawn boom", failure_limit=5
        )
        assert auto is False
        assert _status(conn, task_id) == "ready"
    finally:
        conn.close()


def test_unblock_review_card_returns_to_review(tmp_path):
    """Unblocking an auto-blocked review card returns it to review.

    Repeated spawn failures trip the breaker and park the card in
    ``blocked``.  Unblocking must consult the review lane and land the
    card back in the ``review`` column (via the event-history fallback,
    since the review run was already closed by the auto-block) so the
    review dispatcher re-claims it with the sdlc-review skill.
    """
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = _review_running_task(conn)
        auto = kb._record_spawn_failure(
            conn, task_id, "boom 1", failure_limit=1
        )
        assert auto is True
        assert _status(conn, task_id) == "blocked"
        assert kb.unblock_task(conn, task_id) is True
        assert _status(conn, task_id) == "review", (
            "unblock: auto-blocked reviewer must return to review status"
        )
    finally:
        conn.close()


def test_unblock_plain_card_returns_to_ready(tmp_path):
    """Control: a plain blocked card still unblocks to ready."""
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        task_id = kb.create_task(conn, title="plain", assignee="worker")
        conn.execute(
            "UPDATE tasks SET status='blocked', claim_lock=NULL WHERE id=?",
            (task_id,),
        )
        conn.commit()
        assert kb.unblock_task(conn, task_id) is True
        assert _status(conn, task_id) == "ready"
    finally:
        conn.close()
