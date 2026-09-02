"""Tests: orphaned-card reconciliation for the kanban dispatcher.

Tracked-state vs. reality divergence: a task can sit in ``status='running'``
with broken claim bookkeeping — ``claim_lock IS NULL`` or ``claim_expires IS
NULL`` (crash mid-claim, manual SQL, DB restore, partial migration). None of
the existing recovery paths ever touch such a card:

- ``release_stale_claims`` requires ``claim_expires IS NOT NULL``;
- ``detect_crashed_workers`` requires a host-local ``claim_lock`` prefix and
  a recorded ``worker_pid``;
- ``detect_stale_running`` is disabled by default (``stale_timeout=0``).

Result: a zombie card that shows Running forever. ``reconcile_orphaned_running``
is the reconciliation pass: it finds those orphans, requeues them to ``ready``
with an explanatory note, and logs a ``reconciled`` event. Wired into
``dispatch_once`` each tick, gated by ``kanban.reconcile_orphans`` (config.yaml,
default on) at the gateway watcher layer.

Inspired by openai/symphony's tracker reconciliation (Apache-2.0), idea-level.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def _orphan_running(conn, tid, *, claim_lock=None, claim_expires=None,
                    worker_pid=None):
    """Force a task into running with (partially) broken claim bookkeeping."""
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=? WHERE id=?",
        (claim_lock, claim_expires, worker_pid, tid),
    )
    conn.commit()


class TestReconcileOrphanedRunning:
    def test_null_claim_lock_orphan_requeued(self, conn):
        """running + claim_lock NULL → requeued to ready with a note."""
        tid = kb.create_task(conn, title="zombie", assignee="w")
        _orphan_running(conn, tid)

        reconciled = kb.reconcile_orphaned_running(conn)

        assert reconciled == [tid]
        row = conn.execute(
            "SELECT status, claim_lock, claim_expires, worker_pid "
            "FROM tasks WHERE id=?", (tid,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["claim_expires"] is None
        assert row["worker_pid"] is None

    def test_null_claim_expires_orphan_requeued(self, conn):
        """running + claim_lock set but claim_expires NULL is also invisible
        to release_stale_claims — reconciliation must catch it."""
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="half-claim", assignee="w")
        _orphan_running(conn, tid, claim_lock=f"{host}:dead")

        reconciled = kb.reconcile_orphaned_running(conn)

        assert reconciled == [tid]
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "ready"

    def test_reconciled_event_and_note_logged(self, conn):
        tid = kb.create_task(conn, title="zombie", assignee="w")
        _orphan_running(conn, tid)

        kb.reconcile_orphaned_running(conn)

        events = kb.list_events(conn, tid)
        recon = [e for e in events if e.kind == "reconciled"]
        assert len(recon) == 1
        assert recon[0].payload["reason"] == "orphaned_running"
        comments = kb.list_comments(conn, tid)
        assert any("reconcil" in (c.body or "").lower() for c in comments)

    def test_healthy_running_task_untouched(self, conn):
        """A properly claimed running task is NOT an orphan."""
        tid = kb.create_task(conn, title="healthy", assignee="w")
        kb.claim_task(conn, tid)

        assert kb.reconcile_orphaned_running(conn) == []
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "running"

    def test_live_worker_pid_defers_reconcile(self, conn):
        """If the orphan row still records a live PID on this host, don't
        requeue beside a possibly-alive worker — defer to the next tick."""
        tid = kb.create_task(conn, title="maybe-alive", assignee="w")
        sleeper = subprocess.Popen(["sleep", "30"])
        try:
            _orphan_running(conn, tid, worker_pid=sleeper.pid)
            assert kb.reconcile_orphaned_running(conn) == []
            assert conn.execute(
                "SELECT status FROM tasks WHERE id=?", (tid,)
            ).fetchone()["status"] == "running"
        finally:
            sleeper.terminate()
            sleeper.wait()

    def test_dead_worker_pid_orphan_requeued(self, conn):
        """Orphan with a recorded but dead PID is reconciled."""
        tid = kb.create_task(conn, title="dead-pid", assignee="w")
        dead = subprocess.Popen(["true"])
        dead.wait()
        _orphan_running(conn, tid, worker_pid=dead.pid)

        assert kb.reconcile_orphaned_running(conn) == [tid]

    def test_non_running_statuses_ignored(self, conn):
        for status in ("todo", "ready", "blocked", "done"):
            tid = kb.create_task(conn, title=f"s-{status}", assignee="w")
            conn.execute(
                "UPDATE tasks SET status=?, claim_lock=NULL, "
                "claim_expires=NULL WHERE id=?", (status, tid),
            )
        conn.commit()
        assert kb.reconcile_orphaned_running(conn) == []

    def test_reconciliation_closes_durable_worker_lease(self, conn, tmp_path):
        from hermes_cli.coding_worker_lifecycle import allocate_workspace

        tid = kb.create_task(conn, title="leased orphan", assignee="w")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        allocate_workspace(
            conn, tid, tmp_path / "workspace", lease_token=claimed.claim_lock,
            now=10, ttl_seconds=60,
        )
        _orphan_running(conn, tid, claim_lock=None, claim_expires=None)

        assert kb.reconcile_orphaned_running(conn) == [tid]
        state = conn.execute(
            "SELECT phase, lease_token, lease_expires "
            "FROM coding_worker_lifecycle WHERE task_id=?", (tid,),
        ).fetchone()
        assert tuple(state) == ("reclaimed", None, None)


class TestDispatchOnceReconciles:
    def test_dispatch_once_reconciles_orphans(self, conn):
        tid = kb.create_task(conn, title="zombie", assignee="w")
        _orphan_running(conn, tid)

        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: (True, ""),
                                  dry_run=True)

        assert tid in result.reconciled_orphans
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "ready"

    def test_dispatch_once_reconcile_can_be_disabled(self, conn):
        """kanban.reconcile_orphans=false plumbs through as
        reconcile_orphans=False and skips the pass."""
        tid = kb.create_task(conn, title="zombie", assignee="w")
        _orphan_running(conn, tid)

        result = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: (True, ""),
                                  dry_run=True, reconcile_orphans=False)

        assert result.reconciled_orphans == []
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (tid,)
        ).fetchone()["status"] == "running"


def test_provider_connection_failure_and_gateway_restart_resume_one_wait(
    conn, tmp_path,
):
    """A fresh connection retains the operation key and original deadline."""
    from hermes_cli.coding_worker_lifecycle import (
        allocate_workspace,
        begin_wait,
        observe_wait,
    )

    tid = kb.create_task(conn, title="merge closeout", assignee="worker")
    allocate_workspace(
        conn, tid, tmp_path / "workspace", lease_token="lease-1",
        now=10, ttl_seconds=60,
    )
    initial, started = begin_wait(
        conn, tid, kind="merge", operation_key="merge:abc",
        now=20, timeout_seconds=30,
    )
    assert started is True
    assert initial.wait_deadline == 50
    failed = observe_wait(
        conn, tid, operation_key="merge:abc", observation="connection_error",
        now=25, error="connection refused",
    )
    assert failed.phase == "merge_wait"

    with kb.connect_closing() as restarted:
        resumed, started_again = begin_wait(
            restarted, tid, kind="merge", operation_key="merge:abc",
            now=40, timeout_seconds=300,
        )
        assert started_again is False
        assert resumed.wait_deadline == 50
        assert resumed.last_error == "connection refused"
        observe_wait(
            restarted, tid, operation_key="merge:abc", observation="complete",
            now=45, receipt={"merge_sha": "b" * 40},
        )
        _done, duplicate = begin_wait(
            restarted, tid, kind="merge", operation_key="merge:abc",
            now=46, timeout_seconds=30,
        )
        assert duplicate is False


def test_deploy_wait_is_fail_closed_while_any_chat_is_active(conn, tmp_path):
    from hermes_cli.coding_worker_lifecycle import allocate_workspace, begin_wait

    tid = kb.create_task(conn, title="deploy closeout", assignee="worker")
    allocate_workspace(
        conn, tid, tmp_path / "workspace", lease_token="lease-1",
        now=10, ttl_seconds=60,
    )

    blocked, started = begin_wait(
        conn, tid, kind="deploy", operation_key="deploy:abc",
        now=20, timeout_seconds=60, active_chat_count=1,
    )
    assert started is False
    assert blocked.wait_kind is None
    assert "active chat" in (blocked.last_error or "")

    active, started = begin_wait(
        conn, tid, kind="deploy", operation_key="deploy:abc",
        now=30, timeout_seconds=60, active_chat_count=0,
    )
    retry, started_again = begin_wait(
        conn, tid, kind="deploy", operation_key="deploy:abc",
        now=40, timeout_seconds=600, active_chat_count=0,
    )
    assert started is True and started_again is False
    assert active.wait_deadline == retry.wait_deadline == 90


def test_timeout_requires_two_consecutive_silent_watchdog_checks(conn, tmp_path):
    from hermes_cli.coding_worker_lifecycle import (
        allocate_workspace,
        begin_wait,
        observe_wait,
    )

    tid = kb.create_task(conn, title="silent deploy", assignee="worker")
    allocate_workspace(
        conn, tid, tmp_path / "workspace", lease_token="lease-1",
        now=10, ttl_seconds=60,
    )
    begin_wait(
        conn, tid, kind="deploy", operation_key="deploy:quiet",
        now=20, timeout_seconds=10, active_chat_count=0,
    )
    first = observe_wait(
        conn, tid, operation_key="deploy:quiet", observation="silent", now=31,
    )
    # Any provider response breaks the silence streak without moving the
    # original deadline; two new silent observations are then required.
    pending = observe_wait(
        conn, tid, operation_key="deploy:quiet", observation="pending", now=32,
    )
    second_first = observe_wait(
        conn, tid, operation_key="deploy:quiet", observation="silent", now=33,
    )
    second = observe_wait(
        conn, tid, operation_key="deploy:quiet", observation="silent", now=34,
    )

    assert first.phase == "deploy_wait" and first.silent_checks == 1
    assert pending.silent_checks == 0
    assert second_first.phase == "deploy_wait" and second_first.silent_checks == 1
    assert second.phase == "timed_out" and second.silent_checks == 2
    assert second.wait_deadline == 30
