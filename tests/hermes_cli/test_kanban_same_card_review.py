"""Focused behavioral coverage for same-card native review remediation."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


REVIEW_METADATA = {
    "pr_url": "https://github.com/acme/repo/pull/7",
    "repo": "acme/repo",
    "number": 7,
    "head_sha": "a" * 40,
    "verification_evidence": {"tests_passed": 3},
    "exact_head_checks": True,
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    kb.init_db()
    return kb.connect()


def test_changes_requested_requeues_same_card_with_audited_findings(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        conn.execute(
            "UPDATE tasks SET consecutive_failures=2, last_failure_error=? WHERE id=?",
            ("old attempt", task_id),
        )
        conn.commit()
        assert kb.submit_for_review(
            conn,
            task_id,
            reviewer="orion",
            summary="ready for review",
            metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None

        findings = {"findings": [{"path": "src/app.py", "line": 9, "message": "fix"}]}
        assert kb.request_review_changes(
            conn,
            task_id,
            summary="Fix the review finding",
            metadata=findings,
            expected_run_id=review.current_run_id,
        ) == task_id

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "dev"
        assert task.claim_lock is None
        assert task.current_run_id is None
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM task_links").fetchone()["n"] == 0
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM tasks "
            "WHERE title LIKE 'Address review feedback:%'"
        ).fetchone()["n"] == 0

        run = conn.execute(
            "SELECT status, outcome, summary, metadata, ended_at, "
            "claim_lock, claim_expires, worker_pid "
            "FROM task_runs WHERE id=?",
            (review.current_run_id,),
        ).fetchone()
        assert run["status"] == "done"
        assert run["outcome"] == "changes_requested"
        assert run["summary"] == "Fix the review finding"
        assert '"findings"' in run["metadata"]
        assert run["ended_at"] is not None
        assert run["claim_lock"] is None
        assert run["claim_expires"] is None
        assert run["worker_pid"] is None

        event = conn.execute(
            "SELECT run_id, payload FROM task_events "
            "WHERE task_id=? AND kind='review_changes_requested'",
            (task_id,),
        ).fetchone()
        assert event["run_id"] == review.current_run_id
        assert '"findings"' in event["payload"]
        assert '"remediation_task_id"' not in event["payload"]


def test_same_card_resubmits_new_head_to_orion_and_completes(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="first head", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        first_review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert first_review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="apply findings", expected_run_id=first_review.current_run_id,
        ) == task_id

        second_implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert second_implementation is not None
        second_metadata = {**REVIEW_METADATA, "head_sha": "b" * 40}
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="second head", metadata=second_metadata,
            expected_run_id=second_implementation.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "review"
        assert kb.get_task(conn, task_id).idempotency_key.endswith("b" * 40)

        second_review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert second_review is not None
        assert kb.complete_task(
            conn,
            task_id,
            summary="approved",
            metadata={
                "approved": True,
                "checks_passed": True,
                "exact_head_checks": True,
                "reviewed_head_sha": "b" * 40,
            },
            expected_run_id=second_review.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "done"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_duplicate_changes_request_is_rejected_without_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="one finding", expected_run_id=review.current_run_id,
        ) == task_id

        before_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        before_runs = conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        assert kb.request_review_changes(
            conn, task_id, summary="duplicate finding", expected_run_id=review.current_run_id,
        ) is None
        assert kb.get_task(conn, task_id).status == "ready"
        assert kb.get_task(conn, task_id).assignee == "dev"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"] == before_events
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id=?", (task_id,)
        ).fetchone()["n"] == before_runs


def test_stale_reviewer_run_is_rejected_without_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None
        before = kb.get_task(conn, task_id)
        assert before is not None
        assert kb.request_review_changes(
            conn,
            task_id,
            summary="stale finding",
            expected_run_id=review.current_run_id + 1,
        ) is None
        after = kb.get_task(conn, task_id)
        assert after is not None
        assert (after.status, after.assignee, after.current_run_id) == (
            before.status, before.assignee, before.current_run_id
        )
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id=? AND kind='review_changes_requested'", (task_id,)
        ).fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_non_review_run_cannot_request_changes(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        before_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        assert kb.request_review_changes(
            conn,
            task_id,
            summary="not a review finding",
            expected_run_id=implementation.current_run_id,
        ) is None
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "dev"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"] == before_events


def test_review_identity_alias_is_not_a_mutation_target(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None
        alias = kb.get_task(conn, task_id).idempotency_key
        assert alias and alias != task_id
        assert kb.request_review_changes(
            conn, alias, summary="alias must fail", expected_run_id=review.current_run_id,
        ) is None
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.current_run_id == review.current_run_id
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_preseeded_legacy_remediation_key_fails_closed(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None
        preseeded_id = kb.create_task(conn, title="untrusted", assignee="dev")
        conn.execute(
            "UPDATE tasks SET idempotency_key=? WHERE id=?",
            (f"review-remediation:{task_id}:{review.current_run_id}", preseeded_id),
        )
        conn.commit()
        before_events = conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"]
        assert kb.request_review_changes(
            conn, task_id, summary="must fail closed", expected_run_id=review.current_run_id,
        ) is None
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "orion"
        assert task.current_run_id == review.current_run_id
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 2
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (task_id,)
        ).fetchone()["n"] == before_events


def test_archived_historical_remediation_key_does_not_block_same_card_rejection(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None

        historical_id = kb.create_task(conn, title="historical remediation", assignee="dev")
        conn.execute(
            "UPDATE tasks SET status='archived', idempotency_key=? WHERE id=?",
            (f"review-remediation:{task_id}:{review.current_run_id - 1}", historical_id),
        )
        conn.commit()

        assert kb.request_review_changes(
            conn, task_id, summary="current finding", expected_run_id=review.current_run_id,
        ) == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "dev"
        assert kb.get_task(conn, historical_id).status == "archived"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 2


def test_second_connection_cannot_replay_a_completed_changes_request(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="orion", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:orion")
        assert review is not None
        contender = kb.connect()
        try:
            assert kb.request_review_changes(
                contender, task_id, summary="first writer", expected_run_id=review.current_run_id,
            ) == task_id
        finally:
            contender.close()
        assert kb.request_review_changes(
            conn, task_id, summary="second writer", expected_run_id=review.current_run_id,
        ) is None
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id=? AND kind='review_changes_requested'", (task_id,)
        ).fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
