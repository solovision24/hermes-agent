"""Behavioral tests for the upstream-aligned native review lifecycle."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


PR_METADATA = {
    "pr_url": "https://github.com/acme/repo/pull/8",
    "repo": "acme/repo",
    "number": 8,
    "head_sha": "a" * 40,
    "verification_evidence": {"tests": ["pytest -q"]},
}


@pytest.fixture
def board(tmp_path, monkeypatch):
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    kb.init_db()
    return kb.connect()


def test_implementation_handoff_is_claimable_by_reviewer(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn,
            task_id,
            reviewer="reviewer",
            summary="PR opened; focused tests pass",
            metadata={**PR_METADATA, "tests_run": 3},
            expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.claim_lock is None
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert review.status == "running"
        assert review.assignee == "reviewer"


def test_webhook_first_native_submission_reuses_one_canonical_card(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn,
            repository="acme/repo",
            number=8,
            head_sha="a" * 40,
            title="Webhook review",
            reviewer="reviewer",
        )
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None

        assert kb.submit_for_review(
            conn,
            implementation_id,
            reviewer="reviewer",
            summary="Native handoff after webhook",
            metadata=PR_METADATA,
            expected_run_id=implementation.current_run_id,
        )

        canonical = kb.get_task(conn, webhook_id)
        duplicate = kb.get_task(conn, implementation_id)
        assert canonical is not None
        assert duplicate is not None
        assert canonical.status == "review"
        assert duplicate.status == "archived"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key = ? AND status != 'archived'",
            ("github-pr:acme/repo:8:" + "a" * 40,),
        ).fetchone()["n"] == 1
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_runs WHERE task_id = ?",
            (webhook_id,),
        ).fetchone()["n"] == 1


def test_native_first_webhook_replay_reuses_same_card_and_preserves_review_claim(board):
    with board as conn:
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn,
            implementation_id,
            reviewer="reviewer",
            summary="Native handoff first",
            metadata=PR_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, implementation_id, claimer="worker:reviewer")
        assert review is not None

        webhook_id = kb.ingest_pull_request(
            conn,
            repository="ACME/REPO",
            number=8,
            head_sha="A" * 40,
            title="Webhook replay",
            reviewer="other",
            checks_passed=False,
        )
        replay_id = kb.ingest_pull_request(
            conn,
            repository="acme/repo",
            number=8,
            head_sha="a" * 40,
            title="Webhook replay",
            reviewer="other",
        )

        assert webhook_id == replay_id == implementation_id
        task = kb.get_task(conn, implementation_id)
        assert task is not None
        assert task.status == "running"
        assert task.assignee == "reviewer"
        assert task.current_run_id == review.current_run_id
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status != 'archived'"
        ).fetchone()["n"] == 1


def test_review_approval_completes_and_changes_create_one_remediation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready", metadata=PR_METADATA,
            expected_run_id=implementation.current_run_id
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        )
        assert remediation_id
        remediation = kb.get_task(conn, remediation_id)
        assert remediation is not None
        assert remediation.assignee == "dev"
        assert remediation.status == "ready"
        assert kb.get_task(conn, task_id).status == "done"
        # The closed review card is terminal; replaying the same reviewer run
        # cannot create a second remediation.
        assert kb.request_review_changes(conn, task_id, summary="Fix the regression test") is None
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE idempotency_key LIKE ?",
            (f"review-remediation:{task_id}:%",),
        ).fetchone()
        assert rows["n"] == 1


def test_review_approval_preserves_proof_and_scheduled_is_not_dispatchable(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn,
            task_id,
            reviewer="reviewer",
            summary="Evidence attached",
            metadata={**PR_METADATA, "commit": "abc123", "changed_files": ["src/example.py"]},
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.complete_task(
            conn,
            task_id,
            summary="Approved after independent review",
            metadata={"approved": True, "commit": "abc123"},
            expected_run_id=review.current_run_id,
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.metadata["approved"] is True

        scheduled_id = kb.create_task(conn, title="later", assignee="dev")
        assert kb.schedule_task(conn, scheduled_id, reason="wait for release")
        assert kb.claim_task(conn, scheduled_id) is None
        assert kb.get_task(conn, scheduled_id).status == "scheduled"
