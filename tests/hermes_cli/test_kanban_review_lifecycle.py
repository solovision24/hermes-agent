"""Behavioral tests for the upstream-aligned native review lifecycle."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


REVIEW_METADATA = {
    "pr_url": "https://github.com/acme/repo/pull/1",
    "repo": "acme/repo",
    "number": 1,
    "head_sha": "a" * 40,
    "verification_evidence": {"tests_passed": 3},
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
            conn, task_id, reviewer="reviewer", summary="PR opened; focused tests pass",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert task.claim_lock is None
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert review.status == "running"
        assert review.assignee == "reviewer"


def test_review_changes_returns_same_card_to_implementer(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready", metadata=REVIEW_METADATA,
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        )
        assert remediation_id == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "dev"
        assert task.status == "ready"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run["status"] == "ready"
        assert run["outcome"] == "changes_requested"
        assert run["ended_at"] is not None


def test_changes_requested_reuses_same_card_for_implementer_and_re_review(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", metadata={"approved": False},
            expected_run_id=review.current_run_id,
        ) == task_id
        fix = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert fix is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="fixed",
            metadata={**REVIEW_METADATA, "head_sha": "b" * 40},
            expected_run_id=fix.current_run_id,
        )
        assert kb.get_task(conn, task_id).status == "review"


def test_dev_implementation_rerun_cannot_use_historical_review_submission(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.request_review_changes(
            conn, task_id, summary="Fix the regression test", expected_run_id=review.current_run_id
        ) == task_id
        rerun = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert rerun is not None
        assert kb.request_review_changes(
            conn, task_id, summary="I found another issue", expected_run_id=rerun.current_run_id
        ) is None
        assert kb.get_task(conn, task_id).status == "running"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? AND kind='review_changes_requested'",
            (task_id,),
        ).fetchone()["n"] == 1


def test_review_changes_rejects_delegated_child_context(board):
    from agent.delegation_context import delegated_child_context

    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        with delegated_child_context(), pytest.raises(PermissionError, match="delegate_task child"):
            kb.request_review_changes(
                conn, task_id, summary="child must not mutate the board",
                expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_review_handoff_rejects_self_reviewer_without_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="different from the implementer"):
            kb.submit_for_review(
                conn, task_id, reviewer="dev", summary="ready",
                metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_review_handoff_rejects_nonspawnable_reviewer_before_mutation(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "dev")
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="does not resolve to an on-disk Hermes profile"):
            kb.submit_for_review(
                conn, task_id, reviewer="missing", summary="ready",
                metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
            )
        task = kb.get_task(conn, task_id)
        assert task.status == "running"
        assert task.assignee == "dev"


def test_review_handoff_rejects_abbreviated_sha_before_mutation(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        with pytest.raises(ValueError, match="immutable head_sha"):
            kb.submit_for_review(
                conn, task_id, reviewer="reviewer", summary="ready",
                metadata={**REVIEW_METADATA, "head_sha": "abc123"},
                expected_run_id=implementation.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


def test_webhook_rejects_invalid_producer_without_creating_a_card(board):
    with board as conn:
        with pytest.raises(ValueError, match="full immutable 40-character"):
            kb.ingest_pull_request(
                conn, repository="acme/repo", number=9, head_sha="abc123",
                title="invalid producer",
            )
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 0


def test_webhook_replay_preserves_native_review_claim(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="native handoff",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        replay_id = kb.ingest_pull_request(
            conn, repository="ACME/REPO", number=1, head_sha="A" * 40,
            title="webhook replay", reviewer="other", checks_passed=False,
        )
        assert replay_id == task_id
        task = kb.get_task(conn, task_id)
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_webhook_first_native_submission_preserves_native_card(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=1, head_sha="a" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        assert kb.get_task(conn, webhook_id).status == "archived"
        assert kb.get_task(conn, implementation_id).status == "review"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks WHERE status != 'archived'").fetchone()["n"] == 1


def test_webhook_first_claimed_review_run_is_finalized(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=3, head_sha="c" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        assert webhook_id is not None
        webhook_review = kb.claim_review_task(conn, webhook_id, claimer="worker:reviewer")
        assert webhook_review is not None
        webhook_run_id = webhook_review.current_run_id

        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata={**REVIEW_METADATA, "pr_url": "https://github.com/acme/repo/pull/3",
                      "repo": "acme/repo", "number": 3, "head_sha": "c" * 40},
            expected_run_id=implementation.current_run_id,
        )

        merged_run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id=?",
            (webhook_run_id,),
        ).fetchone()
        assert merged_run["status"] == "review"
        assert merged_run["outcome"] == "submitted_for_review"
        assert merged_run["ended_at"] is not None
        assert kb.get_task(conn, webhook_id).status == "archived"
        assert kb.get_task(conn, implementation_id).status == "review"


def test_webhook_first_reconciliation_merges_duplicate_subscriptions(board):
    with board as conn:
        webhook_id = kb.ingest_pull_request(
            conn, repository="acme/repo", number=2, head_sha="b" * 40,
            title="Webhook review", reviewer="reviewer",
        )
        implementation_id = kb.create_task(conn, title="implement", assignee="dev")
        for task_id, cursor in ((webhook_id, 3), (implementation_id, 7)):
            conn.execute(
                "INSERT INTO kanban_notify_subs "
                "(task_id, platform, chat_id, thread_id, created_at, last_event_id) "
                "VALUES (?, 'telegram', 'same-chat', 'same-thread', 1, ?)",
                (task_id, cursor),
            )
        conn.commit()
        implementation = kb.claim_task(conn, implementation_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, implementation_id, reviewer="reviewer", summary="handoff",
            metadata={**REVIEW_METADATA, "pr_url": "https://github.com/acme/repo/pull/2",
                      "repo": "acme/repo", "number": 2, "head_sha": "b" * 40},
            expected_run_id=implementation.current_run_id,
        )
        row = conn.execute(
            "SELECT task_id, last_event_id FROM kanban_notify_subs "
            "WHERE platform='telegram' AND chat_id='same-chat' AND thread_id='same-thread'"
        ).fetchone()
        assert row["task_id"] == implementation_id
        assert row["last_event_id"] == 7


def test_review_approval_preserves_proof_and_scheduled_is_not_dispatchable(board):
    with board as conn:
        task_id = kb.create_task(conn, title="implement", assignee="dev")
        implementation = kb.claim_task(conn, task_id, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, task_id, reviewer="reviewer", summary="Evidence attached",
            metadata={**REVIEW_METADATA, "commit": "abc123", "changed_files": ["src/example.py"]},
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.complete_task(
            conn, task_id, summary="Approved after independent review",
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
