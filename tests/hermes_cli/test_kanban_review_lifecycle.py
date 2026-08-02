"""Behavioral tests for the upstream-aligned native review lifecycle."""

from pathlib import Path
from argparse import Namespace

import pytest

from hermes_cli import kanban_db as kb


REVIEW_METADATA = {
    "pr_url": "https://github.com/acme/repo/pull/1",
    "repo": "acme/repo",
    "number": 1,
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
            metadata=REVIEW_METADATA,
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


def test_review_changes_creates_one_idempotent_remediation_child(board):
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
        assert remediation_id != task_id
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "reviewer"
        assert task.status == "done"
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks",
        ).fetchone()
        assert rows["n"] == 2
        run = conn.execute(
            "SELECT status, outcome, ended_at FROM task_runs "
            "WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run["status"] == "done"
        assert run["outcome"] == "changes_requested"
        assert run["ended_at"] is not None


def test_review_changes_remediation_is_ready_after_parent_done(board, capsys):
    from hermes_cli import kanban as cli

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

        args = Namespace(
            task_id=task_id, summary=["Fix", "the", "regression"], metadata=None
        )
        assert cli._cmd_review_changes(args) == 0
        assert "remediation status: ready" in capsys.readouterr().out
        remediation = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id=?", (task_id,)
        ).fetchone()
        assert remediation is not None
        child = kb.get_task(conn, remediation["child_id"])
        assert child is not None
        assert child.status == "ready"


def test_remediation_child_can_be_reviewed_again(board, monkeypatch):
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
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

        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test",
            metadata={"approved": False}, expected_run_id=review.current_run_id,
        )
        assert remediation_id != task_id

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.assignee == "reviewer"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 2

        fix = kb.claim_task(conn, remediation_id, claimer="worker:dev")
        assert fix is not None
        assert kb.submit_for_review(
            conn, remediation_id, reviewer="reviewer", summary="fixed",
            metadata={**REVIEW_METADATA, "head_sha": "b" * 40},
            expected_run_id=fix.current_run_id,
        )
        assert kb.get_task(conn, remediation_id).status == "review"


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
        remediation_id = kb.request_review_changes(
            conn, task_id, summary="Fix the regression test",
            expected_run_id=review.current_run_id,
        )
        assert remediation_id != task_id

        assert kb.request_review_changes(
            conn, task_id, summary="I found another issue",
        ) is None
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        assert task.assignee == "reviewer"
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id=? AND kind='review_changes_requested'",
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
        with pytest.raises(ValueError, match="does not exist"):
            kb.submit_for_review(
                conn, task_id, reviewer="missing", summary="ready",
                metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
            )
        task = kb.get_task(conn, task_id)
        assert task is not None
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
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


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
            metadata={**REVIEW_METADATA, "commit": "abc123", "changed_files": ["src/example.py"]},
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, task_id, claimer="worker:reviewer")
        assert review is not None
        assert kb.complete_task(
            conn,
            task_id,
            summary="Approved after independent review",
            metadata={
                "approved": True,
                "checks_passed": True,
                "exact_head_checks": True,
                "reviewed_head_sha": "a" * 40,
                "commit": "abc123",
            },
            expected_run_id=review.current_run_id,
        )
        run = kb.latest_run(conn, task_id)
        assert run is not None
        assert run.metadata["approved"] is True


@pytest.mark.parametrize("exact_head_checks", [None, False])
def test_review_approval_requires_exact_head_checks(board, exact_head_checks):
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
        metadata = {"approved": True, "checks_passed": True}
        if exact_head_checks is not None:
            metadata["exact_head_checks"] = exact_head_checks
        with pytest.raises(ValueError, match="exact_head_checks=true"):
            kb.complete_task(
                conn, task_id, summary="missing exact-head proof",
                metadata=metadata, expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize("reviewed_head_sha", [None, "b" * 40])
def test_review_approval_binds_to_submitted_head(board, reviewed_head_sha):
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
        metadata = {
            "approved": True,
            "checks_passed": True,
            "exact_head_checks": True,
        }
        if reviewed_head_sha is not None:
            metadata["reviewed_head_sha"] = reviewed_head_sha
        with pytest.raises(ValueError, match="reviewed_head_sha"):
            kb.complete_task(
                conn, task_id, summary="unbound review", metadata=metadata,
                expected_run_id=review.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize("terminal_state", ["stale_run", "archived_task"])
def test_review_changes_rejects_stale_or_archived_races(board, terminal_state):
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
        if terminal_state == "stale_run":
            conn.execute(
                "UPDATE task_runs SET status='done', ended_at=1 WHERE id=?",
                (review.current_run_id,),
            )
        else:
            conn.execute(
                "UPDATE tasks SET status='archived', completed_at=1 WHERE id=?",
                (task_id,),
            )
        conn.commit()
        assert kb.request_review_changes(
            conn, task_id, summary="late changes", expected_run_id=review.current_run_id,
        ) is None
        assert conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"] == 1


def test_changes_requested_event_cannot_complete_or_promote_dependency(board):
    """The incident event shape is never a successful dependency signal."""
    with board as conn:
        parent = kb.create_task(conn, title="review", assignee="dev")
        child = kb.create_task(conn, title="downstream", assignee="dev", parents=[parent])
        implementation = kb.claim_task(conn, parent, claimer="worker:dev")
        assert implementation is not None
        assert kb.submit_for_review(
            conn, parent, reviewer="reviewer", summary="ready",
            metadata=REVIEW_METADATA, expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, parent, claimer="worker:reviewer")
        assert review is not None
        remediation_id = kb.request_review_changes(
            conn,
            parent,
            summary="changes required",
            metadata={
                "approved": False,
                "changes_requested": True,
                "review_state": "COMMENTED",
                "checks_passed": False,
            },
            expected_run_id=review.current_run_id,
        )
        assert remediation_id != parent
        assert kb.get_task(conn, child).status == "todo"

        assert kb.get_task(conn, parent).status == "done"


def test_changes_requested_run_outcome_does_not_satisfy_dependency(board):
    """A goal-mode/legacy race ending a card with this outcome is not success."""
    with board as conn:
        parent = kb.create_task(conn, title="review", assignee="dev")
        child = kb.create_task(conn, title="downstream", assignee="dev", parents=[parent])
        claimed = kb.claim_task(conn, parent, claimer="worker:dev")
        assert claimed is not None
        conn.execute(
            "UPDATE tasks SET status='done', current_run_id=? WHERE id=?",
            (claimed.current_run_id, parent),
        )
        conn.execute(
            "UPDATE task_runs SET status='done', outcome='changes_requested', ended_at=1 "
            "WHERE id=?",
            (claimed.current_run_id,),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        conn.commit()
        assert kb.claim_task(conn, child, claimer="worker:dev") is None
        assert kb.get_task(conn, child).status == "todo"

        scheduled_id = kb.create_task(conn, title="later", assignee="dev")
        assert kb.schedule_task(conn, scheduled_id, reason="wait for release")
        assert kb.claim_task(conn, scheduled_id) is None
        assert kb.get_task(conn, scheduled_id).status == "scheduled"
