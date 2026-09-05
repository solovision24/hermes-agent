"""Cross-surface regressions for the complete Kanban review lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def review_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_PROFILE", "builder")
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="Review tool contract", assignee="builder")
        task = kb.claim_task(conn, task_id, claimer="builder:1")
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(task.current_run_id))
    return task_id


def test_review_tools_redact_handoff_and_route_changes(
    review_worker: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import kanban_tools as tools

    secret = "ghp_" + "A" * 40
    requested = json.loads(
        tools._handle_request_review({
            "summary": f"Ready; temporary token was {secret}",
            "metadata": {"token": secret, "tests_run": 7},
            "reviewer": "reviewer",
        })
    )
    assert requested["ok"] is True

    with kb.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "review"
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, review_worker)
        assert handoff is not None
        assert secret not in (handoff.summary or "")
        assert secret not in json.dumps(handoff.metadata)
        review = kb.claim_review_task(conn, review_worker, claimer="reviewer:1")
        assert review is not None

    monkeypatch.setenv("HERMES_PROFILE", "reviewer")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))
    change_secret = "sk-" + "B" * 32
    changed = json.loads(
        tools._handle_request_changes({
            "reason": f"Add a boundary assertion; leaked={change_secret}",
        })
    )
    assert changed["ok"] is True
    assert changed["implementer"] == "builder"

    with kb.connect() as conn:
        task = kb.get_task(conn, review_worker)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"
        event = [
            item
            for item in kb.list_events(conn, review_worker)
            if item.kind == "changes_requested"
        ][-1]
        assert event.payload is not None
        assert change_secret not in event.payload["reason"]
        assert event.payload["reason"] != (
            "Add a boundary assertion; leaked=" + change_secret
        )


def test_review_tools_are_gated_and_visible_to_kanban_workers(
    review_worker: str,
) -> None:
    import tools.kanban_tools  # noqa: F401 - registers the tools
    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    definitions = registry.get_definitions(
        set(resolve_toolset("hermes-cli")), quiet=True
    )
    names = {
        definition["function"]["name"]
        for definition in definitions
        if "function" in definition
    }
    assert "kanban_request_review" in names
    assert "kanban_request_changes" in names

    from acp_adapter.tools import _POLISHED_TOOLS
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert "kanban_request_changes" in _POLISHED_TOOLS
    assert "kanban_request_changes" in EXPOSED_TOOLS
    assert "kanban_request_changes" in resolve_toolset("kanban")


def test_external_github_intake_routes_changes_to_dev_and_can_re_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    head_sha = "0" * 40
    with kb.connect() as conn:
        task_id = kb.ingest_pull_request(
            conn,
            repository="solovisionllc/solorecall",
            number=525,
            head_sha=head_sha,
            title="Harden production access",
            reviewer="orion",
            url="https://github.com/SoLoVisionLLC/SoLoRecall/pull/525",
            metadata={"branch": "codex/rls-fix"},
        )
        assert task_id is not None
        task_before = kb.get_task(conn, task_id)
        assert task_before is not None
        body_before = task_before.body
        assert kb.claim_review_task(conn, task_id, claimer="orion:1") is not None
        review_run = kb.get_task(conn, task_id).current_run_id
        assert review_run is not None

        ok, implementer = kb.request_changes(
            conn,
            task_id,
            reason="Add ownership predicates and two-user regressions",
            expected_run_id=review_run,
        )
        assert (ok, implementer) == (True, "dev")
        task_after = kb.get_task(conn, task_id)
        assert task_after is not None
        assert task_after.status == "ready"
        assert task_after.assignee == "dev"
        assert task_after.body == body_before
        events = kb.list_events(conn, task_id)
        assert [event for event in events if event.kind == "review_requested"] == []
        changed = [event for event in events if event.kind == "changes_requested"][-1]
        assert changed.payload is not None
        assert changed.payload["implementer"] == "dev"
        assert changed.payload["reviewer"] == "orion"
        assert changed.payload["provenance"] == "github_pr_external_intake"
        assert changed.payload["github_pr"]["head_sha"] == head_sha

        # Replaying the identical webhook must not reverse the DEV fallback
        # back to review/orion.  This is the same-head replay regression that
        # protects queued remediation from webhook delivery retries.
        assert kb.ingest_pull_request(
            conn,
            repository="solovisionllc/solorecall",
            number=525,
            head_sha=head_sha,
            title="Harden production access",
            reviewer="orion",
            url="https://github.com/SoLoVisionLLC/SoLoRecall/pull/525",
            metadata={"branch": "codex/rls-fix"},
            action="synchronize",
        ) == task_id
        replayed = kb.get_task(conn, task_id)
        assert replayed is not None
        assert (replayed.status, replayed.assignee) == ("ready", "dev")

        # DEV remediation uses the normal implementer -> review handoff. The
        # next changes request must preserve that standard provenance.
        dev_run = kb.claim_task(conn, task_id, claimer="dev:1")
        assert dev_run is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="Ownership predicates and regressions added",
            reviewer="orion",
            expected_run_id=dev_run.current_run_id,
        )
        assert kb.claim_review_task(conn, task_id, claimer="orion:2") is not None
        second_run = kb.get_task(conn, task_id).current_run_id
        assert second_run is not None
        assert kb.request_changes(
            conn,
            task_id,
            reason="Cover the null-owner sharing case",
            expected_run_id=second_run,
        ) == (True, "dev")


def test_external_intake_replay_preserves_parent_wait_and_completed_states(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A same-head webhook cannot reopen queued or completed remediation."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    head_sha = "1" * 40
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="Upstream work", assignee="builder")
        task_id = kb.ingest_pull_request(
            conn,
            repository="solovisionllc/solorecall",
            number=526,
            head_sha=head_sha,
            title="Replay safety",
            reviewer="orion",
        )
        assert task_id is not None
        assert kb.claim_review_task(conn, task_id, claimer="orion:1") is not None
        kb.link_tasks(conn, parent_id, task_id)
        claimed = kb.get_task(conn, task_id)
        assert claimed is not None
        review_run = claimed.current_run_id
        assert review_run is not None
        assert kb.request_changes(conn, task_id, reason="Needs rework", expected_run_id=review_run) == (True, "dev")
        task = kb.get_task(conn, task_id)
        assert task is not None and (task.status, task.assignee) == ("todo", "dev")

        # The parent is still incomplete, so a replay must preserve todo.
        assert kb.ingest_pull_request(
            conn, repository="solovisionllc/solorecall", number=526,
            head_sha=head_sha, title="Replay safety", reviewer="orion",
            action="synchronize",
        ) == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None and (task.status, task.assignee) == ("todo", "dev")

        assert kb.complete_task(conn, parent_id, result="upstream complete")
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "ready"
        assert kb.complete_task(conn, task_id, result="remediation complete")

        # A completed card is also immutable under delivery retries.
        assert kb.ingest_pull_request(
            conn, repository="solovisionllc/solorecall", number=526,
            head_sha=head_sha, title="Replay safety", reviewer="orion",
            action="synchronize",
        ) == task_id
        task = kb.get_task(conn, task_id)
        assert task is not None and (task.status, task.assignee) == ("done", "dev")


def test_review_cli_round_trip_preserves_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="CLI review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:1")
        assert implementation is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))

    output = kc.run_slash(
        f"request-review {task_id} --summary 'ready for review' "
        "--reviewer reviewer --metadata '{\"tests_run\": 3}'"
    )
    assert "Requested review" in output

    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.assignee == "reviewer"
        handoff = kb.latest_run(conn, task_id)
        assert handoff is not None
        assert handoff.metadata == {"tests_run": 3}
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
        assert review is not None
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    output = kc.run_slash(
        f"request-changes {task_id} 'cover the malformed payload case'"
    )
    assert "Requested changes" in output
    with kb.connect() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "ready"
        assert task.assignee == "builder"


def test_domain_and_cli_review_handoffs_redact_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "R" * 40

    with kb.connect() as conn:
        direct_id = kb.create_task(conn, title="direct redaction", assignee="builder")
        direct_run = kb.claim_task(conn, direct_id)
        assert direct_run is not None
        assert kb.request_review(
            conn,
            direct_id,
            summary=f"direct {secret}",
            metadata={"nested": [secret]},
            expected_run_id=direct_run.current_run_id,
        )
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)

        review = kb.claim_review_task(conn, direct_id)
        assert review is not None
        assert kb.request_changes(
            conn,
            direct_id,
            reason=f"change {secret}",
            expected_run_id=review.current_run_id,
        ) == (True, "builder")
        run = kb.latest_run(conn, direct_id)
        event = [
            item for item in kb.list_events(conn, direct_id)
            if item.kind == "changes_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(event.payload)

        cli_id = kb.create_task(conn, title="CLI redaction", assignee="builder")
    cli_output = kc.run_slash(
        f'request-review {cli_id} --summary "cli {secret}" '
        f"--metadata '{{\"token\":\"{secret}\"}}'"
    )
    assert "Requested review" in cli_output
    assert secret not in cli_output
    with kb.connect() as conn:
        run = kb.latest_run(conn, cli_id)
        event = [
            item for item in kb.list_events(conn, cli_id)
            if item.kind == "review_requested"
        ][-1]
        assert run is not None
        assert secret not in str(run.summary)
        assert secret not in json.dumps(run.metadata)
        assert secret not in json.dumps(event.payload)


def test_worker_guidance_distinguishes_same_card_and_downstream_review() -> None:
    from agent.prompt_builder import KANBAN_GUIDANCE
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert "lists child IDs" in KANBAN_GUIDANCE
    assert "inspect those cards" in KANBAN_GUIDANCE
    assert "pre-created review, QA, or release child" in KANBAN_GUIDANCE
    assert "call `kanban_complete`" in KANBAN_GUIDANCE
    assert "Never sticky-block that parent for `review-required`" in KANBAN_GUIDANCE
    assert "`kanban_request_changes`" in KANBAN_GUIDANCE
    assert "metadata=..." in KANBAN_GUIDANCE
    kanban_defaults = DEFAULT_CONFIG["kanban"]
    assert isinstance(kanban_defaults, dict)
    assert kanban_defaults["review_dispatch"] is True

    repo_root = Path(__file__).resolve().parents[2]
    review_skill = repo_root / "skills" / "devops" / "sdlc-review" / "SKILL.md"
    skill_text = review_skill.read_text(encoding="utf-8")
    assert "kanban_request_changes" in skill_text
    assert "approve" in skill_text.lower()
    assert "escalate" in skill_text.lower()


def test_cli_reopen_review_is_transition_first_and_redacts_reason(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    secret = "ghp_" + "Q" * 40
    with kb.connect() as conn:
        invalid_id = kb.create_task(conn, title="not review", assignee="builder")
        review_id = kb.create_task(conn, title="review", assignee="builder")
        assert kb.request_review(conn, review_id, summary="ready")

    invalid_output = kc.run_slash(
        f'reopen-review {invalid_id} --reason "invalid {secret}"'
    )
    assert "cannot reopen" in invalid_output
    with kb.connect() as conn:
        assert kb.list_comments(conn, invalid_id) == []

    success_output = kc.run_slash(
        f'reopen-review {review_id} --reason "revise {secret}"'
    )
    assert "Reopened" in success_output
    assert secret not in success_output
    with kb.connect() as conn:
        task = kb.get_task(conn, review_id)
        assert task is not None
        assert task.status == "ready"
        comments = kb.list_comments(conn, review_id)
        assert len(comments) == 1
        assert secret not in comments[0].body


def test_goal_mode_review_handoff_cannot_bypass_judge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kb.connect() as conn:
        tool_task = kb.create_task(
            conn,
            title="Goal-mode tool task",
            assignee="builder",
            goal_mode=True,
        )
        claimed = kb.claim_task(conn, tool_task, claimer="builder:1")
        assert claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tool_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))

    from tools import kanban_tools as tools

    monkeypatch.setattr(tools, "_goal_judge_available", lambda: True)
    monkeypatch.setattr(
        tools,
        "judge_goal",
        lambda *args, **kwargs: (
            "continue",
            "acceptance evidence is missing",
            False,
            None,
            False,
        ),
    )
    rejected = json.loads(tools._handle_request_review({"summary": "Looks ready."}))
    assert "error" in rejected
    assert "rejected by judge" in rejected["error"]
    with kb.connect() as conn:
        tool_after = kb.get_task(conn, tool_task)
        assert tool_after is not None
        assert tool_after.status == "running"

    # The shell/CLI path applies the same gate and must not bypass the tool.
    with kb.connect() as conn:
        cli_task = kb.create_task(
            conn,
            title="Goal-mode CLI task",
            assignee="builder",
            goal_mode=True,
        )
        cli_claimed = kb.claim_task(conn, cli_task, claimer="builder:2")
        assert cli_claimed is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", cli_task)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(cli_claimed.current_run_id))

    import agent.auxiliary_client as auxiliary_client
    from hermes_cli import goals

    monkeypatch.setattr(
        auxiliary_client,
        "get_text_auxiliary_client",
        lambda purpose: (object(), "judge-model"),
    )
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: ("continue", "tests are missing", False, None, False),
    )
    output = kc.run_slash(f"request-review {cli_task} --summary 'Looks ready.'")
    assert "rejected by judge" in output
    with kb.connect() as conn:
        cli_after = kb.get_task(conn, cli_task)
        assert cli_after is not None
        assert cli_after.status == "running"


def test_goal_loop_stops_after_reviewer_requests_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import goals

    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *args, **kwargs: pytest.fail(
            "a terminal review verdict must not be judged"
        ),
    )
    result = goals.run_kanban_goal_loop(
        task_id="t_review",
        goal_text="review the change",
        run_turn=lambda prompt: pytest.fail("must not run another reviewer turn"),
        task_status_fn=lambda: "changes_requested",
        block_fn=lambda reason: pytest.fail("must not block"),
        first_response="Changes requested.",
    )
    assert result["outcome"] == "changes_requested_by_reviewer"
    assert result["turns_used"] == 1


def test_cli_and_dashboard_receive_graph_aware_deadlock_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()

    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="Implementation", assignee="builder")
        child_id = kb.create_task(
            conn,
            title="Review",
            assignee="reviewer",
            parents=[parent_id],
        )
        parent = kb.claim_task(conn, parent_id, claimer="builder:1")
        assert parent is not None
        assert kb.block_task(
            conn,
            parent_id,
            reason="review-required: ready",
            expected_run_id=parent.current_run_id,
        )

    payload = json.loads(kc.run_slash(f"diagnostics --task {parent_id} --json"))
    assert any(
        item["kind"] == "review_dependency_deadlock"
        for item in payload[0]["diagnostics"]
    )

    from plugins.kanban.dashboard.plugin_api import _compute_task_diagnostics

    with kb.connect() as conn:
        dashboard = _compute_task_diagnostics(conn, task_ids=[parent_id])
    assert dashboard[parent_id][0]["kind"] == "review_dependency_deadlock"
    assert dashboard[parent_id][0]["data"]["waiting_child_ids"] == [child_id]
