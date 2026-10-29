"""Stage gates on disposable real boards; model transport fixtures are not live verdicts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as cli, kanban_db as kb, goals
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_stages import REVIEW_ADMISSION_POLICY, dependent_context


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_PROFILE", "forge")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    home = tmp_path / ".hermes" / "profiles" / "orion"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    kb.init_db()
    with connect() as conn:
        yield conn


def transport(monkeypatch, verdict="done", before_return=None):
    from agent import auxiliary_client
    calls = []
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda *_: (object(), "fixture"))
    def call_llm(**kwargs):
        calls.append(kwargs)
        if before_return:
            before_return()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "verdict": verdict, "reason": "Fixture response; not a live model verdict"})))])
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    return calls


def task_pair(conn, project):
    if project == "mc1":
        fixture = json.loads((Path(__file__).parent / "fixtures/kanban_scope_mc1.json").read_text())
        body, child_body = fixture["parent_body"], fixture["child_body"]
    else:
        body = "Implement tenant-safe export and pass required unit tests. Independent review then release live."
        child_body = "Done: publish the reviewed export build after backup, drain and rollback rehearsal; verify installed hash."
    task = kb.create_task(conn, title=project + " foundation", body=body, assignee="forge", goal_mode=True)
    child = kb.create_task(conn, title=project + " release", body=child_body, assignee="ops", parents=[task])
    return task, child, body, child_body


@pytest.mark.parametrize("surface", ["domain", "cli", "tool"])
@pytest.mark.parametrize("project", ["mc1", "unrelated-export"])
def test_all_surfaces_admit_review_without_prior_review_or_live_proof(board, monkeypatch, surface, project):
    task, child, original, child_body = task_pair(board, project)
    calls = transport(monkeypatch)
    summary = "Implementation and tenant isolation tests passed; immutable artifact ready; no independent approval or deployment yet."
    if surface == "domain":
        assert kb.request_review(board, task, summary=summary, reviewer="orion")
    elif surface == "cli":
        parser = build_parser(argparse.ArgumentParser().add_subparsers(dest="top"))
        args = parser.parse_args(["request-review", task, "--summary", summary, "--reviewer", "orion"])
        assert cli.kanban_command(args) == 0
    else:
        from tools import kanban_tools
        claimed = kb.claim_task(board, task)
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(claimed.current_run_id))
        output = json.loads(kanban_tools._handle_request_review({"summary": summary, "reviewer": "orion"}))
        assert output.get("ok"), output
    assert len(calls) == 1  # shared domain gate, never double-judge
    system, prompt = [message["content"] for message in calls[0]["messages"]]
    assert REVIEW_ADMISSION_POLICY in system
    assert original in prompt and json.dumps(child_body, ensure_ascii=False) in prompt and child in prompt
    assert kb.get_task(board, task).body == original
    assert kb.get_task(board, task).status == "review"
    assert kb.get_task(board, child).status == "todo"
    assert not [event for event in kb.list_events(board, task) if event.kind == "completed"]


@pytest.mark.parametrize("verdict", ["continue", "blocked", "wait"])
def test_real_judge_rejection_cannot_be_bypassed_by_domain(board, monkeypatch, verdict):
    task, child, _, _ = task_pair(board, "mc1")
    transport(monkeypatch, verdict)
    ok, error = kb.request_review(board, task, summary="Missing authentication regression evidence", with_reason=True)
    assert not ok and "rejected by judge" in error
    assert kb.get_task(board, task).status == "ready" and kb.get_task(board, child).status == "todo"


@pytest.mark.parametrize("metadata", [
    {"required_tests": [{"command": "pytest auth", "status": "failed"}]},
    {"required_tests": [{"command": "pytest auth", "status": "deferred"}]},
    {"required_tests": [{"command": "", "status": "passed"}]},
    {"required_tests": []}, {"required_tests": "passed"},
    {"required_tests": [{"command": "pytest auth", "status": []}]},
    {"safety_blockers": ["authorization leak"]}, {"safety_blockers": ["data loss"]},
    {"safety_blockers": ["unauthorized external action"]},
    {"head_sha": "abc123"}, {"head_sha": "A" * 40}, {"head_sha": 123},
])
def test_declared_real_failures_reject_even_non_goal_tasks(board, metadata):
    task = kb.create_task(board, title="Ordinary project", assignee="forge")
    assert not kb.request_review(board, task, summary="Cannot waive required failure", metadata=metadata)
    assert not kb.complete_task(board, task, summary="Cannot complete with required failure", metadata=metadata)
    assert kb.get_task(board, task).status == "ready"


@pytest.mark.parametrize("change", ["parent-body", "child-body", "owner", "unlink", "claim"])
def test_admission_verdict_bound_to_scope_and_owner_snapshot(board, monkeypatch, change):
    task, child, _, _ = task_pair(board, "mc1")
    def race():
        # Real board mutation between judge's network request and transition lock.
        if change == "parent-body": kb.edit_task(board, task, body="Different implementation requirements")
        if change == "child-body": kb.edit_task(board, child, body="Different release requirements")
        if change == "owner": kb.assign_task(board, child, "chip")
        if change == "unlink": kb.unlink_tasks(board, task, child)
        if change == "claim": kb.claim_task(board, task)
    transport(monkeypatch, before_return=race)
    ok, error = kb.request_review(board, task, summary="Ready", with_reason=True)
    assert not ok and "changed during admission" in error
    assert kb.get_task(board, task).status != "review"


def test_completion_keeps_full_goal_semantics_and_no_release_claim(board, monkeypatch):
    task, _, original, _ = task_pair(board, "mc1")
    calls = transport(monkeypatch, "continue")
    verdict, _ = cli._goal_mode_handoff_rejection(kb.get_task(board, task), "Source exists, but no deployment")
    assert verdict == "continue"
    assert REVIEW_ADMISSION_POLICY not in calls[0]["messages"][0]["content"]
    assert original in calls[0]["messages"][1]["content"]
    assert kb.get_task(board, task).status == "ready"


@pytest.mark.parametrize("field", ["goal", "evidence"])
def test_review_input_never_silently_truncates_requirements(board, monkeypatch, field):
    task, _, _, _ = task_pair(board, "mc1")
    if field == "goal": kb.edit_task(board, task, body="x" * 24001)
    calls = transport(monkeypatch)
    ok, error = kb.request_review(board, task, summary="x" * 24001 if field == "evidence" else "Ready", with_reason=True)
    assert not ok and "budget" in error and calls == []


@pytest.mark.parametrize("mutation", ["wrong-head", "missing-head", "missing-checks", "invalid-checks", "self-review"])
def test_exact_head_native_terminal_approval_gate(board, mutation):
    task = kb.create_task(board, title="Pinned source", assignee="forge")
    head = "a" * 40
    assert kb.request_review(board, task, summary="Ready for FIRST review", reviewer="orion", metadata={"head_sha": head})
    review = kb.claim_review_task(board, task)
    metadata = {"reviewed_head": head, "review_outcome": "approved_source_only", "reviewer_checks": ["pytest passed"]}
    if mutation == "wrong-head": metadata["reviewed_head"] = "b" * 40
    if mutation == "missing-head": metadata.pop("reviewed_head")
    if mutation == "missing-checks": metadata["reviewer_checks"] = []
    if mutation == "invalid-checks": metadata["reviewer_checks"] = [None]
    if mutation == "self-review": board.execute("UPDATE task_runs SET profile='forge' WHERE id=?", (review.current_run_id,))
    assert not kb.complete_task(board, task, summary="Not approved", metadata=metadata, expected_run_id=review.current_run_id)
    assert kb.get_task(board, task).status == "running"


def test_exact_head_independent_source_approval_releases_only_build_dependency(board):
    task = kb.create_task(board, title="Source review", assignee="forge")
    child = kb.create_task(board, title="Live release", assignee="ops", parents=[task])
    head = "a" * 40
    assert kb.request_review(board, task, summary="Ready", reviewer="orion", metadata={"head_sha": head})
    review = kb.claim_review_task(board, task)
    assert kb.complete_task(board, task, summary="Source independently approved, not deployed", metadata={
        "reviewed_head": head, "review_outcome": "approved_source_only", "reviewer_checks": ["ran pytest"]},
        expected_run_id=review.current_run_id)
    assert kb.get_task(board, child).status == "ready"  # never done/live


def test_generic_goal_judge_unchanged_and_unknown_stage_refused(monkeypatch):
    calls = transport(monkeypatch)
    assert goals.judge_goal("Produce a report", "Report exists")[0] == "done"
    assert calls[0]["messages"][0]["content"] == goals.JUDGE_SYSTEM_PROMPT
    with pytest.raises(ValueError, match="unknown"):
        goals.judge_goal("x", "y", lifecycle_stage="force_done")


def test_rereview_receives_all_canonical_findings_not_just_latest(board, monkeypatch):
    task, _, _, _ = task_pair(board, "unrelated-export")
    calls = transport(monkeypatch)
    for reason in ["Fix actual transaction happy path", "Fix phase-specific rollback and identity"]:
        assert kb.request_review(board, task, summary="Preflight evidence", reviewer="orion")
        run = kb.claim_review_task(board, task)
        assert kb.request_changes(board, task, reason=reason, expected_run_id=run.current_run_id)[0]
    assert kb.request_review(board, task, summary="Closure matrix covers both findings with exercised evidence", reviewer="orion")
    system, prompt = [item["content"] for item in calls[-1]["messages"]]
    assert "Fix actual transaction happy path" in prompt
    assert "Fix phase-specific rollback and identity" in prompt
    assert "changes_event_id" in prompt
    assert "ALL" in system and "batch all known blockers" in system
    assert "two unsuccessful remediation rounds" in system


@pytest.mark.parametrize("remote", ["same", "moved", "auth"])
@pytest.mark.parametrize("field", ["published_pr", "url"])
def test_local_only_pr_review_still_checks_live_head_as_reviewer(board, monkeypatch, remote, field):
    from hermes_cli import kanban_pr_acceptance as acceptance
    task = kb.create_task(board, title="Published source", assignee="forge")
    head, url = "a" * 40, "https://github.com/example/export/pull/17"
    assert kb.request_review(board, task, summary="Review", reviewer="orion", metadata={"head_sha": head, field: url})
    review = kb.claim_review_task(board, task)
    calls = []
    monkeypatch.setattr(acceptance, "_assignee_profile_home", lambda assignee: "reviewer-home:" + assignee)
    def api(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        if remote == "auth": raise acceptance._GateAuthError("HTTP 403")
        return {"html_url": url, "head": {"sha": head if remote == "same" else "b" * 40}}
    monkeypatch.setattr(acceptance, "_api", api)
    approved = kb.complete_task(board, task, summary="Source approved, not live", metadata={
        "reviewed_head": head, "review_outcome": "approved_source_only", "reviewer_checks": ["ran tests"]},
        expected_run_id=review.current_run_id)
    assert approved == (remote == "same")
    assert calls == [("repos/example/export/pulls/17", {"profile_home": "reviewer-home:orion"})]


@pytest.mark.parametrize("field", ["published_pr", "url"])
@pytest.mark.parametrize("head", ["missing", None, "", "abc123", "A" * 40, 123])
def test_published_pr_requires_immutable_pin_before_review(board, field, head):
    task = kb.create_task(board, title="Published source", assignee="forge")
    metadata = {field: "https://github.com/example/export/pull/17"}
    if head != "missing":
        metadata["head_sha"] = head
    ok, reason = kb.request_review(board, task, summary="Ready", reviewer="orion",
                                   metadata=metadata, with_reason=True)
    assert not ok and "head_sha" in reason
    assert kb.get_task(board, task).status == "ready"
    assert not [event for event in kb.list_events(board, task) if event.kind == "review_requested"]


@pytest.mark.parametrize("field", ["published_pr", "url"])
@pytest.mark.parametrize("head", ["missing", None, "", "abc123", "A" * 40, 123])
def test_historical_unpinned_pr_handoff_cannot_complete(board, monkeypatch, field, head):
    from hermes_cli import kanban_stages as stages, kanban_pr_acceptance as acceptance
    task = kb.create_task(board, title="Historical PR source", assignee="forge")
    metadata = {field: "https://github.com/example/export/pull/17"}
    if head != "missing":
        metadata["head_sha"] = head
    # Model a handoff accepted by the previous admission policy on a real board.
    # Only admission is replaced; terminal gates use their actual implementation.
    with monkeypatch.context() as old_policy:
        old_policy.setattr(stages, "admission_error", lambda *_: None)
        assert kb.request_review(board, task, summary="Old handoff", reviewer="orion", metadata=metadata)
    review = kb.claim_review_task(board, task)
    calls = []
    monkeypatch.setattr(acceptance, "_api", lambda *args, **kwargs: calls.append(args))
    approval = {"review_outcome": "approved_source_only", "reviewer_checks": ["read diff"]}
    assert stages.review_pin_error(board, task, approval)
    assert stages.live_review_head_error(board, task)
    assert not kb.complete_task(board, task, summary="No immutable evidence", metadata=approval,
                                expected_run_id=review.current_run_id)
    assert calls == []  # Missing pins cannot be repaired by guessing a remote head.
    assert kb.get_task(board, task).status == "running"


@pytest.mark.parametrize("metadata", [
    {"published_pr": None}, {"published_pr": ""}, {"published_pr": 123},
    {"published_pr": "https://example.com/report"},
    {"adapter": "github_pr_native_ingest", "url": "https://example.com/report"},
    {"published_pr": "https://github.com/example/export/pull/17",
     "url": "https://github.com/example/export/pull/18"},
])
def test_published_artifact_url_must_be_unambiguous(board, metadata):
    task = kb.create_task(board, title="Ambiguous PR", assignee="forge")
    metadata = dict(metadata, head_sha="a" * 40)
    assert not kb.request_review(board, task, summary="Ready", reviewer="orion", metadata=metadata)
    assert kb.get_task(board, task).status == "ready"


@pytest.mark.parametrize("metadata", [{"format": "report"}, {"url": "https://example.com/report"}])
def test_non_pr_legacy_review_keeps_existing_lifecycle(board, metadata):
    task = kb.create_task(board, title="Non-PR report", assignee="forge")
    assert kb.request_review(board, task, summary="Report ready", reviewer="orion", metadata=metadata)
    review = kb.claim_review_task(board, task)
    assert kb.complete_task(board, task, summary="Report inspected", expected_run_id=review.current_run_id)
    assert kb.get_task(board, task).status == "done"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("drift", ["parent", "child", "owner", "unlink", "onward"])
def test_ratified_deferral_revalidated_on_review_and_completion(tmp_path, monkeypatch, drift):
    # Reuse the approved candidate's exact real-MC1 fixture and owner-authorized
    # reconciliation test fixture, not a caller-generated replacement goal.
    from tests.hermes_cli.test_kanban_scope import board as scope_board, apply
    fixture = scope_board.__wrapped__(tmp_path, monkeypatch)
    state = next(fixture)
    try:
        conn, task, child, _, _, _, _ = state
        apply(state)
        if drift == "parent": kb.edit_task(conn, task, body="Goal replaced")
        if drift == "child": kb.edit_task(conn, child, body="Criteria dropped")
        if drift == "owner": kb.assign_task(conn, child, "chip")
        if drift == "unlink": kb.unlink_tasks(conn, task, child)
        if drift == "onward": conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES (?,'review_scope_reconciled','{}',1)", (child,))
        transport(monkeypatch)
        ok, error = kb.request_review(conn, task, summary="Ready", with_reason=True)
        assert not ok and "provenance" in error
        assert not kb.complete_task(conn, task, summary="Cannot silently discard program acceptance")
        assert kb.get_task(conn, task).status == "ready"
    finally:
        fixture.close()
