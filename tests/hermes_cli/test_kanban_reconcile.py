"""Owner reconciliation uses the real Kanban lifecycle and a bounded GitHub API fixture."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_reconcile as rec
from hermes_cli.kanban_db_connect import connect

PR = "https://github.com/acme/repo/pull/182"
HEAD = "a" * 40
MERGE = "b" * 40
REVIEW = PR + "#issuecomment-1234"


@pytest.fixture
def board(tmp_path, monkeypatch):
    import pwd
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(rec.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    kb.init_db()
    with connect() as conn:
        task = kb.create_task(conn, title="Exact PR", body="original evidence", completion_contract=PR)
        child = kb.create_task(conn, title="Release", parents=[task])
        yield conn, task, child, pwd.getpwuid(os.geteuid()).pw_name


def options(task, operator):
    return dict(operator=operator, confirm=task, pr_url=PR, head_sha=HEAD,
                merge_sha=MERGE, review_url=REVIEW, reason="Owner reviewed exact merged PR; private hosted policy inaccessible")


def fake_github(monkeypatch, *, merged=True, head=HEAD, checks=0, status=None,
                review_head=HEAD, private=True, rules_unavailable=True):
    from hermes_cli.kanban_pr_acceptance import _GateAuthError

    calls = []
    def api(endpoint):
        calls.append(endpoint)
        if endpoint == "repos/acme/repo/pulls/182":
            return {"html_url": PR, "head": {"sha": head}, "merge_commit_sha": MERGE,
                    "merged": merged, "merged_at": "2026-10-27T20:00:00Z" if merged else None,
                    "state": "closed" if merged else "open",
                    "base": {"ref": "main", "repo": {"full_name": "acme/repo"}}}
        if endpoint == "repos/acme/repo/issues/comments/1234":
            return {"issue_url": "https://api.github.com/repos/acme/repo/issues/182",
                    "body": "Independent review of exact head " + review_head}
        if endpoint == "repos/acme/repo":
            return {"full_name": "acme/repo", "private": private}
        if endpoint.endswith("check-runs?per_page=100"):
            return {"total_count": checks}
        if endpoint.endswith("statuses?per_page=100"):
            return status or []
        if endpoint.endswith("rules/branches/main?per_page=100"):
            if rules_unavailable:
                raise _GateAuthError("HTTP 403 on private rules")
            return []
        raise AssertionError(endpoint)
    monkeypatch.setattr(rec, "_api", api)
    return calls


@pytest.mark.platforms("posix")
def test_authorization_refuses_wrong_account_confirmation_worker_and_noninteractive(board, monkeypatch):
    conn, task, _, actor = board
    fake_github(monkeypatch)
    with pytest.raises(PermissionError, match="OS account"):
        rec.reconcile_pr_contract(conn, task, **options(task, "not-" + actor))
    with pytest.raises(PermissionError, match="confirm"):
        rec.reconcile_pr_contract(conn, task, **{**options(task, actor), "confirm": "t_other"})
    monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    with pytest.raises(PermissionError, match="workers"):
        rec.reconcile_pr_contract(conn, task, **options(task, actor))
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.setattr(rec.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(PermissionError, match="interactive"):
        rec.reconcile_pr_contract(conn, task, **options(task, actor))
    assert kb.get_task(conn, task).completion_contract == PR


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["wrong-pr", "wrong-head", "wrong-merge", "not-merged", "missing-review", "hosted-check", "rules-readable"])
def test_mismatched_or_missing_evidence_never_changes_contract(board, monkeypatch, fault):
    conn, task, _, actor = board
    opts = options(task, actor)
    settings = {}
    if fault == "wrong-pr": opts["pr_url"] = PR.replace("182", "181")
    if fault == "wrong-head": settings["head"] = "c" * 40
    if fault == "wrong-merge": opts["merge_sha"] = "c" * 40
    if fault == "not-merged": settings["merged"] = False
    if fault == "missing-review": settings["review_head"] = "c" * 40
    if fault == "hosted-check": settings["checks"] = 1
    if fault == "rules-readable": settings["rules_unavailable"] = False
    fake_github(monkeypatch, **settings)
    with pytest.raises(ValueError):
        rec.reconcile_pr_contract(conn, task, **opts)
    assert kb.get_task(conn, task).completion_contract == PR
    assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_contract_reconciled'", (task,)).fetchone()[0] == 0


@pytest.mark.platforms("posix")
def test_exact_merge_idempotence_and_normal_completion_promotes_child(board, monkeypatch):
    conn, task, child, actor = board
    calls = fake_github(monkeypatch)
    args = options(task, actor)
    before = kb.get_task(conn, task).body
    first = rec.reconcile_pr_contract(conn, task, **args)
    assert first["changed"] and first["event"]["hosted_ci_accepted"] is False
    assert first["event"]["original_contract"] == PR
    assert not rec.reconcile_pr_contract(conn, task, **args)["changed"]
    assert kb.get_task(conn, task).body == before
    assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_contract_reconciled'", (task,)).fetchone()[0] == 1
    assert kb.get_task(conn, child).status == "todo"
    assert kb.complete_task(conn, task, summary="Reviewed exact PR #182, source merge only")
    assert kb.get_task(conn, task).status == "done"
    assert kb.get_task(conn, child).status == "ready"
    assert calls.count("repos/acme/repo/pulls/182") >= 2  # mutable PR re-read after evidence
    assert any("rules/branches/main" in call for call in calls)


@pytest.mark.platforms("posix")
def test_review_scope_keeps_original_goal_and_judge_path(board, monkeypatch):
    conn, task, child, actor = board
    # A goal-mode foundation card with a direct producer child.
    conn.execute("UPDATE tasks SET goal_mode=1, body=? WHERE id=?", ("Original full-scope task: live producer and ledger", task))
    original = kb.get_task(conn, task).body
    revised = ("Review only the bounded transactional ledger and authenticated principal foundation "
               "on the current exact PR. Live producer ingestion is deferred to child " + child +
               "; the final program acceptance still requires a trusted live producer proof. "
               "Do not claim app delivery, disable the goal judge, or alter the existing PR contract.")
    result = rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
        expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_body=revised, deferred_child=child, reason="Owner scoped foundation review, not release")
    assert result["changed"] and kb.get_task(conn, task).body == revised
    assert kb.get_task(conn, task).goal_mode
    assert kb.get_task(conn, task).completion_contract == PR
    event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'", (task,)).fetchone()
    assert json.loads(event[0])["original_body"] == original
    with pytest.raises(ValueError, match="hash mismatch"):
        rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
            expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
            revised_body=revised, deferred_child=child, reason="retry")


@pytest.mark.platforms("posix")
def test_cli_review_handoff_still_calls_goal_judge_on_revised_scope(board, monkeypatch, capsys):
    from hermes_cli import kanban as cli

    conn, task, child, actor = board
    conn.execute("UPDATE tasks SET goal_mode=1 WHERE id=?", (task,))
    body = "Review bounded ledger foundation; the producer acceptance remains on linked child " + child
    body += ". Do not claim a full release. The PR and live producer must be checked in later integration."
    original = kb.get_task(conn, task).body
    rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
        expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        revised_body=body, deferred_child=child, reason="Owner-declared split")

    seen = []
    def judge(t, evidence):
        seen.append((t.body, evidence))
        return "continue", "the evidence does not meet the bounded review scope"
    monkeypatch.setattr(cli, "_goal_mode_handoff_rejection", judge)
    from hermes_cli.kanban_parser import build_parser
    import argparse
    wrapper = argparse.ArgumentParser()
    parser = build_parser(wrapper.add_subparsers(dest="top"))
    args = parser.parse_args(["request-review", task, "--summary", "missing evidence", "--reviewer", "orion"])
    assert cli.kanban_command(args) == 1
    assert seen == [(body, "missing evidence")]
    assert kb.get_task(conn, task).status == "ready"
    assert "rejected by judge" in capsys.readouterr().err


@pytest.mark.platforms("posix")
def test_cli_exact_target_and_downstream_readback(board, monkeypatch, capsys):
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_parser import build_parser
    import argparse

    conn, task, child, actor = board
    fake_github(monkeypatch)
    wrapper = argparse.ArgumentParser()
    parser = build_parser(wrapper.add_subparsers(dest="top"))
    args = parser.parse_args(["reconcile-pr-contract", task, "--operator", actor,
        "--confirm", task, "--pr-url", PR, "--head-sha", HEAD, "--merge-sha", MERGE,
        "--review-url", REVIEW, "--reason", "Verified private repo no-hosted-evidence owner approval",
        "--complete-summary", "Independently reviewed exact head and merged source; downstream release remains open"])
    assert cli.kanban_command(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"task_id": task, "changed": True, "completion_contract": "local-only",
                      "status": "done", "dependents": {child: "ready"}, "hosted_ci_accepted": False}
    assert kb.get_task(conn, task).status == "done"
    assert kb.get_task(conn, child).status == "ready"
    assert cli.kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
