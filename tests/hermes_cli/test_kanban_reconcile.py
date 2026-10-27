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
REVIEW = PR + "#pullrequestreview-1234"


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


def scope_fixture(conn, task, child):
    deferred = "Verify authenticated live producer ingestion end to end."
    original = ("Implement and independently review the transactional ledger with authenticated principal isolation.\n"
                "Prove persistence, deduplication and restart recovery with real integration tests on the exact PR.\n"
                + deferred + "\n")
    conn.execute("UPDATE tasks SET goal_mode=1, body=? WHERE id=?", (original, task))
    conn.execute("UPDATE tasks SET body=?, assignee='chip' WHERE id=?", (deferred, child))
    return original, {"deferred_acceptance": [deferred]}


def fake_github(monkeypatch, *, merged=True, head=HEAD, checks=0, status=None,
                review_head=HEAD, private=True, rules_unavailable=True,
                review_state="APPROVED", reviewer_id=2, reviewer_login="independent",
                plan_message=rec._PLAN_MESSAGE, admin=True, review_type="User", later_state=None,
                submitted_at="2026-10-27T19:00:00Z"):

    calls = []
    def api(endpoint, **kwargs):
        calls.append(endpoint)
        if endpoint == "repos/acme/repo/pulls/182":
            return {"html_url": PR, "head": {"sha": head}, "merge_commit_sha": MERGE,
                    "merged": merged, "merged_at": "2026-10-27T20:00:00Z" if merged else None,
                    "state": "closed" if merged else "open",
                    "user": {"id": 1, "login": "implementer", "type": "User"},
                    "base": {"ref": "main", "repo": {"full_name": "acme/repo"}}}
        if endpoint == "repos/acme/repo/pulls/182/reviews/1234":
            return {"id": 1234, "html_url": REVIEW,
                    "pull_request_url": "https://api.github.com/repos/acme/repo/pulls/182",
                    "user": {"id": reviewer_id, "login": reviewer_login, "type": review_type},
                    "state": review_state, "commit_id": review_head,
                    "submitted_at": submitted_at, "body": "Exact head approved"}
        if endpoint == "repos/acme/repo/pulls/182/reviews?per_page=100&page=1":
            verdict = api("repos/acme/repo/pulls/182/reviews/1234")
            later = [{**verdict, "id": 1235, "state": later_state}] if later_state else []
            return [verdict, *later]
        if endpoint == "repos/acme/repo":
            return {"full_name": "acme/repo", "private": private, "permissions": {"admin": admin}}
        if endpoint.endswith("check-runs?per_page=100"):
            return {"total_count": checks}
        if endpoint.endswith("statuses?per_page=100"):
            return status or []
        raise AssertionError(endpoint)
    def probe(command, **kwargs):
        endpoint = command[2]
        calls.append(endpoint)
        assert endpoint in {"repos/acme/repo/rules/branches/main?per_page=100",
                            "repos/acme/repo/branches/main/protection"}
        return SimpleNamespace(returncode=1 if rules_unavailable else 0,
            stdout=json.dumps({"status": "403", "message": plan_message}) if rules_unavailable else "[]")
    monkeypatch.setattr(rec, "_api", api)
    monkeypatch.setattr(rec.subprocess, "run", probe)
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
@pytest.mark.parametrize("fault", ["wrong-pr", "wrong-head", "wrong-merge", "not-merged", "missing-review", "hosted-check", "rules-readable", "self-review", "same-login", "negative-review", "irrelevant-review", "issue-comment", "permission-403", "not-admin", "bot-review", "missing-identity", "later-negative", "dismissed-review"])
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
    if fault == "self-review": settings["reviewer_id"] = 1
    if fault == "same-login": settings["reviewer_login"] = "IMPLEMENTER"
    if fault == "negative-review": settings["review_state"] = "CHANGES_REQUESTED"
    if fault == "irrelevant-review": settings["review_state"] = "COMMENTED"
    if fault == "issue-comment": opts["review_url"] = PR + "#issuecomment-1234"
    if fault == "permission-403": settings["plan_message"] = "Resource not accessible by personal access token"
    if fault == "not-admin": settings["admin"] = False
    if fault == "bot-review": settings["review_type"] = "Bot"
    if fault == "missing-identity": settings["reviewer_id"] = None
    if fault == "later-negative": settings["later_state"] = "CHANGES_REQUESTED"
    if fault == "dismissed-review": settings["review_state"] = "DISMISSED"
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
    assert first["event"]["review"]["user"]["id"] == 2
    assert first["event"]["review"]["state"] == "APPROVED"
    assert first["event"]["review"]["commit_id"] == HEAD
    assert first["event"]["review_sha256"] == hashlib.sha256(json.dumps(first["event"]["review"], sort_keys=True).encode()).hexdigest()
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
    original, amendment = scope_fixture(conn, task, child)
    result = rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
        expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        amendment=amendment, deferred_child=child, reason="Owner scoped foundation review, not release")
    revised = kb.get_task(conn, task).body
    assert result["changed"]
    assert original.splitlines()[0] in revised
    assert original.splitlines()[1] in revised
    assert child in revised
    assert hashlib.sha256(original.encode()).hexdigest() in revised
    assert kb.get_task(conn, task).goal_mode
    assert kb.get_task(conn, task).completion_contract == PR
    event = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'", (task,)).fetchone()
    assert json.loads(event[0])["original_body"] == original
    with pytest.raises(ValueError, match="hash mismatch"):
        rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
            expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
            amendment=amendment, deferred_child=child, reason="retry")


@pytest.mark.platforms("posix")
def test_cli_review_handoff_still_calls_goal_judge_on_revised_scope(board, monkeypatch, capsys):
    from hermes_cli import kanban as cli

    conn, task, child, actor = board
    original, amendment = scope_fixture(conn, task, child)
    rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
        expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        amendment=amendment, deferred_child=child, reason="Owner-declared split")
    body = kb.get_task(conn, task).body

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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["freeform", "unknown-field", "all-goal", "majority-goal",
    "partial-line", "foreign-line", "unassigned-child", "unowned-criterion", "not-linked",
    "claim-lock", "worker-pid", "repeated-amendment"])
def test_scope_amendment_rejects_unbounded_or_unowned_deferral(board, fault):
    conn, task, child, actor = board
    original, amendment = scope_fixture(conn, task, child)
    if fault == "freeform": amendment = "Review unrelated prose " * 10 + child
    if fault == "unknown-field": amendment["revised_body"] = "Remove all goals"
    if fault == "all-goal": amendment["deferred_acceptance"] = original.splitlines()
    if fault == "majority-goal": amendment["deferred_acceptance"] = original.splitlines()[:2]
    if fault == "partial-line": amendment["deferred_acceptance"] = [original.splitlines()[0][2:]]
    if fault == "foreign-line": amendment["deferred_acceptance"] = ["Something unrelated to original acceptance"]
    if fault == "unassigned-child": conn.execute("UPDATE tasks SET assignee=NULL WHERE id=?", (child,))
    if fault == "unowned-criterion": conn.execute("UPDATE tasks SET body='Other unrelated work' WHERE id=?", (child,))
    if fault == "not-linked": conn.execute("DELETE FROM task_links WHERE parent_id=? AND child_id=?", (task, child))
    if fault == "claim-lock": conn.execute("UPDATE tasks SET claim_lock='owner:123' WHERE id=?", (task,))
    if fault == "worker-pid": conn.execute("UPDATE tasks SET worker_pid=123 WHERE id=?", (task,))
    kwargs = dict(operator=actor, confirm=task, expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  amendment=amendment, deferred_child=child, reason="Bounded owner review scope")
    if fault == "repeated-amendment":
        rec.reconcile_review_scope(conn, task, **kwargs)
        original = kb.get_task(conn, task).body
        kwargs["expected_body_sha256"] = hashlib.sha256(original.encode()).hexdigest()
    with pytest.raises(ValueError):
        rec.reconcile_review_scope(conn, task, **kwargs)
    assert kb.get_task(conn, task).body == original


@pytest.mark.platforms("posix")
def test_scope_cli_structured_contract_readback(board, tmp_path, capsys):
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_parser import build_parser
    import argparse

    conn, task, child, actor = board
    original, amendment = scope_fixture(conn, task, child)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    wrapper = argparse.ArgumentParser()
    parser = build_parser(wrapper.add_subparsers(dest="top"))
    args = parser.parse_args(["reconcile-review-scope", task, "--operator", actor,
        "--confirm", task, "--expected-body-sha256", hashlib.sha256(original.encode()).hexdigest(),
        "--amendment-file", str(path), "--deferred-child", child, "--reason", "Owner bounded deferral"])
    assert cli.kanban_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    target = kb.get_task(conn, task)
    assert result["body_sha256"] == hashlib.sha256(target.body.encode()).hexdigest()
    assert result["status"] == target.status == "ready"
    assert target.goal_mode and target.completion_contract == PR
    assert kb.get_task(conn, child).status == "todo"


@pytest.mark.platforms("posix")
def test_plan_probe_fails_closed_on_unknown_malformed_or_classic_permission_response(monkeypatch):
    for stdout, code in [("not-json", 1), ("[]", 0),
            (json.dumps({"status": "403", "message": "Resource not accessible by integration"}), 1),
            (json.dumps({"status": "401", "message": rec._PLAN_MESSAGE}), 1)]:
        monkeypatch.setattr(rec.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=stdout, returncode=code))
        with pytest.raises(ValueError):
            rec._plan_limitation("repos/acme/repo/branches/main/protection")


@pytest.mark.platforms("posix")
def test_claim_race_during_evidence_cannot_write_contract(board, monkeypatch):
    conn, task, _, actor = board
    fake_github(monkeypatch)
    collect = rec.collect_merged_pr_evidence
    def claim_during_collection(*args):
        evidence = collect(*args)
        conn.execute("UPDATE tasks SET claim_lock='new-owner:123' WHERE id=?", (task,))
        return evidence
    monkeypatch.setattr(rec, "collect_merged_pr_evidence", claim_during_collection)
    with pytest.raises(ValueError, match="changed during evidence"):
        rec.reconcile_pr_contract(conn, task, **options(task, actor))
    assert kb.get_task(conn, task).completion_contract == PR


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mutation", ["claim", "child-body", "child-assignee", "unlink"])
def test_scope_transaction_rechecks_claim_and_exact_child(board, monkeypatch, mutation):
    from contextlib import contextmanager

    conn, task, child, actor = board
    original, amendment = scope_fixture(conn, task, child)
    transaction = rec.write_txn
    @contextmanager
    def concurrent_edit(connection):
        # A second real connection commits between validation and the write lock.
        db_path = connection.execute("PRAGMA database_list").fetchone()[2]
        import sqlite3
        with sqlite3.connect(db_path) as other:
            if mutation == "claim": other.execute("UPDATE tasks SET claim_lock='other:123' WHERE id=?", (task,))
            if mutation == "child-body": other.execute("UPDATE tasks SET body='unrelated work' WHERE id=?", (child,))
            if mutation == "child-assignee": other.execute("UPDATE tasks SET assignee='someone-else' WHERE id=?", (child,))
            if mutation == "unlink": other.execute("DELETE FROM task_links WHERE parent_id=? AND child_id=?", (task, child))
        with transaction(connection):
            yield
    monkeypatch.setattr(rec, "write_txn", concurrent_edit)
    with pytest.raises(ValueError, match="changed during amendment"):
        rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
            expected_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
            amendment=amendment, deferred_child=child, reason="Bounded owner deferral")
    assert kb.get_task(conn, task).body == original
    assert not conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'", (task,)).fetchone()


@pytest.mark.platforms("posix")
def test_reconciled_policy_does_not_override_parent_completion_gate(board, monkeypatch):
    conn, task, child, actor = board
    parent = kb.create_task(conn, title="Unfinished acceptance prerequisite")
    kb.link_tasks(conn, parent, task)
    conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task,))
    fake_github(monkeypatch)
    rec.reconcile_pr_contract(conn, task, **options(task, actor))
    assert not kb.complete_task(conn, task, summary="Exact PR independently approved and merged")
    assert kb.get_task(conn, task).status == "blocked"
    assert kb.get_task(conn, child).status == "todo"


def test_later_page_negative_review_is_not_hidden_by_first_page_approval(monkeypatch):
    verdict = {"id": 1234, "user": {"id": 2}, "state": "APPROVED"}
    calls = []
    def api(endpoint):
        calls.append(endpoint)
        if endpoint.endswith("page=1"):
            return [verdict, *[{"id": n, "state": "COMMENTED"} for n in range(99)]]
        assert endpoint.endswith("page=2")
        return [{"id": 1235, "user": {"id": 3}, "state": "CHANGES_REQUESTED"}]
    monkeypatch.setattr(rec, "_api", api)
    with pytest.raises(ValueError, match="unresolved changes"):
        rec._affirmative_review_history("acme/repo", 182, verdict)
    assert len(calls) == 2


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("timestamp", ["2026-10-27T20:00:01Z", "invalid", None])
def test_postmerge_or_unverifiable_github_approval_is_not_premerge_review(board, monkeypatch, timestamp):
    conn, task, _, actor = board
    fake_github(monkeypatch, submitted_at=timestamp)
    with pytest.raises(ValueError):
        rec.reconcile_pr_contract(conn, task, **options(task, actor))
    assert kb.get_task(conn, task).completion_contract == PR
