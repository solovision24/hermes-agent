"""Replay captured PR182 native review shape only into an isolated test board.

The JSON is actual supported-accessor/GitHub read output, not invented hosted CI.
SQL below hydrates/mutates disposable fixtures only, never an operator board.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_reconcile as rec
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_parser import build_parser

TARGET = "t_54fe371c"
CHILD = "t_f7243b24"
REVIEW_TASK = "t_5e785fd1"
REVIEW_RUN = 15992


@pytest.fixture
def native_board(tmp_path, monkeypatch):
    import pwd
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(rec.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    data = json.loads((Path(__file__).parent / "fixtures/kanban_native_pr182.json").read_text())
    kb.init_db()
    with connect() as conn:
        # Preserve exact captured IDs/run/event bindings, not mocked accessors.
        for table, rows in (("tasks", data["tasks"]), ("task_runs", data["runs"]), ("task_events", data["events"])):
            for row in rows:
                values = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
                columns = ",".join(values)
                conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({','.join('?' for _ in values)})", tuple(values.values()))
        kb.link_tasks(conn, TARGET, CHILD)
        yield conn, data, pwd.getpwuid(os.geteuid()).pw_name


def args_for(data, actor):
    pr = data["pr"]
    return dict(operator=actor, confirm=TARGET, pr_url=pr["html_url"], head_sha=pr["head"]["sha"],
                merge_sha=pr["merge_commit_sha"], review_url=data["comment"]["html_url"],
                native_review_task=REVIEW_TASK, native_review_run=REVIEW_RUN,
                reason="Explicit owner policy for independently reviewed exact merged source only")


def github_capture(monkeypatch, data):
    pr = data["pr"]
    repo = pr["base"]["repo"]["full_name"]
    def api(endpoint):
        if endpoint == f"repos/{repo}/pulls/182": return data["pr"]
        if endpoint == f"repos/{repo}/issues/comments/{data['comment']['id']}": return data["comment"]
        if endpoint == f"repos/{repo}": return {"full_name": repo, "private": True, "permissions": {"admin": True}}
        if endpoint.endswith("check-runs?per_page=100"): return {"total_count": 0}
        if endpoint.endswith("statuses?per_page=100"): return []
        if endpoint.endswith("reviews?per_page=100&page=1"): return data.get("reviews", [])
        raise AssertionError(endpoint)
    monkeypatch.setattr(rec, "_api", api)
    monkeypatch.setattr(rec.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout=json.dumps({"status": "403", "message": rec._PLAN_MESSAGE})))


@pytest.mark.platforms("posix")
def test_actual_native_shape_exact_target_cli_lifecycle_and_idempotence(native_board, monkeypatch, capsys):
    conn, data, actor = native_board
    github_capture(monkeypatch, data)
    opts = args_for(data, actor)
    parser = build_parser(argparse.ArgumentParser().add_subparsers(dest="top"))
    argv = ["reconcile-pr-contract", TARGET]
    for key, value in opts.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    argv += ["--complete-summary", "Reviewed exact source-only PR182 and immutable merge; release remains with dependent integration"]
    args = parser.parse_args(argv)
    original_body = kb.get_task(conn, TARGET).body
    assert kb.get_task(conn, CHILD).status == "todo"
    assert cli.kanban_command(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {"task_id": TARGET, "changed": True, "completion_contract": "local-only",
                      "status": "done", "dependents": {CHILD: "ready"}, "hosted_ci_accepted": False}
    assert kb.get_task(conn, TARGET).body == original_body
    assert kb.get_task(conn, CHILD).status == "ready"
    audit = [e for e in kb.list_events(conn, TARGET) if e.kind == "pr_contract_reconciled"]
    assert len(audit) == 1
    proof = audit[0].payload
    assert proof["review"]["review_run_id"] == REVIEW_RUN
    assert proof["review"]["reviewer"] not in proof["review"]["implementers"]
    assert proof["review"]["metadata"]["reviewed_head"] == opts["head_sha"]
    assert proof["review"]["corroboration"] == data["comment"]
    assert proof["original_contract"] == opts["pr_url"]
    assert proof["review_sha256"] == hashlib.sha256(json.dumps(proof["review"], sort_keys=True).encode()).hexdigest()
    assert cli.kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out)["changed"] is False
    assert len([e for e in kb.list_events(conn, TARGET) if e.kind == "pr_contract_reconciled"]) == 1
    with pytest.raises(ValueError, match="different evidence"):
        rec.reconcile_pr_contract(conn, TARGET, **{**opts, "native_review_run": REVIEW_RUN + 1})


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["no-native", "missing-run", "wrong-task", "self-review", "not-review-lane",
    "no-spawn", "not-approved", "wrong-head", "wrong-merge", "wrong-pr", "no-session", "wrong-intake",
    "self-asserted-comment", "negative-comment", "negated-comment", "late-comment", "late-edit", "foreign-canonical",
    "negative-github", "later-native-run", "revoked", "unpaired-task", "unpaired-run"])
def test_native_authority_and_exact_premerge_provenance_fail_closed(native_board, monkeypatch, fault):
    conn, data, actor = native_board
    opts = args_for(data, actor)
    run = next(r for r in data["runs"] if r["id"] == REVIEW_RUN)
    meta = run["metadata"].copy()
    if fault == "no-native": opts.update(native_review_task=None, native_review_run=None)
    if fault == "missing-run": opts["native_review_run"] = 999999
    if fault == "wrong-task": opts["native_review_task"] = CHILD
    if fault == "self-review": conn.execute("UPDATE task_runs SET profile='orion' WHERE task_id=?", (TARGET,))
    if fault == "not-review-lane": conn.execute("UPDATE task_events SET payload='{}' WHERE task_id=? AND kind='claimed'", (REVIEW_TASK,))
    if fault == "no-spawn": conn.execute("DELETE FROM task_events WHERE task_id=? AND kind='spawned'", (REVIEW_TASK,))
    if fault == "not-approved": meta["review_outcome"] = "changes_requested"
    if fault == "wrong-head": meta["reviewed_head"] = "a" * 40
    if fault == "wrong-merge": meta["merge_commit"] = "b" * 40
    if fault == "wrong-pr": meta["published_pr"] = opts["pr_url"].replace("182", "181")
    if fault == "no-session": meta.pop("worker_session_id")
    if fault == "wrong-intake":
        conn.execute("UPDATE task_runs SET metadata='{}' WHERE id=?", (15989,))
    if fault == "self-asserted-comment":
        conn.execute("UPDATE task_runs SET profile='quill' WHERE id=?", (REVIEW_RUN,))
    if fault == "negative-comment": data["comment"]["body"] = data["comment"]["body"].replace("approved for source merge", "changes requested")
    if fault == "negated-comment": data["comment"]["body"] = data["comment"]["body"].replace("approved for source merge", "not approved for source merge")
    if fault == "late-comment": data["comment"]["created_at"] = "2026-10-27T20:30:09Z"
    if fault == "late-edit": data["comment"]["updated_at"] = "2026-10-27T20:30:09Z"
    if fault == "foreign-canonical": data["pr"]["body"] = data["pr"]["body"].replace(TARGET, TARGET + "x")
    if fault == "negative-github": data["reviews"] = [{"id": 20, "state": "CHANGES_REQUESTED", "user": {"id": 4}}]
    if fault == "later-native-run":
        conn.execute("INSERT INTO task_runs(task_id,profile,status,started_at) VALUES(?, 'orion', 'running', ?)", (REVIEW_TASK, run["ended_at"] + 1))
    if fault == "revoked": kb._append_event(conn, REVIEW_TASK, "changes_requested", {"reviewer": "orion"})
    if fault == "unpaired-task": opts["native_review_run"] = None
    if fault == "unpaired-run": opts["native_review_task"] = None
    conn.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(meta), REVIEW_RUN))
    github_capture(monkeypatch, data)
    with pytest.raises(ValueError):
        rec.reconcile_pr_contract(conn, TARGET, **opts)
    assert kb.get_task(conn, TARGET).completion_contract == data["pr"]["html_url"]
    assert kb.get_task(conn, CHILD).status == "todo"
    assert not [e for e in kb.list_events(conn, TARGET) if e.kind == "pr_contract_reconciled"]


@pytest.mark.platforms("posix")
def test_native_authority_is_connection_board_bound_a_b_a(native_board, tmp_path):
    from hermes_cli.kanban_native_review import native_review_snapshot
    conn, data, actor = native_board
    opts = args_for(data, actor)
    def snapshot(connection):
        return native_review_snapshot(connection, TARGET, REVIEW_TASK, REVIEW_RUN,
            opts["pr_url"], opts["head_sha"], opts["merge_sha"])
    first = snapshot(conn)
    with connect(tmp_path / "board-b.db") as other:
        with pytest.raises(ValueError, match="independent dispatched"):
            snapshot(other)
    assert snapshot(conn) == first


@pytest.mark.platforms("posix")
def test_native_review_snapshot_is_rechecked_inside_write_lock(native_board, monkeypatch):
    conn, data, actor = native_board
    github_capture(monkeypatch, data)
    collect = rec.collect_merged_pr_evidence
    def race(*args):
        result = collect(*args)
        conn.execute("UPDATE task_runs SET metadata='{}' WHERE id=?", (REVIEW_RUN,))
        return result
    monkeypatch.setattr(rec, "collect_merged_pr_evidence", race)
    with pytest.raises(ValueError, match="native approval"):
        rec.reconcile_pr_contract(conn, TARGET, **args_for(data, actor))
    assert kb.get_task(conn, TARGET).completion_contract == data["pr"]["html_url"]
    assert kb.get_task(conn, CHILD).status == "todo"
