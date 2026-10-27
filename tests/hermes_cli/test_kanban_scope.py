"""Actual MC1/bridge snapshots exercised on a disposable board, never the live DB."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_reconcile as rec
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_scope import clause_overlay, digest


def span(body, text):
    start = body.index(text)
    return {"start": start, "end": start + len(text), "text": text, "sha256": digest(text)}


@pytest.fixture
def board(tmp_path, monkeypatch):
    import pwd
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(rec.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    fixture = json.loads((Path(__file__).parent / "fixtures/kanban_scope_mc1.json").read_text())
    original, child_body = fixture["parent_body"], fixture["child_body"]
    assert digest(original) == "01e20f9349893b172e321dc845fdbb0a73d74f3b1c4b56ad4fc36ef688d60b8f"
    assert digest(child_body) == "226eca2fa6ac960261554a3ffd8b723a7817471e3bb731a967329fffa0f9ee44"
    assert len(original) == 1783 and len(original.splitlines()) == 1
    kb.init_db()
    with connect() as conn:
        task = kb.create_task(conn, title="MC-1 durable transactional inbox and authenticated source adapters",
                              body=original, assignee="forge", goal_mode=True)
        child = kb.create_task(conn, title="Real producer delivery", body=child_body, assignee="forge", parents=[task])
        criterion = child_body[child_body.index("Done:"):].split(". ")[0] + "."
        amendment = {"version": 2, "child_body_sha256": digest(child_body), "clause_deferrals": [{
            "mode": "reference_subset", "source": span(original, "acceptance §6"),
            "child_criteria": [span(child_body, criterion)],
        }]}
        yield conn, task, child, original, child_body, amendment, pwd.getpwuid(os.geteuid()).pw_name


def apply(board, amendment=None):
    conn, task, child, original, _, default, actor = board
    return rec.reconcile_review_scope(conn, task, operator=actor, confirm=task,
        expected_body_sha256=digest(original), amendment=default if amendment is None else amendment,
        deferred_child=child, reason="Owner scopes foundation review; specific producer acceptance remains child-owned")


@pytest.mark.platforms("posix")
def test_actual_mc1_reference_subset_preserves_full_original_and_audit(board):
    conn, task, child, original, child_body, amendment, _ = board
    before_events = list(conn.execute("SELECT id, kind, payload FROM task_events WHERE task_id=?", (task,)))
    apply(board)
    target = kb.get_task(conn, task)
    assert target.body.startswith(original + "\n\n")
    assert target.goal_mode and target.status == "ready" and target.assignee == "forge"
    assert target.completion_contract == "local-only"
    assert "NOT the whole referenced section" in target.body
    assert "source implementation, fixtures, tests, principal isolation" in target.body
    assert kb.get_task(conn, child).body == child_body and kb.get_task(conn, child).status == "todo"
    event = json.loads(conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'", (task,)).fetchone()[0])
    assert event["original_body"] == original and event["original_body_sha256"] == digest(original)
    assert event["revised_body"] == target.body and event["child_body_sha256"] == digest(child_body)
    assert event["clause_deferrals"] == amendment["clause_deferrals"]
    assert event["deferred_assignee"] == "forge"
    assert list(conn.execute("SELECT id, kind, payload FROM task_events WHERE task_id=? ORDER BY id LIMIT ?", (task, len(before_events)))) == before_events
    with pytest.raises(ValueError, match="already amended"):
        rec.reconcile_review_scope(conn, task, operator=board[-1], confirm=task,
            expected_body_sha256=digest(target.body), amendment=amendment, deferred_child=child, reason="second removal")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("fault", ["replacement", "version", "float-version", "source-hash", "source-text", "source-offset",
    "byte-offset", "bool-offset", "out-of-bounds", "child-hash", "criterion-hash", "criterion-text", "criterion-offset",
    "negation-strip", "foreign-clause", "duplicate", "overlap", "unknown-mode", "blanket", "empty-criteria",
    "unknown-span-key", "criterion-substring", "all-child", "missing-child", "unassigned", "finished", "unlinked", "worker"])
def test_adversarial_substitutions_cannot_write_scope(board, monkeypatch, fault):
    conn, task, child, original, child_body, amendment, _ = board
    amendment = copy.deepcopy(amendment)
    entry = amendment["clause_deferrals"][0]
    source, criterion = entry["source"], entry["child_criteria"][0]
    if fault == "replacement": amendment["revised_body"] = "Skip principal security and call this done"
    if fault == "version": amendment["version"] = 3
    if fault == "float-version": amendment["version"] = 2.0
    if fault == "source-hash": source["sha256"] = "0" * 64
    if fault == "source-text": source["text"] = "skip acceptance"
    if fault == "source-offset": source["start"] -= 1
    if fault == "byte-offset": source["start"] = len(original[:source["start"]].encode())
    if fault == "bool-offset": source["start"] = True
    if fault == "out-of-bounds": source["end"] = len(original) + 1
    if fault == "child-hash": amendment["child_body_sha256"] = "0" * 64
    if fault == "criterion-hash": criterion["sha256"] = "0" * 64
    if fault == "criterion-text": criterion["text"] = "Done: All security requirements are waived."
    if fault == "criterion-offset": criterion["start"] += 1
    if fault == "negation-strip":
        entry.update(mode="clause", source=span(original, "AuthSession.sub as owner, no trusting client ID"))
    if fault == "foreign-clause": entry["source"] = span(child_body, "App record not suppressed by Telegram failure.")
    if fault == "duplicate": amendment["clause_deferrals"].append(copy.deepcopy(entry))
    if fault == "overlap": amendment["clause_deferrals"].append({**entry, "source": {**source, "start": source["start"] + 1}})
    if fault == "unknown-mode": entry["mode"] = "replace_body"
    if fault == "blanket": entry.update(mode="clause", source=span(original, original))
    if fault == "empty-criteria": entry["child_criteria"] = []
    if fault == "unknown-span-key": source["replacement"] = "Everything passed"
    if fault == "criterion-substring": entry["child_criteria"] = [span(child_body, "no invented approval authority")]
    if fault == "all-child": entry["child_criteria"] = [span(child_body, child_body)]
    if fault == "missing-child": conn.execute("DELETE FROM task_links WHERE child_id=?", (child,)); conn.execute("DELETE FROM tasks WHERE id=?", (child,))
    if fault == "unassigned": conn.execute("UPDATE tasks SET assignee=NULL WHERE id=?", (child,))
    if fault == "finished": conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
    if fault == "unlinked": conn.execute("DELETE FROM task_links WHERE parent_id=? AND child_id=?", (task, child))
    if fault == "worker": monkeypatch.setenv("HERMES_KANBAN_TASK", task)
    with pytest.raises((ValueError, PermissionError)):
        apply(board, amendment)
    assert kb.get_task(conn, task).body == original
    assert not conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'", (task,)).fetchone()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("mutation", ["claim", "run", "pid", "parent-body", "child-body", "assignee", "status", "unlink", "event", "child-event"])
def test_real_second_connection_races_are_rechecked(board, monkeypatch, mutation):
    conn, task, child, original, _, _, _ = board
    transaction = rec.write_txn
    @contextmanager
    def racing_transaction(connection):
        import sqlite3
        path = connection.execute("PRAGMA database_list").fetchone()[2]
        with sqlite3.connect(path) as other:
            if mutation == "claim": other.execute("UPDATE tasks SET claim_lock='other:123' WHERE id=?", (task,))
            if mutation == "run": other.execute("UPDATE tasks SET current_run_id=123 WHERE id=?", (task,))
            if mutation == "pid": other.execute("UPDATE tasks SET worker_pid=123 WHERE id=?", (task,))
            if mutation == "parent-body": other.execute("UPDATE tasks SET body='changed parent' WHERE id=?", (task,))
            if mutation == "child-body": other.execute("UPDATE tasks SET body='changed child' WHERE id=?", (child,))
            if mutation == "assignee": other.execute("UPDATE tasks SET assignee='chip' WHERE id=?", (child,))
            if mutation == "status": other.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
            if mutation == "unlink": other.execute("DELETE FROM task_links WHERE parent_id=? AND child_id=?", (task, child))
            if mutation in {"event", "child-event"}:
                other.execute("INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'review_scope_reconciled', '{}', 1)", (task if mutation == "event" else child,))
        with transaction(connection):
            yield
    monkeypatch.setattr(rec, "write_txn", racing_transaction)
    with pytest.raises(ValueError, match="changed during amendment|already amended"):
        apply(board)
    assert kb.get_task(conn, task).body == ("changed parent" if mutation == "parent-body" else original)


def test_clause_mode_handles_one_line_without_child_verbatim_and_preserves_unicode():
    original = ("Preserve authorization and principal isolation; verify all fixtures and persistence. "
                "Exercise real producer delivery after review. Retain all tests and source implementation.")
    child = "Done: demonstrate a real producer update and authenticated durable app visibility."
    amendment = {"version": 2, "child_body_sha256": digest(child), "clause_deferrals": [{
        "mode": "clause", "source": span(original, "Exercise real producer delivery after review."),
        "child_criteria": [span(child, child)],
    }]}
    revised, _ = clause_overlay(original, child, amendment, "t_child")
    assert revised.startswith(original + "\n\n")
    assert "Exercise real producer delivery after review." not in child


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("verdict", ["continue", "done"])
def test_actual_cli_scope_and_real_judge_prompt_path(board, tmp_path, monkeypatch, capsys, verdict):
    from hermes_cli import kanban as cli
    from hermes_cli.kanban_parser import build_parser
    from agent import auxiliary_client

    conn, task, child, original, _, amendment, actor = board
    parser = build_parser(argparse.ArgumentParser().add_subparsers(dest="top"))
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment))
    args = parser.parse_args(["reconcile-review-scope", task, "--operator", actor, "--confirm", task,
        "--expected-body-sha256", digest(original), "--amendment-file", str(path), "--deferred-child", child,
        "--reason", "Owner maps only producer acceptance to bridge"])
    assert cli.kanban_command(args) == 0
    result = json.loads(capsys.readouterr().out)
    revised = kb.get_task(conn, task).body
    assert result["body_sha256"] == digest(revised)
    seen = []
    monkeypatch.setattr(auxiliary_client, "get_text_auxiliary_client", lambda *a: (object(), "fixture-model"))
    def call_llm(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            {"verdict": verdict, "reason": "Fixture judge requires retained security evidence"})))])
    monkeypatch.setattr(auxiliary_client, "call_llm", call_llm)
    # No judge or handoff function is replaced: real prompt construction, response
    # parsing, goal gate and review transition execute against the disposable DB.
    args = parser.parse_args(["request-review", task, "--summary", "Exact-head foundation evidence; not live delivery", "--reviewer", "orion"])
    assert cli.kanban_command(args) == (1 if verdict == "continue" else 0)
    assert len(seen) == 1 and seen[0]["task"] == "goal_judge"
    prompt = seen[0]["messages"][1]["content"]
    assert revised in prompt  # regression: previous 2000-char limit hid the overlay
    assert original in prompt
    assert amendment["clause_deferrals"][0]["child_criteria"][0]["text"] in prompt
    assert kb.get_task(conn, task).status == ("ready" if verdict == "continue" else "review")
    assert kb.get_task(conn, child).status == "todo"  # review never releases program acceptance
