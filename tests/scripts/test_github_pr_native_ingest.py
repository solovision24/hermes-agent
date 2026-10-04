"""Actual webhook adapter against the native CLI and an isolated Kanban board."""
from __future__ import annotations

import importlib.util
import io
import json

from pathlib import Path
import subprocess
import sys

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import profiles


SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "github_pr_native_ingest.py"


def test_replay_preserves_explicit_owner_and_resubmission(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("github_pr_native_ingest", SOURCE)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(adapter, "REVIEWER", "reviewer")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"reviewer", "dev"})

    # Only the executable location is substituted. Every adapter invocation
    # crosses the real parser, CLI and database under a private HERMES_HOME.
    def cli(argv, timeout=20):
        env = adapter.command_env()
        env["PYTHONPATH"] = str(SOURCE.parents[1])
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv[1:]],
            text=True, capture_output=True, timeout=timeout, env=env,
        )

    monkeypatch.setattr(adapter, "run_argv", cli)
    event = {
        "repository": "example/widgets", "number": 90, "head_sha": "a" * 40,
        "title": "external change", "url": "https://github.com/example/widgets/pull/90",
        "draft": False, "checks_passed": None, "mergeable": True,
    }
    payload = {
        "repository": {"full_name": event["repository"]}, "action": "opened",
        "pull_request": {"number": 90, "head": {"sha": event["head_sha"]},
                         "title": event["title"], "html_url": event["url"], "draft": False},
    }

    def deliver():
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
        assert adapter.main() == 0

    deliver()
    with kbc.connect_closing(home / "kanban.db") as conn:
        tasks = kb.list_tasks(conn)
        assert len(tasks) == 1
        task_id = tasks[0].id
        first = kb.get_task(conn, task_id)
        assert first is not None
        assert first.status == "review" and first.assignee == "reviewer"
        requested = [e for e in kb.list_events(conn, task_id) if e.kind == "review_requested"][-1]
        assert requested.payload is not None and requested.payload["implementer"] is None
        review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
        assert review is not None
        assert kb.request_changes(conn, task_id, reason="Repair", expected_run_id=review.current_run_id,
                                  remediation_assignee="dev") == (True, "dev")
        before = kb.list_events(conn, task_id)

    deliver()
    with kbc.connect_closing(home / "kanban.db") as conn:
        assert kb.list_events(conn, task_id) == before
        assigned = kb.get_task(conn, task_id)
        assert assigned is not None and (assigned.status, assigned.assignee) == ("ready", "dev")
        dispatched = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)
        assert any(row[0] == task_id for row in dispatched.spawned)
        work = kb.get_task(conn, task_id)
        assert work is not None
        assert work.assignee == "dev" and work.status == "running"

    deliver()  # replay while remediation is in flight
    with kbc.connect_closing(home / "kanban.db") as conn:
        assigned = kb.get_task(conn, task_id)
        assert assigned is not None and assigned.assignee == "dev"
        work = kb.get_task(conn, task_id)
        assert work is not None
        assert kb.request_review(conn, task_id, summary="resubmitted", expected_run_id=work.current_run_id)
        reviews = [e for e in kb.list_events(conn, task_id) if e.kind == "review_requested"]
        assert all(e.payload is not None for e in reviews)
        assert [e.payload["implementer"] for e in reviews] == [None, "dev"]
        rereview = kb.claim_review_task(conn, task_id, claimer="reviewer:again")
        assert rereview is not None
        assert kb.request_changes(conn, task_id, reason="again", expected_run_id=rereview.current_run_id) == (True, "dev")
        assert len(kb.list_tasks(conn)) == 1

    # A different immutable head or repository is new intake, not a replay.
    payload["pull_request"]["head"]["sha"] = "b" * 40
    deliver()
    payload["repository"]["full_name"] = "example/other"
    deliver()
    with kbc.connect_closing(home / "kanban.db") as conn:
        tasks = kb.list_tasks(conn)
        assert len({task.id for task in tasks}) == 3
        original = kb.get_task(conn, task_id)
        assert original is not None and original.assignee == "dev"
