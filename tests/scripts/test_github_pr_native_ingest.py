"""Actual webhook adapter against the native CLI and an isolated Kanban board."""
from __future__ import annotations

import importlib.util
import io
import json

from pathlib import Path
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import profiles


SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "github_pr_native_ingest.py"


def _load_adapter():
    spec = importlib.util.spec_from_file_location("github_pr_native_ingest", SOURCE)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _use_real_cli(adapter, monkeypatch):
    def cli(argv, timeout=20):
        env = adapter.command_env()
        env["PYTHONPATH"] = str(SOURCE.parents[1])
        return subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv[1:]],
            text=True, capture_output=True, timeout=timeout, env=env,
        )

    monkeypatch.setattr(adapter, "run_argv", cli)
    return cli


def _deliver(adapter, monkeypatch, payload):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    return adapter.main()


def _payload(*, repository="example/widgets", number=90, head_sha=None, merged=False):
    head_sha = head_sha or "f" * 40
    return {
        "repository": {"full_name": repository},
        "action": "closed",
        "pull_request": {
            "number": number, "head": {"sha": head_sha}, "title": "external change",
            "html_url": f"https://github.com/{repository}/pull/{number}", "draft": False,
            "merged": merged,
        },
    }


def _event(adapter, *, repository="example/widgets", number=90, head_sha=None):
    normalized = adapter.normalize(_payload(
        repository=repository, number=number, head_sha=head_sha,
    ))
    assert normalized is not None
    return normalized[0]


def _create_cli_card(cli, *, title, body, created_by):
    created = cli([
        "hermes", "kanban", "create", title, "--body", body,
        "--created-by", created_by, "--json",
    ])
    assert created.returncode == 0, created.stderr
    return json.loads(created.stdout)["id"]


@pytest.mark.parametrize("merged", [False, True], ids=["closed", "merged"])
def test_terminal_delivery_retires_only_exact_adapter_pr_identity(tmp_path, monkeypatch, merged):
    adapter = _load_adapter()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cli = _use_real_cli(adapter, monkeypatch)

    positive_ids = []
    for index, status in enumerate(adapter.RETIRE_STATUSES, start=1):
        event = _event(adapter, head_sha=f"{index:040x}")
        positive_ids.append(_create_cli_card(
            cli, title=adapter.review_title(event), body=adapter.review_body(event, "opened"),
            created_by=adapter.CREATED_BY,
        ))

    exact = _event(adapter, head_sha="a" * 40)
    other_repo = _event(adapter, repository="example/widgets-extra", head_sha="b" * 40)
    other_number = _event(adapter, number=900, head_sha="c" * 40)
    copied_body = adapter.review_body(exact, "opened")
    negative_specs = [
        ("Review PR #90: human lookalike", "mentions example/widgets", "human"),
        (adapter.review_title(exact), copied_body, "human"),
        (adapter.review_title(exact), copied_body, "internal-worker"),
        (adapter.review_title(other_repo), adapter.review_body(other_repo, "opened"), adapter.CREATED_BY),
        (adapter.review_title(other_number), adapter.review_body(other_number, "opened"), adapter.CREATED_BY),
        (adapter.review_title(exact), copied_body + "PR: https://github.com/example/widgets/pull/90\n", adapter.CREATED_BY),
    ]
    negative_ids = [
        _create_cli_card(cli, title=title, body=body, created_by=creator)
        for title, body, creator in negative_specs
    ]

    with kbc.connect_closing(home / "kanban.db") as conn:
        for task_id, status in zip(positive_ids, adapter.RETIRE_STATUSES):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()
        negative_before = {
            task_id: (kb.get_task(conn, task_id), kb.list_events(conn, task_id))
            for task_id in negative_ids
        }

    assert _deliver(adapter, monkeypatch, _payload(head_sha="d" * 40, merged=merged)) == 0
    with kbc.connect_closing(home / "kanban.db") as conn:
        assert {kb.get_task(conn, task_id).status for task_id in positive_ids} == {"archived"}
        for task_id in negative_ids:
            task = kb.get_task(conn, task_id)
            before_task, before_events = negative_before[task_id]
            assert task is not None and before_task is not None
            assert (task.status, task.created_by, task.body) == (
                before_task.status, before_task.created_by, before_task.body,
            )
            assert kb.list_events(conn, task_id) == before_events


@pytest.mark.parametrize("readback", ["unavailable", "malformed", "wrong-id"])
def test_terminal_delivery_requires_unchanged_native_show_readback(tmp_path, monkeypatch, readback):
    adapter = _load_adapter()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    cli = _use_real_cli(adapter, monkeypatch)
    event = _event(adapter, head_sha="a" * 40)
    task_id = _create_cli_card(
        cli, title=adapter.review_title(event), body=adapter.review_body(event, "opened"),
        created_by=adapter.CREATED_BY,
    )
    with kbc.connect_closing(home / "kanban.db") as conn:
        before_task = kb.get_task(conn, task_id)
        before = kb.list_events(conn, task_id)
        assert before_task is not None

    real_run_kanban_json = adapter.run_kanban_json

    def fail_show(args, timeout=20):
        result = real_run_kanban_json(args, timeout=timeout)
        if args[:2] != ["show", task_id]:
            return result
        if readback == "unavailable":
            return None
        if readback == "malformed":
            return {"task": {"id": task_id}}
        assert isinstance(result, dict) and isinstance(result.get("task"), dict)
        result["task"]["id"] = "t_wrong"
        return result

    monkeypatch.setattr(adapter, "run_kanban_json", fail_show)
    assert _deliver(adapter, monkeypatch, _payload(head_sha="b" * 40)) == 1
    with kbc.connect_closing(home / "kanban.db") as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == before_task.status
        assert kb.list_events(conn, task_id) == before


def test_replay_preserves_explicit_owner_and_resubmission(tmp_path, monkeypatch):
    adapter = _load_adapter()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(adapter, "REVIEWER", "reviewer")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"reviewer", "dev"})

    # Only the executable location is substituted. Every adapter invocation
    # crosses the real parser, CLI and database under a private HERMES_HOME.
    _use_real_cli(adapter, monkeypatch)
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
