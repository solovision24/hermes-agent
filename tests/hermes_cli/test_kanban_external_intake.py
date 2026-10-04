"""External GitHub intake review ownership regressions."""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import profiles


SHA = "6544b13c24d52a2269a5c16030071d95b018ee76"
KEY = f"github-pr:solovisionllc/solo-skills:90:{SHA}"


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def profile_roster(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"dev", "reviewer"})


def _intake_review(conn, *, key: str = KEY, created_by: str | None = "github-webhook",
                   drop_event: bool = False):
    task_id = kb.create_task(
        conn, title="Review PR #90: external intake", created_by=created_by,
        idempotency_key=key,
    )
    assert kb.request_review(conn, task_id, summary="GitHub delivery", reviewer="reviewer")
    if drop_event:
        with kb.write_txn(conn):
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? AND kind = 'review_requested'",
                (task_id,),
            )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert review is not None
    return task_id, review


@pytest.mark.parametrize("drop_event", [False, True])
def test_exact_intake_requires_explicit_owner_without_fabricating_event(
    conn, profile_roster, drop_event: bool,
):
    task_id, review = _intake_review(conn, drop_event=drop_event)
    before = [event for event in kb.list_events(conn, task_id) if event.kind == "review_requested"]
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=review.current_run_id,
    )[0] is False
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=review.current_run_id,
        remediation_assignee="dev",
    ) == (True, "dev")
    after = [event for event in kb.list_events(conn, task_id) if event.kind == "review_requested"]
    assert after == before
    task = kb.get_task(conn, task_id)
    assert task is not None and (task.status, task.assignee) == ("ready", "dev")
    verdict = [event for event in kb.list_events(conn, task_id) if event.kind == "changes_requested"][-1]
    assert verdict.payload["assignment"] == "explicit_reviewer_remediation_assignee"
    assert verdict.payload["head_sha"] == SHA


@pytest.mark.parametrize(
    "candidate", ["", "missing", "reviewer"],
)
def test_invalid_or_self_remediation_owner_is_atomic(conn, profile_roster, candidate: str):
    task_id, review = _intake_review(conn)
    before = kb.list_events(conn, task_id)
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=review.current_run_id,
        remediation_assignee=candidate,
    )[0] is False
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert (task.status, task.assignee, task.current_run_id) == (
        "running", "reviewer", review.current_run_id,
    )
    assert kb.list_events(conn, task_id) == before


@pytest.mark.parametrize(
    "created_by,key",
    [
        ("builder", KEY),
        ("github-webhook", "github-pr:solovisionllc/solo-skills:0:" + SHA),
        ("github-webhook", "github-pr:solovisionllc/solo-skills:90:" + SHA[:-1]),
        ("github-webhook", "github-pr:solo-skills:90:" + SHA),
        ("github-webhook", KEY.upper()),
    ],
)
def test_malformed_external_identity_fails_closed(
    conn, profile_roster, created_by: str, key: str,
):
    task_id, review = _intake_review(conn, created_by=created_by, key=key)
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=review.current_run_id,
        remediation_assignee="dev",
    )[0] is False
    assert kb.get_task(conn, task_id).status == "running"


def test_internal_provenance_cannot_be_overridden(conn, profile_roster):
    task_id = kb.create_task(conn, title="Internal", assignee="dev")
    work = kb.claim_task(conn, task_id, claimer="dev:test")
    assert work is not None
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", expected_run_id=work.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert review is not None
    assert kb.request_changes(
        conn, task_id, reason="redirect", expected_run_id=review.current_run_id,
        remediation_assignee="reviewer",
    )[0] is False
    assert kb.request_changes(
        conn, task_id, reason="proper", expected_run_id=review.current_run_id,
    ) == (True, "dev")


def test_parent_gate_and_owned_rereview_stay_on_same_card(conn, profile_roster):
    parent = kb.create_task(conn, title="Parent", assignee="dev")
    parent_run = kb.claim_task(conn, parent, claimer="dev:parent")
    assert parent_run is not None and kb.complete_task(conn, parent, summary="done")
    task_id = kb.create_task(
        conn, title="Review PR #91", parents=[parent], created_by="github-webhook",
        idempotency_key="github-pr:solovisionllc/solo-skills:91:" + "a" * 40,
    )
    assert kb.request_review(conn, task_id, reviewer="reviewer")
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:test")
    assert review is not None
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready', completed_at = NULL WHERE id = ?", (parent,))
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=review.current_run_id,
        remediation_assignee="dev",
    ) == (True, "dev")
    assert kb.get_task(conn, task_id).status == "todo"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done', completed_at = 1 WHERE id = ?", (parent,))
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    before = kb.list_events(conn, task_id)
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", with_reason=True,
    ) == (False, "external PR remediation belongs to its assigned worker")
    assert kb.list_events(conn, task_id) == before
    dispatched = kbd.dispatch_once(conn, spawn_fn=lambda *args, **kwargs: 4242)
    assert any(item[0] == task_id for item in dispatched.spawned)
    work = kb.get_task(conn, task_id)
    assert work is not None and work.assignee == "dev"
    assert kb.request_review(
        conn, task_id, reviewer="reviewer", expected_run_id=work.current_run_id,
    )
    rereview = kb.claim_review_task(conn, task_id, claimer="reviewer:again")
    assert rereview is not None
    assert kb.request_changes(
        conn, task_id, reason="again", expected_run_id=rereview.current_run_id,
    ) == (True, "dev")


@pytest.mark.parametrize("mode", ["stale", "non_review", "wrong_owner"])
def test_verdict_requires_current_reviewer_owned_review_run(conn, profile_roster, mode: str):
    task_id, review = _intake_review(conn)
    expected = review.current_run_id
    if mode == "stale":
        expected += 1
    elif mode == "non_review":
        with kb.write_txn(conn):
            event = conn.execute(
                "SELECT id, payload FROM task_events WHERE task_id = ? AND kind = 'claimed' "
                "ORDER BY id DESC LIMIT 1", (task_id,),
            ).fetchone()
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                ('{"source_status":"ready"}', event["id"]),
            )
    else:
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET profile = 'dev' WHERE id = ?", (expected,))
    assert kb.request_changes(
        conn, task_id, reason="repair", expected_run_id=expected,
        remediation_assignee="dev",
    )[0] is False
    assert kb.get_task(conn, task_id).status == "running"


def test_registered_native_tool_threads_explicit_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_roster,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as board:
        task_id, review = _intake_review(board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    import tools.kanban_tools  # noqa: F401 - register the native handlers
    from tools.registry import registry

    result = registry.dispatch(
        "kanban_request_changes",
        {"reason": "repair", "remediation_assignee": "dev"},
    )
    assert '"implementer": "dev"' in result
    with kbc.connect() as board:
        task = kb.get_task(board, task_id)
        assert task is not None and (task.status, task.assignee) == ("ready", "dev")


def test_native_cli_threads_explicit_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_roster,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as board:
        task_id, review = _intake_review(board)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(review.current_run_id))

    from hermes_cli import kanban as kc

    output = kc.run_slash(
        f"request-changes {task_id} repair --remediation-assignee dev"
    )
    assert "Requested changes" in output and "routed to dev" in output
    with kbc.connect() as board:
        task = kb.get_task(board, task_id)
        assert task is not None and (task.status, task.assignee) == ("ready", "dev")
