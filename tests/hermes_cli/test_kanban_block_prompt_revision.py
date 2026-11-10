"""Public API contract for revising a current human block prompt in place."""

from __future__ import annotations

from pathlib import Path
import threading

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kb.connect_closing() as conn:
        yield conn


def _blocked_human_task(conn, reason: str = "NEED: Choose a launch region.\nOPTIONS: EU | US"):
    task_id = kb.create_task(conn, title="human decision", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (task_id,))
    assert kb.claim_task(conn, task_id, claimer="worker") is not None
    assert kb.block_task(conn, task_id, reason=reason, kind="needs_input")
    prompt = kb.get_current_block_prompt(conn, task_id)
    assert prompt is not None
    return task_id, prompt


def test_revise_block_prompt_appends_versioned_event_without_mutating_task_or_runs(board) -> None:
    task_id, before_prompt = _blocked_human_task(board)
    task_before = dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    runs_before = [dict(row) for row in board.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
    )]

    revised = kb.revise_block_prompt(
        board,
        task_id,
        reason="Type the custom hostname you want.",
        kind="needs_input",
        expected_event_id=before_prompt.event_id,
        actor="mission-control:owner",
    )

    assert revised.previous_event_id == before_prompt.event_id
    assert revised.previous_reason == before_prompt.reason
    assert revised.reason == "Type the custom hostname you want."
    assert revised.kind == "needs_input"
    assert revised.event_id > before_prompt.event_id
    assert dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()) == task_before
    assert [dict(row) for row in board.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,)
    )] == runs_before
    event = board.execute("SELECT kind, payload, run_id FROM task_events WHERE id=?", (revised.event_id,)).fetchone()
    assert event["kind"] == "block_prompt_revised"
    assert event["run_id"] is None
    assert kb.get_current_block_prompt(board, task_id) == revised


def test_structured_prompt_can_be_respecified_then_revised_to_free_text(board) -> None:
    task_id, original = _blocked_human_task(board)
    respecified = kb.revise_block_prompt(
        board, task_id,
        reason="NEED: Which region should launch first?\nOPTIONS: Europe | North America",
        kind="needs_input", expected_event_id=original.event_id, actor="owner",
    )
    free_text = kb.revise_block_prompt(
        board, task_id,
        reason="Type the exact region and any constraints.",
        kind="needs_input", expected_event_id=respecified.event_id, actor="owner",
    )
    assert free_text.previous_reason == respecified.reason
    assert "OPTIONS:" not in free_text.reason
    assert kb.get_current_block_prompt(board, task_id) == free_text


def test_stale_compare_and_swap_is_refused_and_only_one_concurrent_writer_wins(
    board,
) -> None:
    task_id, original = _blocked_human_task(board)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def attempt(reason: str) -> None:
        with kb.connect_closing() as conn:
            barrier.wait()
            try:
                kb.revise_block_prompt(
                    conn, task_id, reason=reason, kind="needs_input",
                    expected_event_id=original.event_id, actor="owner",
                )
                outcomes.append("ok")
            except kb.BlockPromptRevisionError as exc:
                outcomes.append(exc.code)

    threads = [
        threading.Thread(target=attempt, args=("Choose A or B?",)),
        threading.Thread(target=attempt, args=("Choose C or D?",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["ok", "stale_event"]
    assert board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='block_prompt_revised'",
        (task_id,),
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"reason": ""}, "invalid_reason"),
        ({"reason": None}, "invalid_reason"),
        ({"reason": "bad\x00reason"}, "invalid_reason"),
        ({"reason": "OPTIONS: only one"}, "invalid_reason"),
        ({"reason": "NEED:\nOPTIONS: A | B"}, "invalid_reason"),
        ({"reason": "NEED: Choose?\nOPTIONS: A"}, "invalid_reason"),
        ({"reason": "NEED: Choose?\nNEED: Other?"}, "invalid_reason"),
        ({"reason": "bad\x1breason"}, "invalid_reason"),
        ({"kind": "capability"}, "invalid_kind"),
        ({"kind": "transient"}, "invalid_kind"),
        ({"expected_event_id": "1"}, "invalid_expected_event_id"),
        ({"expected_event_id": True}, "invalid_expected_event_id"),
        ({"actor": ""}, "invalid_actor"),
    ],
)
def test_malformed_revision_arguments_are_refused_unchanged(board, overrides, code) -> None:
    task_id, prompt = _blocked_human_task(board)
    before = board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)
    ).fetchone()[0]
    args = {
        "reason": "What should happen next?",
        "kind": "needs_input",
        "expected_event_id": prompt.event_id,
        "actor": "owner",
        **overrides,
    }
    with pytest.raises(kb.BlockPromptRevisionError) as caught:
        kb.revise_block_prompt(board, task_id, **args)
    assert caught.value.code == code
    assert board.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)
    ).fetchone()[0] == before


@pytest.mark.parametrize("status,kind", [("ready", "needs_input"), ("blocked", "capability"), ("blocked", "dependency"), ("blocked", "transient")])
def test_wrong_status_or_machine_actionable_lane_is_refused(board, status, kind) -> None:
    task_id, prompt = _blocked_human_task(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status=?, block_kind=? WHERE id=?", (status, kind, task_id))
    with pytest.raises(kb.BlockPromptRevisionError) as caught:
        kb.revise_block_prompt(
            board, task_id, reason="Question?", kind="needs_input",
            expected_event_id=prompt.event_id, actor="owner",
        )
    assert caught.value.code == "invalid_block_state"


def test_live_task_and_run_claim_state_is_refused_without_changes(board) -> None:
    task_id, prompt = _blocked_human_task(board)
    with kb.write_txn(board):
        run_id = board.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'worker', 'running', 'claim', 9999999999, 42, 1)",
            (task_id,),
        ).lastrowid
        board.execute(
            "UPDATE tasks SET current_run_id=?, claim_lock='claim', claim_expires=9999999999, "
            "worker_pid=42 WHERE id=?", (run_id, task_id),
        )
    task_before = dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    run_before = dict(board.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone())
    with pytest.raises(kb.BlockPromptRevisionError) as caught:
        kb.revise_block_prompt(
            board, task_id, reason="Question?", kind="needs_input",
            expected_event_id=prompt.event_id, actor="owner",
        )
    assert caught.value.code == "active_claim"
    assert dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()) == task_before
    assert dict(board.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone()) == run_before


def test_explicit_revision_event_is_authoritative_even_when_generic(board) -> None:
    task_id, prompt = _blocked_human_task(board)
    revised = kb.revise_block_prompt(
        board, task_id, reason="Question?", kind="needs_input",
        expected_event_id=prompt.event_id, actor="owner",
    )
    assert kb.get_current_block_prompt(board, task_id) == revised


def test_real_triage_respecify_freeform_history_can_be_revised_in_place(board) -> None:
    task_id, _ = _blocked_human_task(board)
    assert kb.unblock_task(board, task_id)
    assert kb.block_task(board, task_id, reason="Human decision required: Which region?", kind="needs_input")
    assert kb.get_task(board, task_id).status == "triage"
    assert kb.specify_triage_task(board, task_id, body="Choose region, no deployment authorization.", author="owner")
    assert kb.block_task(board, task_id, reason="Human decision required: Which region?", kind=None)
    before = dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    prompt = kb.get_current_block_prompt(board, task_id)
    assert prompt is not None and prompt.kind is None
    revised = kb.revise_block_prompt(board, task_id, reason="Type the region.", kind="needs_input", expected_event_id=prompt.event_id, actor="owner")
    assert revised.previous_event_id == prompt.event_id
    assert dict(board.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()) == before
    assert kb.get_task(board, task_id).block_kind is None


def test_legacy_untyped_machine_failure_is_not_a_human_prompt(board) -> None:
    task_id, _ = _blocked_human_task(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET block_kind=NULL WHERE id=?", (task_id,))
        kb._append_event(board, task_id, "blocked", {"reason":"CI build failed: retry automatically", "kind":None})
    assert kb.get_current_block_prompt(board, task_id) is None
    with pytest.raises(kb.BlockPromptRevisionError, match="no well-formed"):
        kb.revise_block_prompt(board, task_id, reason="Choose?", kind="needs_input", expected_event_id=1, actor="owner")


def test_malformed_selected_revision_never_falls_back_to_old_prompt(board) -> None:
    task_id, _ = _blocked_human_task(board)
    with kb.write_txn(board):
        kb._append_event(board, task_id, "block_prompt_revised", {"reason":"Choose?", "kind":"needs_input", "schema_version":2})
    assert kb.get_current_block_prompt(board, task_id) is None
