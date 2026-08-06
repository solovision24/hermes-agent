"""Behavioral coverage for centralized Kanban assignee resolution."""

from __future__ import annotations

import time

import pytest


@pytest.fixture
def assignee_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "profiles" / "worker").mkdir(parents=True)
    (home / "profiles" / "worker" / "config.yaml").write_text("model: test\n")
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  external_lanes: [terminal]\n"
        "  assignee_aliases:\n"
        "    build: worker\n"
        "    handoff: terminal\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_resolver_classifies_and_preserves_canonical_targets(assignee_home):
    from hermes_cli.kanban_assignees import AssigneeCategory, resolve_assignee

    assert resolve_assignee(None).category is AssigneeCategory.UNASSIGNED
    profile = resolve_assignee(" WORKER ")
    assert (profile.category, profile.canonical) == (
        AssigneeCategory.PROFILE,
        "worker",
    )
    alias = resolve_assignee("build")
    assert (alias.category, alias.target_category, alias.canonical) == (
        AssigneeCategory.ALIAS,
        AssigneeCategory.PROFILE,
        "worker",
    )
    lane = resolve_assignee("TERMINAL")
    assert (lane.category, lane.canonical) == (
        AssigneeCategory.EXTERNAL_LANE,
        "terminal",
    )
    handoff = resolve_assignee("handoff")
    assert (handoff.category, handoff.target_category, handoff.canonical) == (
        AssigneeCategory.ALIAS,
        AssigneeCategory.EXTERNAL_LANE,
        "terminal",
    )

    with pytest.raises(ValueError, match=r"^invalid_assignee:"):
        resolve_assignee("halo")
    explicit_alias = resolve_assignee(
        "halo",
        config={"kanban": {"assignee_aliases": {"halo": "default"}}},
    )
    assert (explicit_alias.category, explicit_alias.canonical) == (
        AssigneeCategory.ALIAS,
        "default",
    )


def test_path_traversal_is_not_a_profile(assignee_home):
    from hermes_cli.kanban_assignees import InvalidAssigneeError, resolve_assignee

    for value in ("..", ".", "../worker", "profiles/worker"):
        with pytest.raises(InvalidAssigneeError):
            resolve_assignee(value, allow_unassigned=False)


def test_configured_choices_are_typed(assignee_home):
    from hermes_cli.kanban_assignees import (
        AssigneeCategory,
        configured_assignee_choices,
    )

    choices = configured_assignee_choices()
    by_input = {str(choice.input_value): choice for choice in choices}
    assert by_input["worker"].target_category is AssigneeCategory.PROFILE
    assert by_input["terminal"].target_category is AssigneeCategory.EXTERNAL_LANE
    assert by_input["build"].category is AssigneeCategory.ALIAS


def test_known_assignees_preserves_alias_input_and_target_metadata(assignee_home):
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        entries = {entry["name"]: entry for entry in kb.known_assignees(conn)}

    assert entries["build"]["category"] == "alias"
    assert entries["build"]["canonical"] == "worker"
    assert entries["build"]["target_category"] == "profile"
    assert entries["build"]["spawnable"] is True
    assert entries["handoff"]["category"] == "alias"
    assert entries["handoff"]["target_category"] == "external_lane"
    assert entries["handoff"]["spawnable"] is False


def test_create_assign_and_reassign_reject_before_mutation(assignee_home):
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.create_task(conn, title="bad", assignee="typo")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        task_id = kb.create_task(conn, title="valid", assignee="build")
        assert kb.get_task(conn, task_id).assignee == "worker"
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.assign_task(conn, task_id, "typo")
        assert kb.get_task(conn, task_id).assignee == "worker"

        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='lock', "
            "claim_expires=?, worker_pid=123 WHERE id=?",
            (int(time.time()) + 3600, task_id),
        )
        conn.commit()
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.reassign_task(conn, task_id, "typo", reclaim_first=True)
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert (row["status"], row["claim_lock"], row["assignee"]) == (
            "running",
            "lock",
            "worker",
        )


def test_dispatch_distinguishes_registered_lane_and_legacy_invalid_row(assignee_home):
    from hermes_cli import kanban_db as kb

    with kb.connect_closing() as conn:
        external_id = kb.create_task(conn, title="handoff", assignee="terminal")
        invalid_id = kb.create_task(conn, title="old", assignee=None)
        conn.execute(
            "UPDATE tasks SET assignee='stale-typo' WHERE id=?", (invalid_id,)
        )
        conn.commit()
        spawned = []
        result = kb.dispatch_once(
            conn,
            dry_run=True,
            spawn_fn=lambda *args, **kwargs: spawned.append(args),
        )

    assert result.skipped_nonspawnable == [external_id]
    assert result.skipped_invalid_assignee == [invalid_id]
    assert not result.spawned
    assert not spawned


def test_invalid_legacy_row_gets_actionable_diagnostic(assignee_home):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_diagnostics import compute_task_diagnostics

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="old", assignee=None)
        conn.execute(
            "UPDATE tasks SET assignee='stale-typo' WHERE id=?", (task_id,)
        )
        conn.commit()
        task = kb.get_task(conn, task_id)
        diagnostics = compute_task_diagnostics(task, [], [], config={"kanban": {}})

    invalid = [d for d in diagnostics if d.kind == "invalid_assignee"]
    assert len(invalid) == 1
    assert invalid[0].severity == "error"
    assert invalid[0].detail.startswith("invalid_assignee:")
    assert invalid[0].actions[0].kind == "reassign"


def test_create_assign_reassign_reject_on_empty_default_config(tmp_path, monkeypatch):
    """Unresolved assignees must fail on an EMPTY/default config too.

    The permissive write-path fallback historically accepted any
    syntactically valid label whenever no external lanes or aliases were
    configured (the normal default).  The contract now rejects unresolved
    strings BEFORE mutation on every board, configured or not.
    """
    from hermes_cli import kanban_db as kb

    home = tmp_path / "hermes"
    home.mkdir()
    # No profiles dir, no config.yaml -> no lanes, no aliases: default board.
    monkeypatch.setenv("HERMES_HOME", str(home))

    with kb.connect_closing() as conn:
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.create_task(conn, title="bad", assignee="definitely-not-a-profile")
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0

        task_id = kb.create_task(conn, title="ok", assignee=None)
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.assign_task(conn, task_id, "definitely-not-a-profile")
        assert kb.get_task(conn, task_id).assignee is None

        conn.execute(
            "UPDATE tasks SET status='running', claim_lock='lock', "
            "claim_expires=?, worker_pid=123 WHERE id=?",
            (int(time.time()) + 3600, task_id),
        )
        conn.commit()
        with pytest.raises(ValueError, match=r"^invalid_assignee:"):
            kb.reassign_task(conn, task_id, "definitely-not-a-profile", reclaim_first=True)
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert (row["status"], row["claim_lock"], row["assignee"]) == (
            "running", "lock", None,
        )


def test_legacy_invalid_row_stays_listable_by_label(tmp_path, monkeypatch):
    """A pre-existing invalid row remains READABLE on an empty config.

    ``list_tasks(assignee=...)`` is a read/filter path: filtering by an
    unresolved label must return matching legacy rows, not raise.
    """
    from hermes_cli import kanban_db as kb

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee=None)
        conn.execute("UPDATE tasks SET assignee='stale-typo' WHERE id=?", (task_id,))
        conn.commit()
        rows = kb.list_tasks(conn, assignee="stale-typo")
        assert [r.id for r in rows] == [task_id]
        # Unresolved labels for listing never raise even when nothing matches.
        assert kb.list_tasks(conn, assignee="another-stale-label") == []
