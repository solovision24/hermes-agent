"""Pre-dispatch routing guarantees for mutable workflow configuration work."""

from __future__ import annotations

import json

import pytest


def test_workflow_configuration_routes_default_to_forge_or_dev(monkeypatch):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name in {"forge", "dev"})
    assignee, metadata = kanban_intake.preflight_workflow_configuration(
        task_type="workflow_configuration", assignee="default"
    )
    assert assignee == "forge"
    assert metadata["canonical"] is True
    assert metadata["lane"] == "FORGE"
    assert metadata["coding_agent"] == "codex"
    assert metadata["use_coding_router"] is True


@pytest.mark.parametrize("invalid_agent", ["direct", "bogus", "DEV"])
def test_workflow_configuration_rejects_invalid_coding_agent(monkeypatch, invalid_agent):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    with pytest.raises(ValueError, match="coding_agent="):
        kanban_intake.preflight_workflow_configuration(
            task_type="workflow_configuration", assignee="default",
            metadata={"coding_agent": invalid_agent},
        )


def test_read_only_preflight_has_no_write_or_duplicate_counts(monkeypatch):
    from hermes_cli import kanban_intake
    monkeypatch.setattr(kanban_intake, "_profile_exists", lambda name: name == "dev")
    result = kanban_intake.preflight_output(
        task_type="workflow_configuration", assignee="halo",
        metadata={"route": "dev_direct", "use_coding_router": False},
    )
    assert result["accepted"] is True
    assert result["resolved_assignee"] == "dev"
    assert result["writes"] == 0
    assert result["duplicates_created"] == 0
    assert result["metadata"]["coding_agent"] is None
    assert json.dumps(result)


def test_create_task_normalizes_before_insert_and_rejects_invalid_metadata(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "forge").mkdir(parents=True)
    (home / "profiles" / "dev").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import kanban_db as kb

    kb.init_db()
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="workflow config fixture",
            assignee="default",
            metadata={"task_type": "workflow_configuration"},
        )
        task = kb.get_task(conn, task_id)
        assert task.assignee == "forge"
        assert task.metadata["canonical"] is True
        assert task.metadata["coding_agent"] == "codex"
        with pytest.raises(ValueError, match="coding_agent=.*direct"):
            kb.create_task(
                conn,
                title="invalid workflow config",
                assignee="default",
                metadata={"task_type": "workflow_configuration", "coding_agent": "direct"},
            )
        assert len(kb.list_tasks(conn)) == 1