from __future__ import annotations

import json

import pytest

from tools.coding_kanban_gate import coding_tool_gate_refusal, intake_coding_task, is_coding_intent
from tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _coding_profiles_exist(monkeypatch):
    """Keep these unit tests independent of the host profile filesystem."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def test_intake_creates_a_canonical_card_and_persists_request_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    result = intake_coding_task(
        repository=str(tmp_path),
        workspace=str(tmp_path),
        scope="Implement the parser feature",
        acceptance_criteria=["Parser accepts escaped delimiters"],
        provider="openai",
        model="gpt-5-codex",
        provider_metadata={"account": "team"},
        origin_session_id="chat-1",
        origin_message_id="message-1",
    )

    assert result["status"] == "ready"
    assert result["coding_agent"] == "codex"
    assert result["task_id"].startswith("t_")

    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        task = kanban_db.get_task(conn, result["task_id"])
    assert task.metadata["provider_metadata"] == {"account": "team"}
    assert task.metadata["origin"] == {
        "session_id": "chat-1",
        "message_id": "message-1",
    }


def test_same_complete_request_reuses_one_card(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    request = dict(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Fix the parser",
        acceptance_criteria=["Escaped delimiters work"], provider="openai",
        model="gpt-5-codex", provider_metadata={"account": "team"},
        origin_session_id="chat-2", origin_message_id="message-2",
    )

    first = intake_coding_task(**request)
    second = intake_coding_task(**request)

    assert second["task_id"] == first["task_id"]
    assert second["reused"] is True

    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        assert len(kanban_db.list_tasks(conn, session_id="chat-2")) == 1


def test_intake_persists_session_scoped_linkage(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("chat-3", source="cli")
    db.close()

    result = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Fix it",
        origin_session_id="chat-3", origin_message_id="message-3",
    )

    db = SessionDB()
    session = db.get_session("chat-3")
    db.close()
    assert result["session_persisted"] is True
    assert session["kanban_task_id"] == result["task_id"]
    assert session["kanban_origin_session_id"] == "chat-3"
    assert session["kanban_origin_message_id"] == "message-3"


def test_shared_tool_dispatch_hands_off_and_claude_is_subscription_required(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls = []
    registry = ToolRegistry()
    registry.register(
        name="patch", toolset="test", schema={"name": "patch"},
        handler=lambda args, **kwargs: calls.append(args) or json.dumps({"ok": True}),
    )

    handed_off = json.loads(registry.dispatch(
        "patch", {"scope": "Implement this", "origin_message_id": "m-4"},
        session_id="chat-4", user_message="Implement this",
    ))
    claude = json.loads(registry.dispatch(
        "patch", {"coding_agent": "claude"}, session_id="chat-5",
        user_message="Implement this",
    ))

    assert handed_off["task_id"].startswith("t_")
    assert handed_off["status"] == "ready"
    assert claude["status"] == "subscription_required"
    assert claude["error_type"] == "subscription_required"
    assert calls == []


def test_coding_lane_accepts_qualified_specialist_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for index, specialist in enumerate(("Forge", "Quill", "Chip"), start=1):
        result = intake_coding_task(
            repository=str(tmp_path), workspace=str(tmp_path),
            scope=f"Implement specialist task {specialist}",
            origin_session_id=f"specialist-{index}", origin_message_id=f"m-{index}",
            assignee=specialist,
        )
        assert result["ok"] is True
        assert result["assignee"] == specialist.casefold()


def test_cursor_requires_explicit_request_and_claude_never_creates_a_card(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    cursor = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Implement it",
        coding_agent="cursor", origin_session_id="cursor", origin_message_id="m",
    )
    claude = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Implement it",
        coding_agent="claude", origin_session_id="claude", origin_message_id="m",
    )

    assert cursor["ok"] is True
    assert cursor["coding_agent"] == "cursor"
    assert claude["status"] == "subscription_required"
    assert "task_id" not in claude

    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        assert kanban_db.list_tasks(conn, session_id="claude") == []


def test_unknown_assignee_and_mutating_interpreter_commands_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    result = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Implement it",
        origin_session_id="unknown", origin_message_id="m", assignee="ghost",
    )
    assert result["status"] == "invalid_assignee"
    assert is_coding_intent("terminal", {"command": "python3 -c 'open(\"x\", \"w\")'"})
    assert is_coding_intent("terminal", {"command": "gh api /repos/example -X POST"})
    assert is_coding_intent("terminal", {"command": "gh pr review 1 --approve"})


def test_gh_read_only_classifier_fails_closed_for_mutations(monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"dev", "orion"})
    read_only = [
        "gh pr view 1", "gh --repo solovision24/hermes-agent pr view 1", "gh issue list",
        "gh run view 1", "gh api /repos/example",
    ]
    mutating = [
        "gh issue reopen 1", "gh run cancel 1", "gh api /repos/example -f state=closed",
        "gh api /repos/example -F state=closed", "gh api /repos/example --input payload.json",
        "gh variable set NAME --body value", "gh secret set NAME", "gh release upload v1 file.zip",
        "gh api /repos/example -XPOST", "gh api /repos/example -fstate=closed",
    ]
    assert all(not is_coding_intent("terminal", {"command": command}) for command in read_only)
    assert all(is_coding_intent("terminal", {"command": command}) for command in mutating)


def test_shell_and_python_mutation_probes_fail_closed():
    assert is_coding_intent("terminal", {"command": "sed -n 'w /tmp/copied.py' source.py"})
    assert is_coding_intent("terminal", {"command": "git diff --output=/tmp/patch.diff"})
    assert is_coding_intent("terminal", {"command": "cat pyproject.toml\nmake format"})
    assert is_coding_intent("execute_code", {"code": "from pathlib import Path; Path('x.py').touch()"})
    assert is_coding_intent("execute_code", {"code": "DataFrame().to_csv('x.py')"})


def test_active_generic_worker_task_with_legacy_null_metadata_is_allowed(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import kanban_db

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_legacy")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setattr(kanban_db, "connect_closing", lambda: _Connection())
    monkeypatch.setattr(
        kanban_db,
        "get_task",
        lambda _conn, _task_id: SimpleNamespace(
            metadata=None, current_run_id=7, assignee="dev", status="running",
        ),
    )

    assert coding_tool_gate_refusal(
        "patch", function_args={}, user_message="Implement this change",
    ) is None


def test_implicit_routing_never_uses_active_non_engineering_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "orion")
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"dev", "orion"})
    result = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Implement it",
        origin_session_id="routing", origin_message_id="m",
    )
    assert result["ok"] is True
    assert result["assignee"] == "dev"


def test_session_linkage_failure_is_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(
        "tools.coding_kanban_gate._persist_session_linkage",
        lambda *args, **kwargs: (False, "simulated persistence failure"),
    )
    result = intake_coding_task(
        repository=str(tmp_path), workspace=str(tmp_path), scope="Implement it",
        origin_session_id="broken", origin_message_id="m",
    )
    assert result["ok"] is False
    assert result["status"] == "session_linkage_failed"
