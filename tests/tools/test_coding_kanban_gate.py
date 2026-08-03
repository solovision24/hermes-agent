"""Behavioral coverage for the chat-to-DEV coding handoff gate."""

from __future__ import annotations

import json

import pytest

from tools.registry import ToolRegistry
from tools.coding_kanban_gate import coding_tool_gate_refusal, is_coding_intent


def _registry(calls: list[str], *names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in names:
        registry.register(
            name=name,
            toolset="test",
            schema={"name": name},
            handler=lambda args, _name=name, **kwargs: calls.append(_name) or json.dumps({"ok": True}),
        )
    return registry


def _payload(result):
    return json.loads(result)


def test_feature_request_hands_off_before_patch_and_reports_real_task(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, "patch")

    result = _payload(registry.dispatch(
        "patch", {"scope": "Implement the parser feature"},
        session_id="chat-feature", task_id="message-1",
        user_message="Please implement the parser feature",
    ))

    assert result["error_type"] == "kanban_task_required"
    assert result["task_id"].startswith("t_")
    assert result["status"] == "ready"
    assert result["assignee"] == "dev"
    assert result["coding_agent"] == "codex"
    assert calls == []

    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        task = kanban_db.get_task(conn, result["task_id"])
    assert task is not None
    assert task.metadata["lane"] == "DEV"
    assert task.metadata["origin"] == {"session_id": "chat-feature", "message_id": "message-1"}
    assert task.metadata["repository"]
    assert task.metadata["workspace"]
    assert task.metadata["acceptance_criteria"]


def test_tiny_edit_reuses_card_without_unlocking_originating_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, "write_file")
    kwargs = dict(
        session_id="chat-edit", task_id="message-2",
        user_message="Make the tiny edit to the version string",
    )

    first = _payload(registry.dispatch("write_file", {"path": "x", "content": "1"}, **kwargs))
    second = _payload(registry.dispatch("write_file", {"path": "x", "content": "1"}, **kwargs))

    assert second["task_id"] == first["task_id"]
    assert second["status"] == first["status"] == "ready"
    assert second["reused"] is True
    assert calls == []

    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        assert len(kanban_db.list_tasks(conn, session_id="chat-edit")) == 1


@pytest.mark.parametrize(
    ("name", "args", "message"),
    [
        ("terminal", {"command": "pwd && git status --short && rg 'needle' ."}, "Inspect the repo"),
        ("execute_code", {"code": "print(sum(range(10)))"}, "Calculate a value"),
        ("delegate_task", {"goal": "Research the relevant parser documentation"}, "Research only"),
    ],
)
def test_read_only_inspection_calculation_and_research_remain_available(
    monkeypatch, tmp_path, name, args, message,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, name)

    result = _payload(registry.dispatch(name, args, session_id="chat-read", user_message=message))

    assert result == {"ok": True}
    assert calls == [name]


def test_discussion_to_implementation_creates_card_at_transition(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, "read_file", "patch")

    assert _payload(registry.dispatch(
        "read_file", {}, session_id="chat-transition", user_message="Can you inspect this?"
    )) == {"ok": True}
    handoff = _payload(registry.dispatch(
        "patch", {}, session_id="chat-transition", task_id="turn-3",
        user_message="Now implement the fix",
    ))

    assert handoff["task_id"].startswith("t_")
    assert handoff["status"] == "ready"
    assert calls == ["read_file"]


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("terminal", {"command": "git branch feature-x"}),
        ("terminal", {"command": "git worktree add ../feature-x"}),
        ("delegate_task", {"goal": "Implement the parser fix"}),
    ],
)
def test_branch_worktree_and_implementation_delegation_handoff_before_execution(
    monkeypatch, tmp_path, name, args,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, name)

    result = _payload(registry.dispatch(
        name, args, session_id="chat-handoff", task_id="handoff-message",
        user_message="Implement this change",
    ))

    assert result["task_id"].startswith("t_")
    assert result["status"] == "ready"
    assert calls == []


def test_agent_level_delegate_dispatch_cannot_bypass_handoff(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from model_tools import handle_function_call

    result = _payload(handle_function_call(
        "delegate_task", {"goal": "Implement the requested fix"},
        task_id="agent-message", session_id="chat-agent",
        user_task="Please implement the requested fix",
    ))

    assert result["error_type"] == "kanban_task_required"
    assert result["task_id"].startswith("t_")


def test_cursor_is_recorded_only_when_explicit_and_claude_cannot_launch(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    calls: list[str] = []
    registry = _registry(calls, "terminal")

    cursor = _payload(registry.dispatch(
        "terminal", {"command": "cursor agent"}, session_id="chat-cursor",
        task_id="cursor-message", user_message="Use Cursor for this implementation",
    ))
    claude = _payload(registry.dispatch(
        "terminal", {"command": "claude"}, session_id="chat-claude",
        task_id="claude-message", user_message="Use Claude for this implementation",
    ))

    assert cursor["coding_agent"] == "cursor"
    assert claude["error_type"] == "unsupported_coding_agent"
    assert calls == []


def test_existing_canonical_matching_task_is_reused(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        task_id = kanban_db.create_task(
            conn,
            title="Implement parser",
            assignee="dev",
            session_id="chat-existing",
            workspace_kind="dir",
            workspace_path=str(tmp_path),
            metadata={
                "canonical": True, "lane": "DEV", "coding_agent": "codex",
                "origin": {"session_id": "chat-existing", "message_id": "existing-message"},
                "repository": str(tmp_path), "workspace": str(tmp_path),
            },
        )

    calls: list[str] = []
    registry = _registry(calls, "patch")
    result = _payload(registry.dispatch(
        "patch", {"scope": "Implement parser", "workspace": str(tmp_path), "repository": str(tmp_path)}, session_id="chat-existing",
        task_id="existing-message", user_message="Implement parser",
    ))

    assert result["task_id"] == task_id
    assert result["reused"] is True
    with kanban_db.connect_closing() as conn:
        assert len(kanban_db.list_tasks(conn, session_id="chat-existing")) == 1


def test_kanban_unavailable_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db

    monkeypatch.setattr(kanban_db, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    calls: list[str] = []
    registry = _registry(calls, "patch")

    result = _payload(registry.dispatch("patch", {}, session_id="chat-offline", user_message="Fix it"))

    assert result["error_type"] == "kanban_unavailable"
    assert calls == []


def test_arbitrary_session_task_does_not_unlock_originating_chat(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        kanban_db.create_task(conn, title="unrelated", assignee="worker", session_id="chat-unrelated")

    calls: list[str] = []
    registry = _registry(calls, "patch")
    result = _payload(registry.dispatch("patch", {}, session_id="chat-unrelated", user_message="Fix it"))

    assert result["error_type"] == "kanban_task_required"
    assert result["task_id"].startswith("t_")
    assert calls == []


def test_worker_can_change_code_only_from_own_canonical_dev_task(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        own_id = kanban_db.create_task(
            conn, title="worker task", assignee="dev", session_id="origin",
            metadata={
                "canonical": True, "lane": "DEV", "coding_agent": "codex",
                "origin": {"session_id": "origin", "message_id": "m"},
            },
        )
        other_id = kanban_db.create_task(
            conn, title="other task", assignee="worker", session_id="origin",
        )

    calls: list[str] = []
    registry = _registry(calls, "patch")
    monkeypatch.setenv("HERMES_KANBAN_TASK", own_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    with kanban_db.connect_closing() as conn:
        conn.execute("UPDATE tasks SET current_run_id = 42 WHERE id = ?", (own_id,))
    assert _payload(registry.dispatch("patch", {}, session_id="worker-session", user_message="Implement")) == {"ok": True}
    assert calls == ["patch"]

    monkeypatch.setenv("HERMES_KANBAN_TASK", other_id)
    denied = _payload(registry.dispatch("patch", {}, session_id="worker-session", user_message="Implement"))
    assert denied["error_type"] == "kanban_task_required"
    assert calls == ["patch"]


def test_fake_task_environment_without_active_run_cannot_unlock_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import kanban_db

    with kanban_db.connect_closing() as conn:
        task_id = kanban_db.create_task(
            conn, title="fake", assignee="dev", session_id="origin",
            metadata={
                "canonical": True, "lane": "DEV", "coding_agent": "codex",
                "origin": {"session_id": "origin", "message_id": "m"},
            },
        )
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999")
    calls: list[str] = []
    registry = _registry(calls, "patch")
    result = _payload(registry.dispatch("patch", {}, session_id="worker-session", user_message="Implement"))
    assert result["error_type"] == "kanban_task_required"
    assert calls == []


def test_codex_app_server_enters_the_same_intake_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    result = _payload(coding_tool_gate_refusal(
        "codex_app_server",
        session_id="chat-codex",
        task_id="message-1",
        user_message="Implement the requested feature",
    ))
    assert result["task_id"].startswith("t_")
    assert result["status"] == "ready"
    assert result["coding_agent"] == "codex"


@pytest.mark.parametrize("command", [
    "date", "ps -ef", "curl -I https://example.com", "sqlite3 state.db '.tables'",
    "gh pr view 19", "git remote -v", "git remote show origin",
])
def test_read_only_operational_probes_are_not_coding_intent(command):
    assert is_coding_intent("terminal", {"command": command}, "Inspect runtime state") is False


def test_mutating_shell_probes_are_coding_intent():
    assert is_coding_intent("terminal", {"command": "find . -delete"}, "Inspect the tree") is True
    assert is_coding_intent("terminal", {"command": "git remote add origin https://example.com"}, "Inspect remotes") is True


def test_report_write_and_read_only_codex_turn_do_not_create_intake_task(tmp_path):
    assert is_coding_intent(
        "write_file", {"path": str(tmp_path / "report.md"), "content": "Findings"},
        "Write the audit report",
    ) is False
    assert is_coding_intent("codex_app_server", {}, "Inspect the repository and report findings") is False


def test_active_non_dev_worker_can_still_inspect(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "not-a-real-task")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "run-1")
    calls: list[str] = []
    registry = _registry(calls, "terminal")
    result = _payload(registry.dispatch(
        "terminal", {"command": "date"}, session_id="worker", user_message="Inspect time",
    ))
    assert result == {"ok": True}
    assert calls == ["terminal"]
