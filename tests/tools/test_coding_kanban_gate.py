"""Behavioral tests for the chat-originated coding Kanban gate."""

import json

from tools.registry import ToolRegistry


def test_code_changing_tool_is_refused_before_handler_without_active_task(
    monkeypatch, tmp_path,
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    called = False

    def handler(args, **kwargs):
        nonlocal called
        called = True
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(
        name="write_file",
        toolset="file",
        schema={"name": "write_file"},
        handler=handler,
    )

    result = json.loads(registry.dispatch("write_file", {}, session_id="chat-1"))

    assert result["error_type"] == "kanban_task_required"
    assert result["tool"] == "write_file"
    assert called is False


def test_chat_task_association_does_not_bypass_worker_gate(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from hermes_cli import kanban_db

    conn = kanban_db.connect()
    try:
        kanban_db.create_task(
            conn,
            title="implement change",
            assignee="worker",
            session_id="chat-2",
        )
    finally:
        conn.close()

    called = False

    def handler(args, **kwargs):
        nonlocal called
        called = True
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(
        name="patch",
        toolset="file",
        schema={"name": "patch"},
        handler=handler,
    )

    result = json.loads(registry.dispatch("patch", {}, session_id="chat-2"))

    assert result["error_type"] == "kanban_task_required"
    assert called is False


def test_code_changing_tool_is_allowed_for_kanban_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-worker-1")
    called = False

    def handler(args, **kwargs):
        nonlocal called
        called = True
        return json.dumps({"ok": True})

    registry = ToolRegistry()
    registry.register(
        name="terminal",
        toolset="terminal",
        schema={"name": "terminal"},
        handler=handler,
    )

    result = json.loads(registry.dispatch(
        "terminal", {"command": "git status --short"}, session_id="",
    ))

    assert result == {"ok": True}
    assert called is True


def test_read_only_tool_remains_allowed_without_active_task(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    registry = ToolRegistry()
    registry.register(
        name="read_file",
        toolset="file",
        schema={"name": "read_file"},
        handler=lambda args, **kwargs: json.dumps({"content": "unchanged"}),
    )

    result = json.loads(registry.dispatch("read_file", {}, session_id="chat-3"))

    assert result == {"content": "unchanged"}


def test_read_only_terminal_remains_allowed_without_active_task(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    registry = ToolRegistry()
    registry.register(
        name="terminal",
        toolset="terminal",
        schema={"name": "terminal"},
        handler=lambda args, **kwargs: json.dumps({"ok": True}),
    )
    assert json.loads(registry.dispatch(
        "terminal", {"command": "git diff --check"}, session_id="chat-3",
    )) == {"ok": True}


def test_delegation_is_refused_before_agent_level_dispatch(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from model_tools import handle_function_call

    result = json.loads(
        handle_function_call(
            "delegate_task",
            {"goal": "implement the requested feature"},
            session_id="chat-4",
        )
    )

    assert result["error_type"] == "kanban_task_required"
    assert result["tool"] == "delegate_task"


def test_agent_runtime_delegation_cannot_bypass_gate(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    from agent.agent_runtime_helpers import invoke_tool

    class Agent:
        session_id = "chat-5"

        def _dispatch_delegate_task(self, args):
            raise AssertionError("delegation handler must not execute")

    result = json.loads(invoke_tool(
        Agent(), "delegate_task", {"goal": "fix the bug"}, "task-5",
    ))

    assert result["error_type"] == "kanban_task_required"
    assert result["tool"] == "delegate_task"
