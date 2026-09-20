"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    session_called_kanban_terminal,
)


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.


# ── Terminal board transitions (review lane) ──────────────────────────
# ``kanban_request_changes`` (reviewer rework verdict: closes the review run
# and requeues the card to its implementer) and ``kanban_request_review``
# (implementer -> Review handoff) are terminal exactly like ``kanban_complete``
# / ``kanban_block``.  A false "still running" nudge after a correct review
# transition tells the review worker to call ``kanban_complete``, which would
# record a false approval of the work it just rejected, or to fabricate a
# blocker on a card another profile already owns.


def _board_call_round_trip(tool_name: str):
    """Minimal assistant tool_call + tool result pair for ``tool_name``."""
    return [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": tool_name, "tool_call_id": "1", "content": "ok"},
    ]


@pytest.mark.parametrize(
    "tool_name",
    [
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    ],
)
def test_every_terminal_transition_suppresses_the_nudge(clear_kanban_env, tool_name):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review")
    messages = _board_call_round_trip(tool_name)
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages, attempts=0) is None
    assert build_kanban_stop_nudge(messages=messages, attempts=1) is None


def test_review_verdict_is_terminal(clear_kanban_env):
    """Regression: a review run that ended with ``kanban_request_changes`` was
    nudged twice to call ``kanban_complete``/``kanban_block`` for a task that
    was already ``ready`` and requeued to its implementer."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_1a6fef16")
    messages = _board_call_round_trip("kanban_request_changes")
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_nudge_offers_the_review_lane_transition(clear_kanban_env):
    """A review worker that has not ended yet must be told about the rework
    verdict, not only about ``kanban_complete``."""
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_1a6fef16")
    messages = _board_call_round_trip("kanban_heartbeat")
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_request_changes" in nudge
    assert "kanban_request_review" in nudge


def test_terminal_set_matches_the_installed_kanban_surface():
    """Drift guard: every recognised terminal tool must be a real installed
    kanban tool, and the review-lane transitions must stay in the set."""
    from agent.kanban_stop import _TERMINAL_KANBAN_TOOLS
    from tools import kanban_tools as kt

    registered = {name for name, _schema, _handler, _emoji in kt._TOOLS}
    assert _TERMINAL_KANBAN_TOOLS <= registered
    assert {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    } <= set(_TERMINAL_KANBAN_TOOLS)

