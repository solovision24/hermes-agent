"""Turn-end guard for kanban workers, which must end with a terminal board transition
(``kanban_complete`` / ``kanban_request_review`` / ``kanban_request_changes`` / ``kanban_block``).
Some models narrate the next step and stop with no tool calls;
Hermes treats that as a clean exit → ``rc=0`` → dispatcher ``protocol_violation``.
Policy-only: return a bounded synthetic nudge so the loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


# Board transitions that END the worker's run — the kanban tools registered in
# ``tools/kanban_tools.py``. ``kanban_request_review`` (implementer → Review handoff) and
# ``kanban_request_changes`` (reviewer rework verdict: closes the review run and requeues the
# card to its implementer) are terminal exactly like ``kanban_complete`` / ``kanban_block``.
# Missing them fired a false "still ``running``" nudge at a review worker that had just returned
# the card; obeying that nudge records a **false approval** (``kanban_complete``) of the work
# being rejected, or a fabricated blocker on a card another profile already owns.
_TERMINAL_KANBAN_TOOLS = frozenset(
    {
        "kanban_complete",
        "kanban_block",
        "kanban_request_review",
        "kanban_request_changes",
    }
)

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """On when ``HERMES_KANBAN_TASK`` is set, unless ``HERMES_KANBAN_STOP_NUDGE`` disables it."""
    if (os.environ.get("HERMES_KANBAN_STOP_NUDGE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    return bool((os.environ.get("HERMES_KANBAN_TASK") or "").strip())


def _tool_call_name(tc: Any) -> str:
    """Tool name from a dict or object tool call (``function.name`` first, then ``name``)."""
    if isinstance(tc, dict):
        fn = tc.get("function")
        return str((fn.get("name") if isinstance(fn, dict) else tc.get("name")) or "")
    fn = getattr(tc, "function", None)
    return str((getattr(fn, "name", "") if fn is not None else getattr(tc, "name", "")) or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    for msg in filter(lambda m: isinstance(m, dict), messages or ()):
        role = msg.get("role")
        if role == "assistant" and any(
            _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS for tc in msg.get("tool_calls") or []
        ):
            return True
        if role == "tool" and str(msg.get("name") or "") in _TERMINAL_KANBAN_TOOLS:
            return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Synthetic follow-up when a kanban worker exits without a terminal tool; ``None`` when
    the guard should not fire (not a kanban worker, already completed/blocked, budget exhausted)."""
    if (
        not kanban_stop_nudge_enabled()
        or attempts >= max_attempts
        or session_called_kanban_terminal(messages)
    ):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"This session has not ended Task `{tid}` with a board tool. Ending now "
        "without one causes a protocol violation (clean exit with no "
        "`kanban_complete` / `kanban_request_review` / `kanban_request_changes` "
        "/ `kanban_block`).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call the terminal tool that matches your lane:\n"
        "   - implementation done → `kanban_complete(summary=..., artifacts=[...])`\n"
        "   - implementation ready for review → `kanban_request_review(summary=...)`\n"
        "   - reviewer verdict, rework needed → "
        "`kanban_request_changes(reason=...)`\n"
        "   - genuinely blocked → `kanban_block(reason=..., kind=...)`\n\n"
        "If you are the review worker, `kanban_request_changes` is the correct "
        "terminal call for a rework verdict; calling `kanban_complete` on work "
        "you are rejecting records a FALSE approval.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


__all__ = ["build_kanban_stop_nudge", "kanban_stop_nudge_enabled", "session_called_kanban_terminal"]
