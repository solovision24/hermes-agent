"""Tests for Buzz reply-anchor special-case in _reply_anchor_for_event().

Buzz's ``messages send --reply-to`` CLI flag creates a Nostr thread.  The
adapter must only pass it when the inbound event was itself part of a thread
(has source.thread_id set from an ``["e", ..., "", "reply"]`` tag).  Flat DMs
and channel messages must produce no reply anchor so the agent response lands
inline rather than inside a Thread panel.

These tests exercise the base-gateway function that decides reply anchoring;
the event-level thread_id parsing is tested separately in test_buzz_adapter.py.
"""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    _reply_anchor_for_event,
)
from gateway.session import SessionSource


def _source(
    chat_type: str = "dm",
    thread_id: str | None = None,
) -> SessionSource:
    """Build a Buzz SessionSource for testing."""
    return SessionSource(
        platform=Platform("buzz"),
        chat_id="ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd",
        chat_name="test-channel",
        chat_type=chat_type,
        user_id="a" * 64,
        user_name="Alice",
        thread_id=thread_id,
    )


def test_dm_message_no_thread_returns_none():
    """A plain DM (no thread ancestry) must not produce a reply anchor."""
    event = MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=_source(chat_type="dm", thread_id=None),
        message_id="e1",
    )
    assert _reply_anchor_for_event(event) is None


def test_channel_message_no_thread_returns_none():
    """A flat channel message (no thread ancestry) must not produce a reply anchor."""
    event = MessageEvent(
        text="hello everyone",
        message_type=MessageType.TEXT,
        source=_source(chat_type="group", thread_id=None),
        message_id="e1",
    )
    assert _reply_anchor_for_event(event) is None


def test_thread_reply_preserves_anchor():
    """A message that IS a thread reply (has thread_id set) must keep the anchor."""
    event = MessageEvent(
        text="replying in thread",
        message_type=MessageType.TEXT,
        source=_source(chat_type="group", thread_id="root-event-id"),
        message_id="e2",
    )
    assert _reply_anchor_for_event(event) == "e2"


def test_dm_prose_no_mention_no_thread():
    """A DM without explicit mention (prose DM) is still flat."""
    event = MessageEvent(
        text="just a note without @mention",
        message_type=MessageType.TEXT,
        source=_source(chat_type="dm", thread_id=None),
        message_id="e1",
    )
    assert _reply_anchor_for_event(event) is None
