"""Unit tests for the temporary Telegram group profile-prefix fallback.

Covers ``gateway.platforms.base._telegram_group_profile_prefix``: profile-routed
Telegram GROUP replies get a concise uppercase ``[PROFILE] `` prefix so a
shared-bot gateway's specialist replies are identifiable; DMs, other platforms,
the default/Halo profile, and already-prefixed replies are never double-tagged.

Also covers the integration placement: the prefix must be applied AFTER the
auto-TTS and caption blocks so speech synthesis receives the unprefixed text
(the tag must not be spoken aloud) and the 1024-char caption-eligibility check
sees the unprefixed reply (a boundary reply must not silently lose its caption).
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    _telegram_group_profile_prefix,
    build_auto_tts_output_path,
    build_session_key,
)
from gateway.session import SessionSource


def _source(platform=Platform.TELEGRAM, chat_type="group", profile=None):
    return SimpleNamespace(
        platform=platform,
        chat_type=chat_type,
        profile=profile,
    )


def test_group_named_profile_prefixes():
    out = _telegram_group_profile_prefix(
        _source(profile="orion"), "hello world"
    )
    assert out == "[ORION] hello world"


def test_group_profile_lowercase_is_uppercased_in_tag():
    out = _telegram_group_profile_prefix(
        _source(profile="chase"), "reply"
    )
    assert out == "[CHASE] reply"


def test_group_no_profile_stays_unprefixed():
    text = "plain reply"
    assert _telegram_group_profile_prefix(_source(), text) == text


def test_group_default_profile_stays_unprefixed():
    text = "default reply"
    assert _telegram_group_profile_prefix(
        _source(profile="default"), text
    ) == text


def test_group_caseinsensitive_default_stays_unprefixed():
    text = "Default reply"
    assert _telegram_group_profile_prefix(
        _source(profile="Default"), text
    ) == text


def test_dm_never_prefixed():
    text = "dm reply"
    assert _telegram_group_profile_prefix(
        _source(chat_type="dm", profile="orion"), text
    ) == text


def test_non_telegram_platform_never_prefixed():
    text = "slack reply"
    assert _telegram_group_profile_prefix(
        SimpleNamespace(platform=Platform.SLACK, chat_type="group", profile="orion"),
        text,
    ) == text


def test_channel_never_prefixed():
    text = "channel reply"
    assert _telegram_group_profile_prefix(
        _source(chat_type="channel", profile="orion"), text
    ) == text


def test_empty_text_is_unchanged():
    assert _telegram_group_profile_prefix(_source(profile="orion"), "") == ""
    assert _telegram_group_profile_prefix(_source(profile="orion"), "  ") == "  "


def test_already_prefixed_reply_not_double_prefixed():
    text = "[ORION] hello"
    assert _telegram_group_profile_prefix(
        _source(profile="orion"), text
    ) == text


def test_already_prefixed_other_agent_not_double_prefixed():
    text = "[VECTOR] hello"
    assert _telegram_group_profile_prefix(
        _source(profile="chase"), text
    ) == text


def test_leading_whitespace_before_existing_prefix_guarded():
    text = "  [ORION] hello"
    assert _telegram_group_profile_prefix(
        _source(profile="orion"), text
    ) == text


def test_long_boundary_reply_not_double_prefixed():
    text = "[ORION] " + "x" * 2000
    assert _telegram_group_profile_prefix(
        _source(profile="orion"), text
    ) == text


# ---------------------------------------------------------------------------
# Integration: prefix applied AFTER the auto-TTS + caption blocks so the tag is
# not spoken aloud and does not break the 1024-char caption eligibility.
# ---------------------------------------------------------------------------


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


class _DummyAdapter(BasePlatformAdapter):
    """Minimal in-process adapter for final-delivery integration tests."""

    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append({"chat_id": chat_id, "content": content})
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_voice_event(platform: Platform, profile: str | None = None) -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=platform,
            chat_id="-1001",
            chat_type="group",
            profile=profile,
        ),
        message_id="voice-1",
    )


@pytest.mark.asyncio
async def test_prefix_applied_after_auto_tts_tag_not_spoken():
    """The profile tag must NOT appear in TTS speech text or caption.

    A Telegram group voice reply routed to the 'orion' profile must be
    audible as the bare reply (no '[ORION] …'), while the visible text
    message carries the prefix.
    """
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="tts-1")
    )
    # Body longer than 1024 chars so the TTS caption-eligibility check
    # fails and the text send is performed (lets us assert on the prefix).
    short_reply = "B" * 1100
    adapter.set_message_handler(
        lambda _event: asyncio.sleep(0, result=short_reply)
    )
    event = _make_voice_event(Platform.TELEGRAM, profile="orion")
    tts_calls = []

    def fake_tts(*, text, output_path=None):
        tts_calls.append(text)
        out = output_path or build_auto_tts_output_path(Platform.TELEGRAM)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"fake audio")
        return json.dumps({"success": True, "file_path": str(out)})

    with patch(
        "tools.tts_tool.check_tts_requirements", return_value=True
    ), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

    # Speech synthesis received the UNPREFIXED text — the tag is not spoken.
    assert tts_calls == [short_reply], tts_calls

    # Caption is None: the 1024-char eligibility check sees the UNPREFIXED
    # text (1100 chars > 1024), so no caption is attached to the voice note.
    assert adapter.play_tts.await_count == 1
    caption = adapter.play_tts.await_args.kwargs.get("caption")
    assert caption is None
    assert "[ORION]" not in (caption or "")

    # The final text send carries the prefix.
    assert adapter.sent, adapter.sent
    assert adapter.sent[0]["content"] == f"[ORION] {short_reply}"


@pytest.mark.asyncio
async def test_prefix_applied_after_caption_boundary_check():
    """A ~1024-char reply keeps its caption when the prefix is applied post-TTS.

    Before the fix, the prefix was applied before the caption-eligibility
    check ``text_content[:1024] == text_content``, so a 1017-char reply
    became 1025 chars once prefixed, flipping eligibility to False and
    silently dropping the caption. With the fix the unprefixed 1017-char
    reply passes the check and the caption is delivered (without the tag).
    """
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="tts-1")
    )

    # 1017 chars: [ORION]  (8 chars) would push it to 1025, failing the
    # unprefixed text_content[:1024] == text_content check.
    body = "B" * 1017
    adapter.set_message_handler(
        lambda _event: asyncio.sleep(0, result=body)
    )
    event = _make_voice_event(Platform.TELEGRAM, profile="orion")

    def fake_tts(*, text, output_path=None):
        out = output_path or build_auto_tts_output_path(Platform.TELEGRAM)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"fake audio")
        return json.dumps({"success": True, "file_path": str(out)})

    with patch(
        "tools.tts_tool.check_tts_requirements", return_value=True
    ), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

    # Caption eligibility checked on the UNPREFIXED text (1017 chars < 1024),
    # so the caption is delivered — and it carries no prefix.
    assert adapter.play_tts.await_count == 1
    caption = adapter.play_tts.await_args.kwargs.get("caption")
    assert caption == body  # unprefixed, under the 1024-char boundary

    # TTS caption was delivered → text send is skipped entirely.
    assert adapter.sent == []


@pytest.mark.asyncio
async def test_prefix_not_applied_when_tts_caption_delivered():
    """When TTS caption is delivered, the follow-up text send is skipped."""
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda chat_id: True
    adapter.play_tts = AsyncMock(
        return_value=SendResult(success=True, message_id="tts-1")
    )
    short_reply = "hi there"
    adapter.set_message_handler(
        lambda _event: asyncio.sleep(0, result=short_reply)
    )
    event = _make_voice_event(Platform.TELEGRAM, profile="orion")

    def fake_tts(*, text, output_path=None):
        out = output_path or build_auto_tts_output_path(Platform.TELEGRAM)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(b"fake audio")
        return json.dumps({"success": True, "file_path": str(out)})

    with patch(
        "tools.tts_tool.check_tts_requirements", return_value=True
    ), patch(
        "tools.tts_tool.text_to_speech_tool", side_effect=fake_tts
    ):
        await adapter._process_message_background(
            event, build_session_key(event.source)
        )

    # Caption carries the UNPREFIXED text (short enough for eligibility).
    assert adapter.play_tts.await_count == 1
    caption = adapter.play_tts.await_args.kwargs.get("caption")
    assert caption == short_reply

    # TTS caption was delivered → text send is skipped entirely.
    assert adapter.sent == []
