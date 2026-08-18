"""Unit tests for the temporary Telegram group profile-prefix fallback.

Covers ``gateway.platforms.base._telegram_group_profile_prefix``: profile-routed
Telegram GROUP replies get a concise uppercase ``[PROFILE] `` prefix so a
shared-bot gateway's specialist replies are identifiable; DMs, other platforms,
the default/Halo profile, and already-prefixed replies are never double-tagged.
"""

from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import _telegram_group_profile_prefix


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
