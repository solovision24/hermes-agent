import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway import kanban_watchers
from tools import operational_sender


def test_operational_sender_requires_dedicated_token(monkeypatch):
    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SOLO_HERMES_BOT_TOKEN"):
        operational_sender.send_operational_message("Changed: test")


def test_operational_sender_rejects_wrong_identity(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "not-printed")
    calls = []

    def fake_call(token, method, data):
        calls.append(method)
        return {"ok": True, "result": {"username": "solovision_halo_bot"}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_call)
    with pytest.raises(RuntimeError, match="identity"):
        operational_sender.send_operational_message("Changed: test")
    assert calls == ["getMe"]


@pytest.mark.asyncio
async def test_real_watcher_delivery_path_forbids_halo_adapter(monkeypatch):
    adapter = type("Adapter", (), {"send": AsyncMock()})()
    sent = {}

    def fake_operational(message):
        sent.update(message=message)
        return {"ok": True, "result": {"message_id": 99}}

    monkeypatch.setattr(operational_sender, "send_operational_message", fake_operational)
    result = await kanban_watchers._deliver_kanban_text(
        adapter, "telegram", "wrong-chat-is-ignored", "⏸ Kanban t_test blocked", {"thread_id": "topic"}
    )
    assert result["result"]["message_id"] == 99
    assert sent["message"] == "⏸ Kanban t_test blocked"
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_telegram_delivery_preserves_adapter_and_metadata():
    adapter = type("Adapter", (), {"send": AsyncMock(return_value=None)})()
    await kanban_watchers._deliver_kanban_text(
        adapter, "slack", "chat", "Kanban update", {"thread_id": "thread"}
    )
    adapter.send.assert_awaited_once_with("chat", "Kanban update", metadata={"thread_id": "thread"})
