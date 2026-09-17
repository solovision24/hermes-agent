import io
import json
from urllib.error import HTTPError, URLError

import pytest

from tools import operational_sender

IDENTITY = {
    "ok": True,
    "result": {
        "id": operational_sender.EXPECTED_BOT_ID,
        "username": "solo_hermes_bot",
        "is_bot": True,
    },
}


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _sent_proof(method, data):
    return {
        "ok": True,
        "result": {
            "message_id": 7,
            "from": {
                "id": operational_sender.EXPECTED_BOT_ID,
                "username": "solo_hermes_bot",
                "is_bot": True,
            },
            "chat": {"id": operational_sender.DEFAULT_CHAT_ID},
        },
    }


def test_sender_requires_dedicated_token(monkeypatch):
    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="SOLO_HERMES_BOT_TOKEN"):
        operational_sender.send_operational_message("Updated: test")


def test_sender_verifies_identity_before_send(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    calls = []

    def fake_api(token, method, data):
        calls.append(method)
        return {"result": {"username": "solovision_halo_bot"}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    with pytest.raises(RuntimeError, match="identity"):
        operational_sender.send_operational_message("Updated: test")
    assert calls == ["getMe"]


def test_sender_sends_only_after_exact_identity(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    calls = []

    def fake_api(token, method, data):
        calls.append((method, data))
        if method == "getMe":
            return {"result": {"id": operational_sender.EXPECTED_BOT_ID,
                                "username": "solo_hermes_bot", "is_bot": True}}
        return {"ok": True, "result": {"message_id": 7,
                                         "from": {"id": operational_sender.EXPECTED_BOT_ID,
                                                  "username": "solo_hermes_bot", "is_bot": True},
                                         "chat": {"id": "8148316720"}}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    result = operational_sender.send_operational_message("Updated: test")
    assert result["result"]["message_id"] == 7
    assert [method for method, _ in calls] == ["getMe", "sendMessage"]
    assert calls[1][1] == {"chat_id": "8148316720", "text": "Updated: test"}


def test_sender_does_not_accept_arbitrary_destination(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    monkeypatch.setattr(operational_sender, "_api_call", lambda *_: {
        "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": "solo_hermes_bot", "is_bot": True},
    })
    with pytest.raises(RuntimeError, match="destination"):
        operational_sender.send_operational_message("Updated: test", "different-chat")


# ── Failure diagnosis: HTTP status + bounded response body, no credentials ──


def test_rejected_send_reports_http_status_and_bounded_body(monkeypatch):
    """A rejected sendMessage must say HTTP 400 and why — never just "HTTPError"."""
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret-token")
    body = b'{"ok":false,"error_code":400,"description":"Bad Request: message is too long"}'

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/getMe"):
            return _FakeResponse(json.dumps(IDENTITY).encode())
        raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(operational_sender, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        operational_sender.send_operational_message("x" * 5000)

    text = str(excinfo.value)
    assert "HTTP 400" in text
    assert "message is too long" in text
    # Diagnosis only: the token and the URL that embeds it must never be in the text.
    assert "secret-token" not in text
    assert "api.telegram.org" not in text


def test_error_detail_is_bounded(monkeypatch):
    """A huge response body is truncated, so an incident row stays readable."""
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret-token")

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/getMe"):
            return _FakeResponse(json.dumps(IDENTITY).encode())
        body = b'{"ok":false,"description":"' + b"z" * 4000 + b'"}'
        raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))

    monkeypatch.setattr(operational_sender, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        operational_sender.send_operational_message("short")

    text = str(excinfo.value)
    assert len(text) <= operational_sender.MAX_ERROR_DETAIL_CHARS + 80


def test_url_error_reports_reason_without_url(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret-token")

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/getMe"):
            return _FakeResponse(json.dumps(IDENTITY).encode())
        raise URLError(OSError("Connection refused"))

    monkeypatch.setattr(operational_sender, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        operational_sender.send_operational_message("short")

    text = str(excinfo.value)
    assert "sendMessage" in text
    assert "Connection refused" in text
    assert "api.telegram.org" not in text


def test_ok_false_rejection_keeps_bounded_body(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret-token")

    def fake_urlopen(request, timeout=None):
        if request.full_url.endswith("/getMe"):
            return _FakeResponse(json.dumps(IDENTITY).encode())
        return _FakeResponse(json.dumps({
            "ok": False, "error_code": 403,
            "description": "Forbidden: bot was blocked by the user",
        }).encode())

    monkeypatch.setattr(operational_sender, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as excinfo:
        operational_sender.send_operational_message("short")

    text = str(excinfo.value)
    assert "rejected the request" in text
    assert "bot was blocked by the user" in text
    assert "secret-token" not in text


# ── Chunking over Telegram's 4096-character sendMessage cap ────────────────


def test_long_message_is_chunked_losslessly(monkeypatch):
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "secret")
    sent = []

    def fake_api(token, method, data):
        if method == "getMe":
            return {"result": {"id": operational_sender.EXPECTED_BOT_ID,
                               "username": "solo_hermes_bot", "is_bot": True}}
        sent.append(data["text"])
        return _sent_proof(method, data)

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    message = "\n\n".join(f"line {i} " + ("y" * 90) for i in range(120))
    assert len(message) > operational_sender.TELEGRAM_TEXT_LIMIT

    operational_sender.send_operational_message(message)

    assert len(sent) > 1, "an over-long message must be split, not dropped"
    assert all(len(chunk) <= operational_sender.TELEGRAM_TEXT_LIMIT for chunk in sent)
    assert "".join(sent) == message, "chunking must be lossless"


def test_chunk_message_preserves_text_at_every_boundary():
    for separator in ("\n\n", "\n", " "):
        message = (("a" * 3000) + separator) * 4
        chunks = operational_sender._chunk_message(message, limit=1000)
        assert all(0 < len(chunk) <= 1000 for chunk in chunks)
        assert "".join(chunks) == message
    hard_cut = "z" * 2500
    assert operational_sender._chunk_message(hard_cut, limit=1000) == [
        "z" * 1000, "z" * 1000, "z" * 500]
