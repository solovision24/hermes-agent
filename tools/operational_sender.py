"""Fail-closed sender for verified Hermes operational notifications."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

EXPECTED_USERNAME = "solo_hermes_bot"
EXPECTED_BOT_ID = 8611668567
DEFAULT_CHAT_ID = "8148316720"

# Telegram rejects sendMessage text longer than this with HTTP 400 "message is too long".
TELEGRAM_TEXT_LIMIT = 4096
MAX_ERROR_DETAIL_CHARS = 300


def _bound_detail(text: object, limit: int = MAX_ERROR_DETAIL_CHARS) -> str:
    """Collapse whitespace, redact secrets (best-effort) and bound error detail for a message.

    Callers pass response bodies / reasons only — never the request URL, which embeds the bot token.
    """
    detail = re.sub(r"\s+", " ", str(text or "")).strip()
    try:
        from agent.redact import redact_sensitive_text

        detail = redact_sensitive_text(detail, force=True)
    except Exception:
        pass
    return detail[:limit]


def _http_error_detail(exc: HTTPError) -> str:
    """Bounded, redacted response body of an HTTPError ("" when it cannot be read)."""
    try:
        body = exc.read()
    except Exception:
        return ""
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    return _bound_detail(body)


def _describe_api_failure(method: str, exc: BaseException) -> str:
    """Diagnosis-carrying error text: HTTP status + bounded, redacted response body.

    Shape matches the ops-side helper (``Telegram {method} failed with HTTP {code}: {detail}``).
    The status and body are what a human needs to tell "message is too long" from a transient
    api.telegram.org failure; the old ``type(exc).__name__``-only text left the operator guessing.
    The request URL is never included — it embeds the bot token.
    """
    if isinstance(exc, HTTPError):
        detail = _http_error_detail(exc)
        if detail:
            return f"Telegram {method} failed with HTTP {exc.code}: {detail}"
        return f"Telegram {method} failed with HTTP {exc.code}"
    if isinstance(exc, URLError):
        return f"Telegram {method} failed: {_bound_detail(getattr(exc, 'reason', exc))}"
    return f"Telegram {method} failed: {type(exc).__name__}: {_bound_detail(exc)}"


def _chunk_message(message: str, limit: int = TELEGRAM_TEXT_LIMIT) -> list[str]:
    """Split text into consecutive chunks of at most ``limit`` characters.

    ``sendMessage`` fails closed above Telegram's 4096-character cap, so a lane whose output is
    longer than that would never reach the operator. Splits prefer a paragraph break, then a line
    break, then a space, and only then a hard cut. Lossless by construction:
    ``"".join(_chunk_message(text)) == text`` — no character is added, moved or dropped.
    """
    text = str(message or "")
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = 0
        for separator in ("\n\n", "\n", " "):
            index = window.rfind(separator)
            if index > 0:
                cut = index
                break
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    chunks.append(remaining)
    return chunks


def _api_call(token: str, method: str, data: dict[str, str]) -> dict:
    request = Request(
        f"https://api.telegram.org/bot{token}/{method}",
        data=urlencode(data).encode(),
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(_describe_api_failure(method, exc)) from exc
    if not payload.get("ok"):
        raise RuntimeError(
            "Telegram operational sender rejected the request: "
            f"{_bound_detail(json.dumps(payload, ensure_ascii=False))}"
        )
    return payload


def send_operational_message(message: str, chat_id: str = DEFAULT_CHAT_ID) -> dict:
    """Verify @solo_hermes_bot and send one operational message, chunked over Telegram's cap.

    Chunking is lossless (see ``_chunk_message``); a rejection part-way through leaves the earlier
    chunks delivered and raises with the HTTP status + body, so the caller's delivery incident
    records exactly what happened instead of a bare exception type.

    There is deliberately no fallback to TELEGRAM_BOT_TOKEN or a gateway
    adapter: operational notifications must never silently use a profile bot.
    """
    # This credential is intentionally process-global: lifecycle notifications run
    # outside any routed profile scope and must not inherit another profile's bot.
    token = os.environ.get("SOLO_HERMES_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    bot = identity.get("result", {})
    if (bot.get("id") != EXPECTED_BOT_ID
            or str(bot.get("username", "")).lower() != EXPECTED_USERNAME
            or bot.get("is_bot") is not True):
        raise RuntimeError("operational sender identity verification failed")
    if str(chat_id) != DEFAULT_CHAT_ID:
        raise RuntimeError("operational sender destination verification failed")
    result: dict = {}
    for chunk in _chunk_message(message):
        result = _api_call(token, "sendMessage", {"chat_id": DEFAULT_CHAT_ID, "text": chunk})
        sent = result.get("result", {})
        sent_from, sent_chat = sent.get("from", {}), sent.get("chat", {})
        if (sent_from.get("id") != EXPECTED_BOT_ID
                or str(sent_from.get("username", "")).lower() != EXPECTED_USERNAME
                or sent_from.get("is_bot") is not True
                or str(sent_chat.get("id")) != DEFAULT_CHAT_ID
                or "message_thread_id" in sent):
            raise RuntimeError("operational sender delivery proof failed")
    return result


def send_operational_document(file_path: str) -> dict:
    """Send one artifact to the verified operational DM, without a topic."""
    token = os.environ.get("SOLO_HERMES_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SOLO_HERMES_BOT_TOKEN is not configured")
    identity = _api_call(token, "getMe", {})
    bot = identity.get("result", {})
    if (bot.get("id") != EXPECTED_BOT_ID
            or str(bot.get("username", "")).lower() != EXPECTED_USERNAME
            or bot.get("is_bot") is not True):
        raise RuntimeError("operational sender identity verification failed")
    path = Path(file_path)
    boundary = "----hermes-operational-sender"
    with path.open("rb") as artifact:
        payload = artifact.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=chat_id\r\n\r\n{DEFAULT_CHAT_ID}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=document; filename={path.name}\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    request = Request(
        f"https://api.telegram.org/bot{token}/sendDocument",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode())
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise RuntimeError(_describe_api_failure("sendDocument", exc)) from exc
    if not result.get("ok"):
        raise RuntimeError(
            "Telegram operational sender rejected the request: "
            f"{_bound_detail(json.dumps(result, ensure_ascii=False))}"
        )
    sent = result.get("result", {})
    sent_from, sent_chat = sent.get("from", {}), sent.get("chat", {})
    if (sent_from.get("id") != EXPECTED_BOT_ID
            or str(sent_from.get("username", "")).lower() != EXPECTED_USERNAME
            or sent_from.get("is_bot") is not True
            or str(sent_chat.get("id")) != DEFAULT_CHAT_ID
            or "message_thread_id" in sent):
        raise RuntimeError("operational sender delivery proof failed")
    return result
