"""Fork-local, verified Telegram egress for cron (Kanban t_e805dab0).

No adapter registration or profile credential fallback. Call only after the shared
cron redaction/media-policy boundary; attachments travel unchanged as documents.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BOT_ID = 8611668567
BOT_USERNAME = "solo_hermes_bot"
CHAT_ID = "8148316720"
_FORBIDDEN_RECEIPT_FIELDS = frozenset({
    "message_thread_id", "direct_messages_topic", "direct_messages_topic_id",
    "reply_to_message", "reply_to_story", "external_reply", "quote", "reply_parameters",
    "business_connection_id", "sender_business_bot", "sender_chat", "is_topic_message",
})


class OperationalDeliveryError(RuntimeError):
    """Safe to persist: never includes credentials, response bodies or exception URLs."""


def _root_token() -> str:
    from agent.secret_scope import build_profile_secret_scope
    from hermes_constants import get_default_hermes_root

    # Same root resolver as hermes_cli.profiles._get_default_hermes_home. Read
    # the mapping directly: get_secret may fall back to the launch profile env.
    try:
        token = build_profile_secret_scope(get_default_hermes_root()).get(
            "SOLO_HERMES_BOT_TOKEN", "").strip()
    except Exception:
        raise OperationalDeliveryError("operational Telegram root credential read failed") from None
    if not token:
        raise OperationalDeliveryError("SOLO_HERMES_BOT_TOKEN missing from root secret store")
    if not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
        raise OperationalDeliveryError("operational Telegram root credential is invalid")
    return token


def _api_call(credential: str, method: str, data: dict, document: Path | None = None) -> dict:
    headers = {}
    if document is None:
        body = urlencode(data).encode()
    else:
        # A generated boundary and escaped filename keep arbitrary artifact names
        # from adding multipart fields (especially routing metadata).
        boundary = uuid.uuid4().hex
        filename = document.name.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")
        try:
            payload = document.read_bytes()
        except OSError:
            raise OperationalDeliveryError("operational Telegram attachment read failed") from None
        fields = "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            for key, value in data.items())
        body = (fields +
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"document\"; filename=\"{filename}\"\r\n"
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    try:
        request = Request(
            f"https://api.telegram.org/bot{credential}/{method}",
            data=body, headers=headers, method="POST")
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        # Neither the exception URL nor the API body is safe to log. Do not chain
        # exceptions: traceback rendering can expose the credential-bearing URL.
        raise OperationalDeliveryError(f"operational Telegram {method} HTTP {exc.code}") from None
    except Exception:
        raise OperationalDeliveryError(f"operational Telegram {method} request failed") from None
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise OperationalDeliveryError(f"operational Telegram {method} rejected or invalid response")
    value = result.get("result")
    if not isinstance(value, dict):
        raise OperationalDeliveryError(f"operational Telegram {method} missing proof")
    return value


def _verified_bot(bot) -> bool:
    return (isinstance(bot, dict) and bot.get("id") == BOT_ID
            and str(bot.get("username", "")).lower() == BOT_USERNAME
            and bot.get("is_bot") is True)


def _verify_receipt(sent: dict, *, document: bool = False) -> None:
    chat = sent.get("chat")
    message_id = sent.get("message_id")
    if (not _verified_bot(sent.get("from"))
            or not isinstance(chat, dict) or str(chat.get("id")) != CHAT_ID
            or chat.get("type") != "private"
            or type(message_id) is not int or message_id <= 0
            or _FORBIDDEN_RECEIPT_FIELDS.intersection(sent)):
        raise OperationalDeliveryError("operational Telegram delivery proof failed")
    if document and not (isinstance(sent.get("document"), dict)
                         and sent["document"].get("file_id")):
        raise OperationalDeliveryError("operational Telegram attachment proof failed")


def _message_chunks(text: str):
    """Lossless plain text, bounded even for astral characters (UTF-16 units)."""
    start = units = 0
    for index, char in enumerate(text):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > 4096:
            yield text[start:index]
            start, units = index, 0
        units += width
    if start < len(text):
        yield text[start:]


def send_cron_telegram(target: dict, text: str, media_files: list, *, notify: bool = True) -> None:
    """Verify destination, root credential, identity and every send; never retry.

    Inherited topics are discarded. An explicitly addressed topic is a different
    destination, so refuse it just like a different chat. Partial sends remain
    failures: replaying an uncertain/partially delivered result can duplicate it.
    """
    if (str(target.get("chat_id")) != CHAT_ID
            or (target.get("_resolved_from") == "explicit" and target.get("thread_id") is not None)):
        raise OperationalDeliveryError("operational Telegram destination verification failed")
    if not text.strip() and not media_files:
        raise OperationalDeliveryError("operational Telegram empty delivery")
    token = _root_token()
    if not _verified_bot(_api_call(token, "getMe", {})):
        raise OperationalDeliveryError("operational Telegram identity verification failed")
    destination = {"chat_id": CHAT_ID}
    if not notify:
        destination["disable_notification"] = "true"
    for chunk in _message_chunks(text):
        sent = _api_call(token, "sendMessage", {**destination, "text": chunk})
        _verify_receipt(sent)
    from gateway.platforms.base import validate_media_delivery_path

    for path, _is_voice in media_files:
        # Recheck at the file read, as the common filter may have run well before
        # a multi-chunk send. No profile adapter may rescue a failed attachment.
        safe_path = validate_media_delivery_path(str(path))
        if safe_path is None:
            raise OperationalDeliveryError("operational Telegram attachment denied by media path policy")
        sent = _api_call(token, "sendDocument", destination, Path(safe_path))
        _verify_receipt(sent, document=True)
