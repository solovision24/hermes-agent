"""hermes.approval.v1 — parent-process approval IPC (Hermes runtime half).

Mission Control implements the registry half of this wire protocol in
``src-tauri/src/approval_ipc.rs``; this module is the Hermes runtime half:

* ``probe()`` — the exact payload MC's ``validate_runtime_probe`` requires. MC
  runs ``python -c "from hermes_cli.approval_ipc import probe"`` against the
  runtime venv before each chat spawn and only wires the approval channel when
  the probe validates, so the module must stay importable with stdlib only.
* ``request_approval(...)`` — write one ``approval.pending`` frame on the
  inherited socketpair, then block for the parent's ``approval.decision``
  frame. Sequential approvals share the one channel; requests are serialized
  process-wide so pending frames and their decisions cannot interleave.

Transport: when the probe validates, MC spawns the chat child with a
socketpair dup2'd onto fd ``HERMES_APPROVAL_IPC_FD`` and binding env vars
``HERMES_APPROVAL_IPC_PROFILE`` / ``HERMES_APPROVAL_IPC_SESSION`` /
``HERMES_APPROVAL_IPC_RUN``. Frames are newline-delimited JSON in both
directions. MC validates every pending frame: exact protocol/version/type,
choices exactly ``["approve", "deny"]``, binding echo, expiry strictly after
creation and at most one hour out, and size caps — malformed or
scope-widening frames fail closed on the MC side, so this module must emit
exactly the contract shape and nothing wider.

Fail-closed contract (mirrors the protected-instruction gate this serves):
``timeout`` and ``closed`` are never consent; only ``approve`` is.
"""

from __future__ import annotations

import json
import os
import select
import socket
import threading
import time
import uuid

APPROVAL_PROTOCOL = "hermes.approval.v1"
APPROVAL_VERSION = 1
APPROVAL_CLASS_PROTECTED_INSTRUCTION_WRITE = "protected_instruction_write"
CHOICES = ("approve", "deny")

# Wire caps mirrored from MC's registry (approval_ipc.rs): a frame larger than
# this is rejected before parsing, so cap both directions here too.
MAX_FRAME_BYTES = 64 * 1024
MAX_LABEL_CHARS = 512
MAX_DESCRIPTION_CHARS = 2_000
MAX_APPROVAL_CLASS_CHARS = 128
MAX_APPROVAL_ID_CHARS = 256
# MC rejects pending frames whose expiry is more than one hour after creation.
MAX_TTL_SECONDS = 3_600.0
# Approval prompts wait up to 15 minutes by default — comfortably inside MC's
# frame TTL cap and its 15-minute chat idle cadence; callers may override.
DEFAULT_TIMEOUT_SECONDS = 900.0

# Result strings returned by request_approval().
RESULT_APPROVE = "approve"
RESULT_DENY = "deny"
RESULT_TIMEOUT = "timeout"
RESULT_CLOSED = "closed"
RESULT_UNAVAILABLE = "unavailable"

_ENV_FD = "HERMES_APPROVAL_IPC_FD"
_ENV_PROFILE = "HERMES_APPROVAL_IPC_PROFILE"
_ENV_SESSION = "HERMES_APPROVAL_IPC_SESSION"
_ENV_RUN = "HERMES_APPROVAL_IPC_RUN"

_STALE_DECISION = "stale_decision"


def _clip(text: str, max_chars: int) -> str:
    """Fit MC's per-field caps: truncate over-long text and strip NULs (both are
    InvalidFrame on the MC side)."""
    return str(text).replace("\0", "")[:max_chars]


class ApprovalIpcChannel:
    """One authenticated approval channel to the parent process.

    Holds the child end of the socketpair MC dup2'd onto ``HERMES_APPROVAL_IPC_FD``
    plus the binding (profile/session/run) every pending frame must echo. The
    wrapped fd is duplicated at open time so forked grandchildren do not inherit
    a racing writer onto the same channel.
    """

    def __init__(self, sock: socket.socket, profile: str, session: str, run: str) -> None:
        self._sock = sock
        self.profile = profile
        self.session = session
        self.run = run
        self._read_buffer = bytearray()
        self._request_lock = threading.Lock()

    def request_approval(self, *, label: str, description: str,
                         approval_class: str = APPROVAL_CLASS_PROTECTED_INSTRUCTION_WRITE,
                         timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
        """Emit one ``approval.pending`` frame and block for its decision.

        Returns one of RESULT_APPROVE / RESULT_DENY / RESULT_TIMEOUT /
        RESULT_CLOSED. Never raises: a broken channel is RESULT_CLOSED, which
        the caller must treat as no-consent.
        """
        with self._request_lock:
            try:
                return self._request_approval_locked(
                    label=label, description=description,
                    approval_class=approval_class, timeout_seconds=timeout_seconds)
            except Exception:
                return RESULT_CLOSED

    # ── internals ────────────────────────────────────────────────────────

    def _request_approval_locked(self, *, label: str, description: str,
                                 approval_class: str, timeout_seconds: float) -> str:
        created = time.time()
        ttl = max(0.1, min(float(timeout_seconds or DEFAULT_TIMEOUT_SECONDS), MAX_TTL_SECONDS))
        approval_id = f"hermes-{os.getpid()}-{uuid.uuid4().hex}"
        frame = {
            "protocol": APPROVAL_PROTOCOL,
            "version": APPROVAL_VERSION,
            "type": "approval.pending",
            "approval_id": approval_id,
            "profile": self.profile,
            "session": self.session,
            "run": self.run,
            "label": _clip(label, MAX_LABEL_CHARS),
            "description": _clip(description, MAX_DESCRIPTION_CHARS),
            "approval_class": _clip(approval_class, MAX_APPROVAL_CLASS_CHARS),
            "choices": [CHOICES[0], CHOICES[1]],
            "created_at": created,
            "expires_at": created + ttl,
        }
        try:
            payload = json.dumps(frame, separators=(",", ":")).encode("utf-8")
            if len(payload) > MAX_FRAME_BYTES:
                return RESULT_CLOSED
            self._sock.sendall(payload + b"\n")
        except OSError:
            return RESULT_CLOSED

        deadline = time.monotonic() + ttl
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return RESULT_TIMEOUT
            line = self._read_line(remaining)
            if line is None:  # deadline hit inside _read_line
                return RESULT_TIMEOUT
            if line == b"":  # EOF: parent closed the channel (cancel/kill)
                return RESULT_CLOSED
            decision = self._parse_decision(line, approval_id)
            if decision == _STALE_DECISION:
                # A decision for a different approval id (late resolution of an
                # earlier request whose local deadline already fired). Never
                # act on it; keep reading for our own decision.
                continue
            return decision

    def _read_line(self, timeout_seconds: float) -> bytes | None:
        """Read through the next newline. ``None`` on timeout, ``b""`` on EOF."""
        buffer = self._read_buffer
        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline])
                del buffer[:newline + 1]
                return line
            if len(buffer) > MAX_FRAME_BYTES:
                return b""  # oversized garbage — fail closed like MC's reader
            ready, _, _ = select.select([self._sock], [], [], max(0.0, timeout_seconds))
            if not ready:
                return None
            try:
                chunk = self._sock.recv(64 * 1024)
            except OSError:
                return b""
            if not chunk:
                return b""
            buffer.extend(chunk)

    def _parse_decision(self, line: bytes, approval_id: str) -> str:
        try:
            value = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return RESULT_CLOSED
        if not isinstance(value, dict):
            return RESULT_CLOSED
        if (value.get("protocol") != APPROVAL_PROTOCOL
                or value.get("version") != APPROVAL_VERSION
                or value.get("type") != "approval.decision"):
            return RESULT_CLOSED
        if value.get("approval_id") != approval_id:
            return _STALE_DECISION
        if (value.get("profile") != self.profile
                or value.get("session") != self.session
                or value.get("run") != self.run):
            return RESULT_CLOSED
        choice = value.get("choice")
        if choice == RESULT_APPROVE:
            return RESULT_APPROVE
        if choice == RESULT_DENY:
            return RESULT_DENY
        return RESULT_CLOSED


def _env_binding() -> tuple[int, str, str, str] | None:
    """Return ``(fd, profile, session, run)`` from the MC handoff env, or ``None``
    when any piece is missing or malformed. Never opens the fd."""
    raw_fd = os.environ.get(_ENV_FD, "").strip()
    if not raw_fd:
        return None
    try:
        fd = int(raw_fd)
    except ValueError:
        return None
    profile = os.environ.get(_ENV_PROFILE, "").strip()
    session = os.environ.get(_ENV_SESSION, "").strip()
    run = os.environ.get(_ENV_RUN, "").strip()
    if not (profile and session and run):
        return None
    try:
        os.fstat(fd)
    except (OSError, ValueError):
        return None
    return fd, profile, session, run


_channel_lock = threading.Lock()
_channel_cache: ApprovalIpcChannel | None = None
_channel_checked = False


def _reset_channel_cache_for_tests() -> None:
    global _channel_cache, _channel_checked
    with _channel_lock:
        _channel_cache = None
        _channel_checked = False


def channel_from_env() -> ApprovalIpcChannel | None:
    """The process-wide approval channel, opened lazily from the MC handoff env.

    ``None`` when this process was not spawned with a validated approval channel.
    The fd is duplicated (non-inheritable) so forked subprocesses cannot write
    racing frames onto the channel.
    """
    global _channel_cache, _channel_checked
    with _channel_lock:
        if _channel_checked:
            return _channel_cache
        _channel_checked = True
        binding = _env_binding()
        if binding is None:
            _channel_cache = None
            return None
        fd, profile, session, run = binding
        try:
            dup_fd = os.dup(fd)
            sock = socket.socket(fileno=dup_fd)
        except OSError:
            _channel_cache = None
            return None
        _channel_cache = ApprovalIpcChannel(sock, profile, session, run)
        return _channel_cache


def probe() -> dict:
    """Return the exact runtime-probe payload MC's ``validate_runtime_probe`` checks.

    ``available`` reflects THIS process's live channel (the probe subprocess MC
    spawns has no fd env, so it reports ``False`` but still validates — that is
    what makes MC wire the fd into the real chat child).
    """
    available = _env_binding() is not None
    return {
        "protocol": APPROVAL_PROTOCOL,
        "version": APPROVAL_VERSION,
        "feature": "parent_process_approval_ipc",
        "choices": [CHOICES[0], CHOICES[1]],
        "available": available,
        "source": "parent_process_approval_ipc" if available else "source_chat_required",
    }


def request_approval(*, label: str, description: str,
                     approval_class: str = APPROVAL_CLASS_PROTECTED_INSTRUCTION_WRITE,
                     timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> str:
    """One-shot approve/deny round trip on the parent-process approval channel.

    Returns RESULT_UNAVAILABLE when this process has no approval channel (the
    caller then falls back to its other surfaces); otherwise one of
    RESULT_APPROVE / RESULT_DENY / RESULT_TIMEOUT / RESULT_CLOSED.
    """
    channel = channel_from_env()
    if channel is None:
        return RESULT_UNAVAILABLE
    return channel.request_approval(
        label=label, description=description,
        approval_class=approval_class, timeout_seconds=timeout_seconds)
