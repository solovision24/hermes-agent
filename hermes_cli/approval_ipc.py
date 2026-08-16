"""Secure inherited Unix-socket approval transport for supervised children.

Protocol v1 is newline-delimited JSON over a connected socketpair. The parent
passes one endpoint with HERMES_APPROVAL_IPC_FD and binds it to profile,
session, and run values. No listener or persistent approval state exists.
"""
from __future__ import annotations
import json, os, secrets, socket, threading, time
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1
FEATURE_NAME = "parent_process_approval_ipc"
FD_ENV = "HERMES_APPROVAL_IPC_FD"
PROFILE_ENV = "HERMES_APPROVAL_IPC_PROFILE"
SESSION_ENV = "HERMES_APPROVAL_IPC_SESSION"
RUN_ENV = "HERMES_APPROVAL_IPC_RUN"
MAX_FRAME_BYTES = 64 * 1024
_io_lock = threading.Lock()

@dataclass(frozen=True)
class IpcBinding:
    profile: str
    session: str
    run: str

@dataclass(frozen=True)
class IpcResult:
    choice: str
    failure: str | None = None

def _binding() -> IpcBinding | None:
    values = tuple(os.environ.get(k, "") for k in (PROFILE_ENV, SESSION_ENV, RUN_ENV))
    if not all(values) or any(len(v) > 256 for v in values):
        return None
    return IpcBinding(*values)

def _fd() -> int | None:
    try:
        value = int(os.environ.get(FD_ENV, ""))
        os.fstat(value)
        return value if value >= 0 else None
    except (TypeError, ValueError, OSError):
        return None

def probe() -> dict[str, Any]:
    available = _fd() is not None and _binding() is not None
    return {"protocol": f"hermes.approval.v{PROTOCOL_VERSION}", "version": PROTOCOL_VERSION,
            "feature": FEATURE_NAME, "available": available,
            "transport": "inherited_unix_socketpair" if available else None,
            "choices": ["approve", "deny"],
            "source": "parent_process" if available else "source_chat_required"}

def request_supported() -> bool:
    return bool(probe()["available"])

def _text(value: Any, limit: int) -> str:
    return str(value or "").replace("\x00", " ").strip()[:limit]

def _send(sock: socket.socket, obj: dict[str, Any]) -> None:
    frame = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if len(frame) > MAX_FRAME_BYTES:
        raise ValueError("approval frame too large")
    sock.sendall(frame)

def _recv(sock: socket.socket, deadline: float) -> bytes:
    data = bytearray()
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError
        sock.settimeout(min(left, 1.0))
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError
        data.extend(chunk)
        if len(data) > MAX_FRAME_BYTES:
            raise ValueError("approval frame too large")
        if b"\n" in chunk:
            return bytes(data).partition(b"\n")[0]

def request_approval(*, label: str, description: str, approval_class: str,
                     ttl_seconds: float = 300.0, on_poll=None,
                     is_interrupted=None) -> IpcResult | None:
    fd, binding = _fd(), _binding()
    if fd is None or binding is None:
        return None
    ttl = max(.001, min(float(ttl_seconds), 3600.0))
    approval_id = secrets.token_urlsafe(32)
    created, deadline = time.time(), time.monotonic() + ttl
    pending = {"type": "approval.pending", "protocol": "hermes.approval.v1", "version": 1,
               "approval_id": approval_id, "profile": binding.profile,
               "session": binding.session, "run": binding.run,
               "label": _text(label, 512), "description": _text(description, 2000),
               "choices": ["approve", "deny"], "created_at": created,
               "expires_at": created + ttl, "approval_class": _text(approval_class, 128)}
    with _io_lock:
        try:
            sock = socket.socket(fileno=os.dup(fd))
        except OSError:
            return IpcResult("deny", "unavailable")
        try:
            try:
                _send(sock, pending)
            except (OSError, ValueError):
                return IpcResult("deny", "disconnect")
            while True:
                if is_interrupted is not None and is_interrupted():
                    return IpcResult("deny", "interrupted")
                if on_poll is not None:
                    try: on_poll()
                    except Exception: pass
                try:
                    response = json.loads(_recv(sock, deadline).decode())
                except TimeoutError:
                    return IpcResult("deny", "timeout")
                except (ConnectionError, OSError):
                    return IpcResult("deny", "disconnect")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                    return IpcResult("deny", "malformed")
                if not isinstance(response, dict) or response.get("type") != "approval.decision":
                    return IpcResult("deny", "malformed")
                if any(response.get(k) != getattr(binding, k) for k in ("profile", "session", "run")) or response.get("approval_id") != approval_id:
                    continue
                if response.get("choice") not in {"approve", "deny"}:
                    return IpcResult("deny", "invalid_choice")
                return IpcResult(response["choice"])
        finally:
            sock.close()
