import json, os, socket, threading
import pytest
from hermes_cli import approval_ipc

@pytest.fixture
def pair(monkeypatch):
    child, parent = socket.socketpair()
    for key, value in ((approval_ipc.FD_ENV, str(child.fileno())), (approval_ipc.PROFILE_ENV, "p"), (approval_ipc.SESSION_ENV, "s"), (approval_ipc.RUN_ENV, "r")):
        monkeypatch.setenv(key, value)
    yield child, parent
    child.close(); parent.close()

def read(sock):
    data = bytearray()
    while b"\n" not in data: data.extend(sock.recv(4096))
    return json.loads(bytes(data).partition(b"\n")[0])

def respond(sock, pending, choice="approve", **extra):
    value = {"type":"approval.decision", "approval_id":pending["approval_id"], "profile":"p", "session":"s", "run":"r", "choice":choice}
    value.update(extra); sock.sendall((json.dumps(value)+"\n").encode())

def test_probe_and_approve(pair):
    _, parent = pair; box=[]
    t=threading.Thread(target=lambda: box.append(approval_ipc.request_approval(label="<write AGENTS.md>", description="redacted", approval_class="protected_instruction_write", ttl_seconds=1)))
    t.start(); pending=read(parent)
    assert pending["protocol"] == "hermes.approval.v1" and pending["choices"] == ["approve","deny"]
    assert pending["approval_class"] == "protected_instruction_write"
    respond(parent, pending); t.join(1); assert box == [approval_ipc.IpcResult("approve")]

def test_deny_disconnect_and_malformed(pair):
    _, parent=pair; box=[]
    t=threading.Thread(target=lambda: box.append(approval_ipc.request_approval(label="x", description="y", approval_class="command", ttl_seconds=1)))
    t.start(); p=read(parent); respond(parent,p,"deny"); t.join(1); assert box == [approval_ipc.IpcResult("deny")]
    parent.close(); result=approval_ipc.request_approval(label="x", description="y", approval_class="command", ttl_seconds=.01)
    assert result is not None and result.choice == "deny" and result.failure == "disconnect"

def test_forged_id_times_out_and_malformed_denies(pair):
    _, parent=pair; box=[]
    t=threading.Thread(target=lambda: box.append(approval_ipc.request_approval(label="x", description="y", approval_class="command", ttl_seconds=.03)))
    t.start(); p=read(parent); respond(parent,p,approval_id="forged"); t.join(1); assert box == [approval_ipc.IpcResult("deny","timeout")]
    box=[]
    t=threading.Thread(target=lambda: box.append(approval_ipc.request_approval(label="x", description="y", approval_class="command", ttl_seconds=1)))
    t.start(); read(parent); parent.sendall(b"bad\n"); t.join(1); assert box == [approval_ipc.IpcResult("deny","malformed")]

def test_missing_channel_is_truthful(monkeypatch):
    for key in (approval_ipc.FD_ENV, approval_ipc.PROFILE_ENV, approval_ipc.SESSION_ENV, approval_ipc.RUN_ENV): monkeypatch.delenv(key, raising=False)
    assert approval_ipc.probe()["source"] == "source_chat_required" and approval_ipc.request_supported() is False
    assert approval_ipc.request_approval(label="x", description="y", approval_class="command") is None
