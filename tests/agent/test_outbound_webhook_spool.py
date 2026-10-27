"""Durable outbound lifecycle hook contracts (opt-in, never the default queue)."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent import outbound_webhooks as hooks
from agent import outbound_webhook_spool as spool

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture(autouse=True)
def _reset_hooks():
    hooks.reset_for_tests()
    yield
    hooks.reset_for_tests()


def _target(url="https://example.test/hook", secret="test-secret"):
    return hooks.WebhookTarget(url=url, events=["on_session_end"], secret=secret, durable=True)


def _delivery(target, identity="a" * 32):
    body = hooks._serialize_payload("on_session_end", {"session_id": "sample"}, identity)
    return hooks._build_delivery("on_session_end", target, body, identity)


def test_opt_in_and_profile_scoped_storage(tmp_path):
    assert hooks.iter_configured_targets({"hooks": {"outbound": [{"url": "https://x", "events": ["on_session_end"]}]}})[0].durable is False
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(); second.mkdir()
    target = _target()
    store = spool.Spool(first)
    assert store.enqueue(target, _delivery(target))
    assert spool.Spool(second).inspect() == []
    assert len(store.inspect()) == 1
    assert os.stat(first / "outbound-webhooks" / "spool.db").st_mode & 0o077 == 0


def test_restart_same_signed_body_and_identity(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    initial = _delivery(target)
    assert store.enqueue(target, initial)
    received = []
    monkeypatch.setattr(hooks, "_post_once", lambda d: (received.append(d) or "ok", "HTTP 200"))
    assert spool.Spool(tmp_path).process({spool.target_key(target): target})
    assert received[0]["body"] == initial["body"]
    assert received[0]["headers"] == initial["headers"]
    assert store.inspect()[0]["state"] == "delivered"


def test_retries_5xx_and_ambiguous_timeout_with_same_id(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    captured = []
    def send(d):
        captured.append(d["headers"]["X-Hermes-Delivery"])
        return ("retry", "timeout") if len(captured) == 1 else ("ok", "HTTP 200")
    monkeypatch.setattr(hooks, "_post_once", send)
    now = time.time() + 1
    assert store.process({spool.target_key(target): target}, now=now)
    assert store.inspect()[0]["state"] == "pending"
    assert not store.process({spool.target_key(target): target}, now=now)
    assert store.process({spool.target_key(target): target}, now=now + 10)
    assert captured == ["a" * 32] * 2
    assert store.inspect()[0]["state"] == "delivered"


def test_attempt_cap_dead_letters_and_discards_sensitive_body(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    calls = []
    monkeypatch.setattr(hooks, "_post_once", lambda d: (calls.append(d) or "retry", "HTTP 503"))
    now = time.time() + 1
    for attempt in range(spool.MAX_ATTEMPTS):
        assert store.process({spool.target_key(target): target}, now=now + attempt * 100)
    assert len(calls) == spool.MAX_ATTEMPTS
    assert store.inspect()[0]["state"] == "dead"
    assert store.inspect()[0]["attempts"] == spool.MAX_ATTEMPTS
    assert store.inspect()[0]["reason"] == "HTTP 503"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT body, signature FROM deliveries").fetchone() == (None, None)


def test_terminal_metadata_is_bounded_after_last_delivery(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    monkeypatch.setattr(spool, "MAX_TERMINAL", 1)
    monkeypatch.setattr(hooks, "_post_once", lambda d: ("terminal", "HTTP 400"))
    for identity in ("a" * 32, "b" * 32):
        assert store.enqueue(target, _delivery(target, identity))
        assert store.process({spool.target_key(target): target})
    assert len(store.inspect()) == 1


@pytest.mark.parametrize("status,expected", [(400, "terminal"), (404, "terminal"), (408, "retry"), (425, "retry"), (429, "retry"), (500, "retry"), (302, "terminal")])
def test_http_classification(status, expected):
    assert hooks._classify_status(status) == expected


def test_real_http_5xx_then_success_retries_same_signed_request(tmp_path):
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen.append((body, self.headers["X-Hermes-Delivery"], self.headers["X-Hermes-Signature-256"]))
            self.send_response(500 if len(seen) == 1 else 200)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = _target(url=f"http://127.0.0.1:{server.server_port}/hook")
        store = spool.Spool(tmp_path)
        assert store.enqueue(target, _delivery(target))
        now = time.time() + 1
        targets = {spool.target_key(target): target}
        assert store.process(targets, now=now)
        assert store.inspect()[0]["state"] == "pending"
        assert store.process(targets, now=now + 2)
        assert store.inspect()[0]["state"] == "delivered"
        assert len(seen) == 2 and seen[0] == seen[1]
    finally:
        server.shutdown()
        server.server_close()


def test_terminal_4xx_discards_body_and_records_metadata(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    monkeypatch.setattr(hooks, "_post_once", lambda d: ("terminal", "HTTP 400"))
    assert store.process({spool.target_key(target): target})
    row = store.inspect()[0]
    assert row["state"] == "dead" and row["reason"] == "HTTP 400"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT body FROM deliveries").fetchone()[0] is None


def test_delivered_sensitive_body_not_retained_in_database_pages(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    identity = "e" * 32
    marker = "sensitive-hook-input-unique-payload"
    body = hooks._serialize_payload("on_session_end", {"session_id": marker}, identity)
    assert store.enqueue(target, hooks._build_delivery("on_session_end", target, body, identity))
    monkeypatch.setattr(hooks, "_post_once", lambda d: ("ok", "HTTP 200"))
    assert store.process({spool.target_key(target): target})
    assert marker.encode() not in store.path.read_bytes()


def test_removed_or_changed_target_never_replays(tmp_path, monkeypatch):
    old = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(old, _delivery(old))
    monkeypatch.setattr(hooks, "_post_once", lambda d: pytest.fail("stale delivery escaped"))
    new = _target(url="https://new.test/hook")
    assert store.process({spool.target_key(new): new})
    assert store.inspect()[0]["reason"] == "target_changed_or_removed"
    assert store.inspect()[0]["state"] == "dead"


def test_signing_key_rotation_discards_old_body(tmp_path, monkeypatch):
    old = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(old, _delivery(old))
    monkeypatch.setattr(hooks, "_post_once", lambda d: pytest.fail("old signed body escaped"))
    rotated = _target(secret="rotated")
    assert store.process({spool.target_key(rotated): rotated})
    assert store.inspect()[0]["reason"] == "invalid_record"


def test_copied_spool_cannot_replay_in_another_profile(tmp_path, monkeypatch):
    target = _target()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    assert spool.Spool(a).enqueue(target, _delivery(target))
    shutil.copytree(a / "outbound-webhooks", b / "outbound-webhooks")
    monkeypatch.setattr(hooks, "_post_once", lambda d: pytest.fail("cross-profile replay"))
    assert spool.Spool(b).process({spool.target_key(target): target})
    assert spool.Spool(b).inspect()[0]["reason"] == "profile_mismatch"
    assert spool.Spool(a).inspect()[0]["state"] == "pending"


def test_expired_lease_reclaim_and_expired_payload_retention(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    now = time.time() + 1
    assert store._claim(now) is not None
    assert not store.process({spool.target_key(target): target}, now=now + spool.LEASE_SECONDS - 1)
    monkeypatch.setattr(hooks, "_post_once", lambda d: ("ok", "HTTP 200"))
    assert store.process({spool.target_key(target): target}, now=now + spool.LEASE_SECONDS + 1)
    assert store.inspect()[0]["state"] == "delivered"
    assert not store.process({}, now=now + spool.RETENTION_SECONDS + 10)
    assert store.inspect() == []


def test_corrupt_database_fails_closed_without_replacing_file(tmp_path):
    store = spool.Spool(tmp_path)
    store.directory.mkdir(mode=0o700)
    store.path.write_bytes(b"not a SQLite database")
    before = store.path.read_bytes()
    assert not store.enqueue(_target(), _delivery(_target()))
    assert store.path.read_bytes() == before


def test_disk_error_diagnostics_do_not_log_secret_or_url(tmp_path, monkeypatch, caplog):
    target = _target(url="https://user:password@example.test/hook", secret="supersecret")
    monkeypatch.setattr(spool.Spool, "_connect", lambda self: (_ for _ in ()).throw(OSError("supersecret password")))
    assert not spool.Spool(tmp_path).enqueue(target, _delivery(target))
    assert "supersecret" not in caplog.text
    assert "password" not in caplog.text


def test_incompatible_schema_refuses_admission(tmp_path):
    store = spool.Spool(tmp_path)
    assert store.enqueue(_target(), _delivery(_target()))
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=999")
    assert not store.enqueue(_target(), _delivery(_target(), "b" * 32))
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 1


def test_registered_worker_drains_durable_event_without_volatile_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    received = threading.Event()
    monkeypatch.setattr(hooks, "_post_once", lambda d: (received.set() or "ok", "HTTP 200"))
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    try:
        target = _target()
        hooks.register_from_config({"hooks": {"outbound": [{
            "url": target.url, "events": target.events, "secret": target.secret, "durable": True,
        }]}})
        manager.invoke_hook("on_session_end", session_id="sample")
        assert received.wait(5)
        assert hooks._delivery_queue.empty()
    finally:
        manager._hooks["on_session_end"] = [cb for cb in manager._hooks.get("on_session_end", [])
                                            if not getattr(cb, "__name__", "").startswith("outbound_webhook[")]


def test_bounds_corruption_and_storage_failure(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    monkeypatch.setattr(spool, "MAX_PENDING", 1)
    assert store.enqueue(target, _delivery(target))
    assert not store.enqueue(target, _delivery(target, "b" * 32))
    assert not store.enqueue(target, {**_delivery(target, "c" * 32), "body": b"x" * (spool.MAX_BODY_BYTES + 1)})
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE deliveries SET body = ? WHERE delivery_id = ?", (b"bad", "a" * 32))
    monkeypatch.setattr(hooks, "_post_once", lambda d: pytest.fail("corrupt delivery escaped"))
    assert store.process({spool.target_key(target): target})
    assert store.inspect()[0]["reason"] == "invalid_record"
    monkeypatch.setattr(spool.Spool, "_connect", lambda self: (_ for _ in ()).throw(OSError("disk full")))
    assert not store.enqueue(target, _delivery(target, "d" * 32))


def test_concurrent_claim_single_sender(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    entered = threading.Event()
    release = threading.Event()
    calls = []
    def send(d):
        calls.append(d)
        entered.set()
        assert release.wait(5)
        return "ok", "HTTP 200"
    monkeypatch.setattr(hooks, "_post_once", send)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(spool.Spool(tmp_path).process, {spool.target_key(target): target})
        assert entered.wait(5)
        second = pool.submit(spool.Spool(tmp_path).process, {spool.target_key(target): target})
        assert not second.result(timeout=5)
        release.set()
        assert first.result(timeout=5)
    assert len(calls) == 1


def test_claim_is_exclusive_across_real_processes(tmp_path, monkeypatch):
    target = _target()
    store = spool.Spool(tmp_path)
    assert store.enqueue(target, _delivery(target))
    entered, release = tmp_path / "entered", tmp_path / "release"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(hooks.__file__).resolve().parents[1])!r})\n"
        "from pathlib import Path\n"
        "from tests.waiting import wait_until\n"
        "from agent import outbound_webhooks as h, outbound_webhook_spool as s\n"
        f"t = h.WebhookTarget(url='https://example.test/hook', events=['on_session_end'], secret='test-secret', durable=True)\n"
        "def send(d):\n"
        f"    Path({str(entered)!r}).touch()\n"
        f"    assert wait_until(lambda: Path({str(release)!r}).exists(), timeout=10)\n"
        "    return ('ok', 'HTTP 200')\n"
        "h._post_once = send\n"
        f"assert s.Spool({str(tmp_path)!r}).process({{s.target_key(t): t}})\n"
    )
    from tests.waiting import wait_until
    with subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        try:
            assert wait_until(entered.exists, timeout=10)
            monkeypatch.setattr(hooks, "_post_once", lambda d: pytest.fail("second process sent"))
            assert not store.process({spool.target_key(target): target})
        finally:
            release.touch()
        _, stderr = child.communicate(timeout=15)
        assert child.returncode == 0, stderr.decode()
    assert store.inspect()[0]["state"] == "delivered"


def test_concurrent_admission_respects_bound(tmp_path, monkeypatch):
    target = _target()
    monkeypatch.setattr(spool, "MAX_PENDING", 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda identity: spool.Spool(tmp_path).enqueue(
            target, _delivery(target, identity)), ("a" * 32, "b" * 32)))
    assert results.count(True) == 1
    assert len(spool.Spool(tmp_path).inspect()) == 1


def test_removed_target_at_registration_drains_old_spool(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = _target()
    assert spool.Spool(tmp_path).enqueue(target, _delivery(target))
    hooks.register_from_config({"hooks": {"outbound": []}})
    from tests.waiting import wait_until
    assert wait_until(lambda: spool.Spool(tmp_path).inspect()[0]["state"] == "dead", timeout=5)
    assert spool.Spool(tmp_path).inspect()[0]["reason"] == "target_changed_or_removed"


def test_secure_directory_refuses_symlink(tmp_path):
    target = _target()
    outside = tmp_path / "outside"; outside.mkdir()
    (tmp_path / "outbound-webhooks").symlink_to(outside)
    assert not spool.Spool(tmp_path).enqueue(target, _delivery(target))
    assert not (outside / "spool.db").exists()


def test_existing_permissive_acl_mask_refused(tmp_path):
    target = _target()
    store = spool.Spool(tmp_path)
    store.directory.mkdir(mode=0o755)
    assert not store.enqueue(target, _delivery(target))
    assert not store.path.exists()


@pytest.mark.parametrize("abrupt", [True, False])
def test_real_process_restart_replays_after_exit(tmp_path, monkeypatch, abrupt):
    target = _target()
    # The producer exits with os._exit: no atexit hooks, worker or queue survives.
    script = ("import os, sys\n"
              f"sys.path.insert(0, {str(Path(hooks.__file__).resolve().parents[1])!r})\n"
              "from agent import outbound_webhooks as h, outbound_webhook_spool as s\n"
              f"t = h.WebhookTarget(url='https://example.test/hook', events=['on_session_end'], secret='test-secret', durable=True)\n"
              "body = h._serialize_payload('on_session_end', {'session_id':'sample'}, 'a'*32)\n"
              f"assert s.Spool({str(tmp_path)!r}).enqueue(t, h._build_delivery('on_session_end', t, body, 'a'*32))\n"
              + ("os._exit(0)\n" if abrupt else "sys.exit(0)\n"))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=15, check=False)
    assert proc.returncode == 0, proc.stderr.decode()
    store = spool.Spool(tmp_path)
    assert store.inspect()[0]["state"] == "pending"
    # The restarted worker gets the exact original signed wire, not a regenerated timestamp.
    observed = []
    monkeypatch.setattr(hooks, "_post_once", lambda d: (observed.append(d) or "ok", "HTTP 200"))
    assert store.process({spool.target_key(target): target})
    assert json.loads(observed[0]["body"])["delivery_id"] == "a" * 32
    assert observed[0]["headers"]["X-Hermes-Signature-256"] == hooks._build_delivery(
        "on_session_end", target, observed[0]["body"], "a" * 32)["headers"]["X-Hermes-Signature-256"]
