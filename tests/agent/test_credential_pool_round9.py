"""Round9 regressions for canonical Codex pool resync."""
from __future__ import annotations

import json
import threading

import pytest


def _write_store(tmp_path, entry, tokens):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {"openai-codex": {"tokens": tokens, "last_refresh": "OLD-G"}},
        "credential_pool": {"openai-codex": [entry]},
    }))


@pytest.mark.parametrize("status", ["dead", "exhausted"])
def test_resync_clears_omitted_generation_metadata(tmp_path, monkeypatch, status):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    entry = {"id": "omission", "label": "omission", "auth_type": "oauth", "priority": 0,
             "source": "device_code", "access_token": "OLD", "refresh_token": "OLD-R",
             "last_status": status, "expires_at": "2000-01-01T00:00:00Z",
             "expires_at_ms": 946684800000, "token_type": "OldBearer", "scope": "old",
             "tls": "keep"}
    old = {"access_token": "OLD", "refresh_token": "OLD-R", "expires_at": "2000-01-01T00:00:00Z",
           "expires_at_ms": 946684800000, "token_type": "OldBearer", "scope": "old"}
    _write_store(tmp_path, entry, old)
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _save_codex_tokens
    pool = load_pool("openai-codex")
    _save_codex_tokens({"access_token": "NEW", "refresh_token": "NEW-R"}, last_refresh="NEW-G")
    selected = pool.select()
    assert selected is not None
    assert selected.expires_at is None and selected.expires_at_ms is None
    assert "token_type" not in selected.extra and "scope" not in selected.extra
    assert selected.extra["tls"] == "keep"
    saved = json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]["openai-codex"][0]
    assert all(field not in saved for field in ("expires_at", "expires_at_ms", "token_type", "scope"))
    assert saved["tls"] == "keep"


def test_resync_cannot_restore_pool_over_concurrent_login(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setattr("hermes_cli.auth._import_codex_cli_tokens", lambda: None)
    entry = {"id": "race", "label": "race", "auth_type": "oauth", "priority": 0,
             "source": "device_code", "access_token": "A", "refresh_token": "A-R", "last_status": "dead"}
    _write_store(tmp_path, entry, {"access_token": "A", "refresh_token": "A-R"})
    from agent.credential_pool import load_pool
    from hermes_cli.auth import _save_codex_tokens
    pool = load_pool("openai-codex")
    _save_codex_tokens({"access_token": "B", "refresh_token": "B-R"}, last_refresh="B-G")
    replaced = threading.Event()
    resume = threading.Event()
    original = pool._replace_entry

    def paused(old, new):
        original(old, new)
        replaced.set()
        assert resume.wait(timeout=5)

    monkeypatch.setattr(pool, "_replace_entry", paused)
    selector = threading.Thread(target=pool.select, daemon=True)
    selector.start()
    assert replaced.wait(timeout=5)
    writer = threading.Thread(target=lambda: _save_codex_tokens(
        {"access_token": "C", "refresh_token": "C-R"}, last_refresh="C-G"), daemon=True)
    writer.start()
    assert writer.is_alive()
    resume.set()
    selector.join(timeout=5)
    writer.join(timeout=5)
    assert not selector.is_alive() and not writer.is_alive()
    saved = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    assert saved["providers"]["openai-codex"]["tokens"]["access_token"] == "C"
    assert saved["credential_pool"]["openai-codex"][0]["access_token"] == "C"
