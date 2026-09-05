"""Behavioral coverage for canonical Codex generations and profile isolation."""

from __future__ import annotations

import json
import multiprocessing
import threading
import time
from pathlib import Path

import pytest


CODEX_ID = "SYNTHETIC_CODEX_SINGLETON"
ROOT_ACCESS = "SYNTHETIC_ROOT_ACCESS_0"
ROOT_REFRESH = "SYNTHETIC_ROOT_REFRESH_0"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _codex_entry(**overrides: object) -> dict:
    entry = {
        "id": CODEX_ID,
        "label": "synthetic-codex",
        "auth_type": "oauth",
        "priority": 0,
        "source": "device_code",
        "access_token": ROOT_ACCESS,
        "refresh_token": ROOT_REFRESH,
        "last_refresh": "SYNTHETIC_GENERATION_0",
        "expires_at": "2099-01-01T00:00:00Z",
        "expires_at_ms": 4070908800000,
        "token_type": "Bearer",
        "scope": "synthetic",
    }
    entry.update(overrides)
    return entry


def _canonical_store(*, entry: dict | None = None) -> dict:
    return {
        "version": 1,
        "providers": {
            "openai-codex": {
                "tokens": {
                    "access_token": ROOT_ACCESS,
                    "refresh_token": ROOT_REFRESH,
                    "expires_at": "2099-01-01T00:00:00Z",
                    "expires_at_ms": 4070908800000,
                },
                "last_refresh": "SYNTHETIC_GENERATION_0",
                "auth_mode": "chatgpt",
            },
            "openrouter": {
                "api_key": "SYNTHETIC_ROOT_OTHER_PROVIDER",
            },
        },
        "credential_pool": {
            "openai-codex": [entry or _codex_entry()],
            "openrouter": [
                {
                    "id": "SYNTHETIC_ROOT_OPENROUTER",
                    "label": "root-other-provider",
                    "auth_type": "api_key",
                    "priority": 0,
                    "source": "manual",
                    "access_token": "SYNTHETIC_ROOT_OTHER_PROVIDER",
                }
            ],
        },
    }


@pytest.fixture()
def canonical_profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    profile_a = root / "profiles" / "vector"
    profile_b = root / "profiles" / "orion"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("HERMES_HOME", str(profile_a))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "empty-codex"))
    _write(root / "auth.json", _canonical_store())
    return {"root": root, "vector": profile_a, "orion": profile_b}


@pytest.mark.parametrize("mode", ["equal", "missing", "expired"])
def test_status_snapshot_preserves_complete_canonical_generation(
    canonical_profile_env, mode
):
    """Status-only writes use generation identity, not timestamp ordering."""
    from hermes_cli.auth import write_credential_pool

    disk_entry = _codex_entry(
        access_token="SYNTHETIC_FRESH_ACCESS",
        refresh_token="SYNTHETIC_FRESH_REFRESH",
        last_refresh=None if mode == "missing" else (
            "1970-01-01T00:00:00Z" if mode == "expired" else "SYNTHETIC_COLLISION"
        ),
        expires_at="2099-12-31T00:00:00Z",
        expires_at_ms=4102358400000,
    )
    if disk_entry["last_refresh"] is None:
        disk_entry.pop("last_refresh")
    _write(canonical_profile_env["root"] / "auth.json", _canonical_store(entry=disk_entry))

    stale = _codex_entry(
        access_token="SYNTHETIC_STALE_ACCESS",
        refresh_token="SYNTHETIC_STALE_REFRESH",
        last_refresh=(
            "SYNTHETIC_COLLISION" if mode == "equal" else "SYNTHETIC_OLD_GENERATION"
        ),
        expires_at="2000-01-01T00:00:00Z",
        expires_at_ms=946684800000,
        last_status="exhausted",
    )
    write_credential_pool("openai-codex", [stale])

    persisted = json.loads((canonical_profile_env["root"] / "auth.json").read_text())
    result = persisted["credential_pool"]["openai-codex"][0]
    assert result["access_token"] == "SYNTHETIC_FRESH_ACCESS"
    assert result["refresh_token"] == "SYNTHETIC_FRESH_REFRESH"
    assert result["expires_at"] == "2099-12-31T00:00:00Z"
    assert result["expires_at_ms"] == 4102358400000
    if mode == "missing":
        assert "last_refresh" not in result
    else:
        assert result["last_refresh"] == disk_entry["last_refresh"]
    assert persisted["credential_pool"]["openrouter"][0]["access_token"] == (
        "SYNTHETIC_ROOT_OTHER_PROVIDER"
    )


@pytest.mark.parametrize("mode", ["equal", "missing"])
def test_explicit_login_and_refresh_replacement_carry_generation_metadata(
    canonical_profile_env, mode
):
    from hermes_cli.auth import _save_codex_tokens, write_credential_pool

    old = _codex_entry(
        last_status="dead",
        last_error_reason="invalid_grant",
        expires_at="2000-01-01T00:00:00Z",
        expires_at_ms=946684800000,
    )
    _write(canonical_profile_env["root"] / "auth.json", _canonical_store(entry=old))

    login_tokens = {
        "access_token": "SYNTHETIC_LOGIN_ACCESS",
        "refresh_token": "SYNTHETIC_LOGIN_REFRESH",
        "expires_at": "2099-06-01T00:00:00Z",
        "expires_at_ms": 4082918400000,
    }
    # Keep the source value literal and synthetic while exercising an explicit
    # login replacement with a timestamp collision or generated timestamp.
    login_tokens["expires_at_ms"] = 4082918400000
    login_timestamp = "SYNTHETIC_COLLISION" if mode == "equal" else None
    _save_codex_tokens(login_tokens, last_refresh=login_timestamp)

    after_login = json.loads((canonical_profile_env["root"] / "auth.json").read_text())
    login_entry = after_login["credential_pool"]["openai-codex"][0]
    assert login_entry["access_token"] == "SYNTHETIC_LOGIN_ACCESS"
    assert login_entry["refresh_token"] == "SYNTHETIC_LOGIN_REFRESH"
    assert login_entry["expires_at"] == "2099-06-01T00:00:00Z"
    assert login_entry["expires_at_ms"] == 4082918400000
    assert login_entry["last_status"] is None

    refresh_entry = _codex_entry(
        access_token="SYNTHETIC_REFRESH_ACCESS",
        refresh_token="SYNTHETIC_REFRESH_REFRESH",
        last_refresh=(login_entry.get("last_refresh") if mode == "equal" else None),
        expires_at="2099-07-01T00:00:00Z",
        expires_at_ms=4085510400000,
        last_status=None,
    )
    if mode == "missing":
        refresh_entry.pop("last_refresh", None)
    write_credential_pool(
        "openai-codex",
        [refresh_entry],
        replaced_ids=[CODEX_ID],
    )
    after_refresh = json.loads((canonical_profile_env["root"] / "auth.json").read_text())
    result = after_refresh["credential_pool"]["openai-codex"][0]
    assert result["access_token"] == "SYNTHETIC_REFRESH_ACCESS"
    assert result["refresh_token"] == "SYNTHETIC_REFRESH_REFRESH"
    assert result["expires_at_ms"] == 4085510400000


def _terminal_worker(profile: str, entered, release, result_pipe) -> None:
    import os

    os.environ["HERMES_HOME"] = profile
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("openai-codex")
        entry = pool.select()
        result = pool._refresh_entry(entry, force=True) if entry else None
        result_pipe.send({"ok": True, "result": result is None})
    except BaseException as exc:  # pragma: no cover - serialized to parent
        result_pipe.send({"ok": False, "error": repr(exc)})
    finally:
        result_pipe.close()


def _stale_writer_worker(profile: str, result_pipe) -> None:
    import os

    os.environ["HERMES_HOME"] = profile
    try:
        from hermes_cli.auth import read_credential_pool, write_credential_pool

        stale = read_credential_pool("openai-codex")
        write_credential_pool("openai-codex", stale)
        result_pipe.send({"ok": True})
    except BaseException as exc:  # pragma: no cover - serialized to parent
        result_pipe.send({"ok": False, "error": repr(exc)})
    finally:
        result_pipe.close()


def _recv_process_result(pipe, process) -> dict:
    assert pipe.poll(10), f"child did not return before timeout (exit={process.exitcode})"
    result = pipe.recv()
    process.join(10)
    assert process.exitcode == 0
    return result


def test_actual_concurrent_singleton_pool_refresh_serializes_each_generation(
    canonical_profile_env, monkeypatch
):
    """Two profile-loaded pools contend on one synthetic canonical generation."""
    from concurrent.futures import ThreadPoolExecutor

    import hermes_cli.auth as auth_mod
    from agent.credential_pool import load_pool

    monkeypatch.setenv("HERMES_HOME", str(canonical_profile_env["vector"]))
    vector_pool = load_pool("openai-codex")
    monkeypatch.setenv("HERMES_HOME", str(canonical_profile_env["orion"]))
    orion_pool = load_pool("openai-codex")
    vector_entry = vector_pool.select()
    orion_entry = orion_pool.select()
    assert vector_entry and orion_entry

    calls = []
    calls_lock = threading.Lock()

    def fake_refresh(access_token, refresh_token):
        with calls_lock:
            generation = len(calls)
            calls.append((access_token, refresh_token))
        time.sleep(0.15)
        return {
            "access_token": f"SYNTHETIC_ROTATED_ACCESS_{generation}",
            "refresh_token": f"SYNTHETIC_ROTATED_REFRESH_{generation}",
            "last_refresh": f"SYNTHETIC_GENERATION_{generation + 1}",
            "expires_at": "2099-08-01T00:00:00Z",
            "expires_at_ms": 4088198400000,
        }

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_refresh)
    start = threading.Barrier(2)

    def refresh(pool, entry):
        start.wait(5)
        return pool._refresh_entry(entry, force=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                refresh,
                (vector_pool, orion_pool),
                (vector_entry, orion_entry),
            )
        )

    assert calls == [(ROOT_ACCESS, ROOT_REFRESH)]
    assert all(result and result.access_token == "SYNTHETIC_ROTATED_ACCESS_0" for result in results)
    persisted = json.loads((canonical_profile_env["root"] / "auth.json").read_text())
    provider = persisted["providers"]["openai-codex"]
    pool_entry = persisted["credential_pool"]["openai-codex"][0]
    assert provider["tokens"]["access_token"] == "SYNTHETIC_ROTATED_ACCESS_0"
    assert provider["tokens"]["refresh_token"] == "SYNTHETIC_ROTATED_REFRESH_0"
    assert pool_entry["access_token"] == "SYNTHETIC_ROTATED_ACCESS_0"
    assert pool_entry["refresh_token"] == "SYNTHETIC_ROTATED_REFRESH_0"
    assert not (canonical_profile_env["vector"] / "auth.json").exists()
    assert not (canonical_profile_env["orion"] / "auth.json").exists()
    assert persisted["credential_pool"]["openrouter"][0]["access_token"] == (
        "SYNTHETIC_ROOT_OTHER_PROVIDER"
    )


def test_actual_terminal_quarantine_blocks_stale_writer_and_preserves_other_provider(
    canonical_profile_env, monkeypatch
):
    """Terminal refresh clears root state and a waiting stale writer cannot resurrect it."""
    import hermes_cli.auth as auth_mod

    ctx = multiprocessing.get_context("fork")
    entered = ctx.Event()
    release = ctx.Event()

    def fake_terminal_refresh(access_token, refresh_token):
        entered.set()
        assert release.wait(10)
        raise auth_mod.AuthError(
            "synthetic invalid grant",
            provider="openai-codex",
            code="invalid_grant",
            relogin_required=True,
        )

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake_terminal_refresh)
    terminal_parent, terminal_child = ctx.Pipe(False)
    writer_parent, writer_child = ctx.Pipe(False)
    terminal = ctx.Process(
        target=_terminal_worker,
        args=(str(canonical_profile_env["vector"]), entered, release, terminal_child),
    )
    writer = ctx.Process(
        target=_stale_writer_worker,
        args=(str(canonical_profile_env["orion"]), writer_child),
    )
    terminal.start()
    assert entered.wait(10), "terminal refresh did not reach simulated provider"
    writer.start()
    time.sleep(0.1)
    release.set()

    terminal_result = _recv_process_result(terminal_parent, terminal)
    writer_result = _recv_process_result(writer_parent, writer)
    assert terminal_result == {"ok": True, "result": True}
    assert writer_result["ok"] is True, writer_result

    persisted = json.loads((canonical_profile_env["root"] / "auth.json").read_text())
    assert persisted["providers"]["openai-codex"]["tokens"] == {}
    assert persisted["providers"]["openai-codex"]["last_auth_error"]["code"] == (
        "invalid_grant"
    )
    assert persisted["credential_pool"].get("openai-codex", []) == []
    assert persisted["credential_pool"]["openrouter"][0]["access_token"] == (
        "SYNTHETIC_ROOT_OTHER_PROVIDER"
    )
    assert not (canonical_profile_env["vector"] / "auth.json").exists()
    assert not (canonical_profile_env["orion"] / "auth.json").exists()


def test_actual_profile_create_clone_import_keep_codex_canonical(
    tmp_path, monkeypatch
):
    """Lifecycle APIs create isolated homes without local Codex shadow state."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_ROOT", raising=False)
    root = tmp_path / ".hermes"
    root.mkdir()
    _write(root / "auth.json", _canonical_store())

    from hermes_cli.auth import read_credential_pool
    from hermes_cli.profiles import (
        create_profile,
        export_profile,
        get_profile_dir,
        import_profile,
    )

    vector = create_profile("vector", no_alias=True, no_skills=True)
    (vector / "config.yaml").write_text("model:\n  provider: openai-codex\n")
    _write(
        vector / "auth.json",
        {
            "version": 1,
            "credential_pool": {
                "openai-codex": [_codex_entry(
                    access_token="SYNTHETIC_LOCAL_STALE_ACCESS",
                    refresh_token="SYNTHETIC_LOCAL_STALE_REFRESH",
                )],
                "openrouter": [{
                    "id": "SYNTHETIC_LOCAL_OTHER",
                    "auth_type": "api_key",
                    "source": "manual",
                    "access_token": "SYNTHETIC_LOCAL_OTHER_PROVIDER",
                }],
            },
        },
    )
    clone = create_profile(
        "clone",
        clone_from="vector",
        clone_config=True,
        no_alias=True,
    )
    archive = export_profile("vector", str(tmp_path / "vector-export.tar.gz"))
    imported = import_profile(str(archive), name="imported")

    assert vector == get_profile_dir("vector")
    assert clone == get_profile_dir("clone")
    assert imported == get_profile_dir("imported")
    assert (clone / "config.yaml").read_text().startswith("model:")
    assert not (clone / "auth.json").exists()
    assert not (imported / "auth.json").exists()

    for profile in (vector, clone, imported):
        monkeypatch.setenv("HERMES_HOME", str(profile))
        codex = read_credential_pool("openai-codex")
        all_pools = read_credential_pool()
        assert [entry["access_token"] for entry in codex] == [ROOT_ACCESS]
        assert [entry["access_token"] for entry in all_pools["openai-codex"]] == [ROOT_ACCESS]
        if profile == vector:
            assert all_pools["openrouter"][0]["access_token"] == (
                "SYNTHETIC_LOCAL_OTHER_PROVIDER"
            )
        else:
            assert all_pools["openrouter"][0]["access_token"] == (
                "SYNTHETIC_ROOT_OTHER_PROVIDER"
            )
