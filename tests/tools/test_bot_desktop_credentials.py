"""Metadata-only readiness must never activate/unlock/read a credential provider."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from tools.bot_desktop import credentials

BUS = "unix:path=/private/profile-bus,guid=012345"
COLLECTION = "/org/freedesktop/secrets/collection/test"


def replies(monkeypatch, values):
    calls = []
    results = iter(values)
    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(stdout=json.dumps({"data": result}))
    monkeypatch.setattr(credentials.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("locked,state", [(True, "locked"), (False, "unlocked")])
def test_collection_metadata_only(monkeypatch, locked, state):
    calls = replies(monkeypatch, [[True], [COLLECTION], locked])
    result = credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS, "SECRET": "must-not-inherit"})
    assert result["state"] == state
    assert len(calls) == 3
    for argv, kwargs in calls:
        assert "--address=" + BUS in argv
        assert "--auto-start=no" in argv
        assert 0 < kwargs["timeout"] <= .8
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert not {"SECRET", "DBUS_SESSION_BUS_ADDRESS", "HOME"} & kwargs["env"].keys()
        assert not {"Unlock", "Prompt", "GetSecret", "GetSecrets", "SearchItems", "CreateCollection"} & set(argv)
    assert BUS not in json.dumps(result)


@pytest.mark.parametrize("address", ["tcp:host=example.com", "unix:path=/a;unix:path=/b", "unix:path=/a\n", "unix:abstract=ambient"])
def test_unsafe_bus_address_never_probed(monkeypatch, address):
    calls = replies(monkeypatch, [])
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": address})["state"] == "unknown"
    assert not calls


def test_missing_bus_never_falls_back_to_ambient(monkeypatch):
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", BUS)
    calls = replies(monkeypatch, [])
    assert credentials.probe({})["state"] == "unavailable"
    assert not calls


@pytest.mark.parametrize("available,state", [([["org.freedesktop.secrets"]], "inactive"), ([[]], "unavailable"), (["malformed"], "unknown")])
def test_inactive_provider_is_not_activated(monkeypatch, available, state):
    calls = replies(monkeypatch, [[False], available])
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})["state"] == state
    assert len(calls) == 2
    assert all("org.freedesktop.secrets" not in argv[5:8] for argv, _ in calls)


def test_setup_required_preserves_existing_collections(monkeypatch):
    calls = replies(monkeypatch, [[True], ["/"]])
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})["state"] == "setup_required"
    assert len(calls) == 2


@pytest.mark.parametrize("value", [[], ["/unrelated"], [COLLECTION, COLLECTION], [12]])
def test_alias_malformed_is_unknown(monkeypatch, value):
    replies(monkeypatch, [[True], value])
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})["state"] == "unknown"


@pytest.mark.parametrize("exc", [FileNotFoundError("private-path"), subprocess.TimeoutExpired("private-command", .8), subprocess.CalledProcessError(1, "private-command"), ValueError("private-response")])
def test_probe_failure_never_leaks_error_or_claims_ready(monkeypatch, exc):
    replies(monkeypatch, [exc])
    result = credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})
    assert result["state"] == "unknown"
    assert "private" not in json.dumps(result)


def test_total_deadline_not_per_command(monkeypatch):
    calls = replies(monkeypatch, [[True]])
    times = iter([1., 1.1, 2.])
    monkeypatch.setattr(credentials.time, "monotonic", lambda: next(times))
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})["state"] == "unknown"
    assert len(calls) == 1


def test_default_status_contract_does_not_imply_encryption():
    result = credentials._result("unlocked")
    assert "not verified" in result["message"]


@pytest.mark.parametrize("owned", [[1], ["true"], [], True])
def test_malformed_owner_does_not_claim_readiness(monkeypatch, owned):
    replies(monkeypatch, [owned])
    assert credentials.probe({"DBUS_SESSION_BUS_ADDRESS": BUS})["state"] == "unknown"


def test_sandbox_status_never_probes_gateway_credential_bus(monkeypatch):
    from tools.bot_desktop import runtime, placement
    result = object()
    monkeypatch.setattr(placement, "resolve", lambda: SimpleNamespace(where=placement.TERMINAL))
    monkeypatch.setattr(runtime, "_sandbox_status", lambda *_: result)
    monkeypatch.setattr(credentials, "probe", lambda _: pytest.fail("must probe at the remote target"))
    assert runtime.status() is result


def test_runtime_routes_a_b_a_to_each_published_bus(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import browser, placement, resources, runtime, sandbox_host
    homes = [tmp_path/"a", tmp_path/"b"]
    for home, bus in zip(homes, ["unix:path=/a", "unix:path=/b"]):
        (home/"bot-desktop").mkdir(parents=True)
        (home/"bot-desktop/env").write_text("DISPLAY=:121\nDBUS_SESSION_BUS_ADDRESS=" + bus + "\n")
    monkeypatch.setattr(placement, "resolve", lambda: SimpleNamespace(where=placement.GATEWAY))
    monkeypatch.setattr(sandbox_host, "_read_marker", lambda: None)
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: 123)
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "missing_binaries", lambda: [])
    monkeypatch.setattr(runtime, "geometry", lambda: "800x600")
    monkeypatch.setattr(resources, "memory_info", lambda: resources.MemoryInfo(None, None))
    monkeypatch.setattr(browser, "executable", lambda: None)
    seen = []
    def probe(env):
        seen.append(env["DBUS_SESSION_BUS_ADDRESS"])
        return credentials._result("locked" if env["DBUS_SESSION_BUS_ADDRESS"].endswith('/a') else "unlocked")
    monkeypatch.setattr(credentials, "probe", probe)
    states = []
    for home in [homes[0], homes[1], homes[0]]:
        token = set_hermes_home_override(home)
        try:
            states.append(runtime.status().as_dict()["credential_storage"]["state"])
        finally:
            reset_hermes_home_override(token)
    assert states == ["locked", "unlocked", "locked"]
    assert seen == ["unix:path=/a", "unix:path=/b", "unix:path=/a"]


def test_screen_status_prints_unlock_guidance(monkeypatch, capsys):
    from hermes_cli.subcommands.computer_use_screen import _screen_status
    from tools.bot_desktop import lease, runtime
    st = SimpleNamespace(supported=True, installed=True, running=True, profile="test", display=":121", geometry="800x600", pid=123, socket="private-socket", credential_storage=credentials._result("locked"))
    monkeypatch.setattr(runtime, "status", lambda: st)
    monkeypatch.setattr(lease, "get", lambda: None)
    monkeypatch.setattr(lease, "public_view", lambda _: {"holder":"agent"})
    assert _screen_status(SimpleNamespace(json=False)) == 0
    text = capsys.readouterr().out
    assert "system authentication dialog" in text
    assert "never in chat" in text
