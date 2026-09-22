"""A config write from an MC chat child must not silently no-op.

Mission Control spawns chat children with ``HERMES_HOME`` = a per-chat temp
bridge dir and ``HERMES_ROOT`` = the canonical install (``api.rs``
``configure_hermes_chat_environment``: "Keep bridge config/plugins isolated in
HERMES_HOME while anchoring"). That isolation is intentional, but ``hermes
config set`` from such a child writes the throwaway bridge config and reports
success — so the write path warns instead of letting a user-approved change
vanish. Reads stay silent (the bridge view is the runtime's live view).
"""

import pytest

from hermes_cli import config


def _written(capsys):
    config._warn_if_bridge_config_write()
    return capsys.readouterr().out


def test_warns_for_bridge_home_under_tempdir(monkeypatch, tmp_path, capsys):
    bridge = tmp_path / "mission-control-hermes-bridge-abc" / "profiles" / "default"
    bridge.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(bridge))
    monkeypatch.setenv("HERMES_ROOT", "/home/solo/.hermes")
    monkeypatch.setattr(config.tempfile, "gettempdir", lambda: str(tmp_path))
    out = _written(capsys)
    assert "throwaway bridge config" in out
    assert "HERMES_HOME=/home/solo/.hermes hermes config set" in out


@pytest.mark.parametrize("home,root", [
    ("/home/solo/.hermes", "/home/solo/.hermes"),                      # default home: same path
    ("/home/solo/.hermes/profiles/forge", "/home/solo/.hermes"),       # named profile: real write
])
def test_silent_for_real_homes(monkeypatch, capsys, home, root):
    monkeypatch.setenv("HERMES_HOME", home)
    monkeypatch.setenv("HERMES_ROOT", root)
    assert _written(capsys) == ""


@pytest.mark.parametrize("env", [
    {},                                                                # plain CLI: no anchoring env
    {"HERMES_HOME": "/tmp/x"},                                         # HERMES_ROOT absent
    {"HERMES_ROOT": "/home/solo/.hermes"},                             # HERMES_HOME absent
])
def test_silent_without_both_anchors(monkeypatch, capsys, env):
    for name in ("HERMES_HOME", "HERMES_ROOT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert _written(capsys) == ""
