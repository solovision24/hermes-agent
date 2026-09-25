import json
import os

import pytest

import hermes_constants
from hermes_cli.auth import read_credential_pool, write_credential_pool
from hermes_cli.auth_codex import resolve_codex_runtime_credentials
from hermes_constants import get_default_hermes_root


def _store(path, pool):
    path.write_text(json.dumps({"credential_pool": pool}), encoding="utf-8")


def test_codex_profile_shadow_is_ignored_and_writes_use_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    profile = root / "profiles" / "vector"
    profile.mkdir(parents=True)
    _store(root / "auth.json", {"openai-codex": [{"id": "canonical"}], "openrouter": [{"id": "other"}]})
    _store(profile / "auth.json", {"openai-codex": [{"id": "stale-local"}], "openrouter": [{"id": "local-other"}]})
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    assert get_default_hermes_root() == root
    assert [row["id"] for row in read_credential_pool("openai-codex")] == ["canonical"]
    assert [row["id"] for row in read_credential_pool()["openai-codex"]] == ["canonical"]

    write_credential_pool("openai-codex", [{"id": "rotated"}])
    root_rows = json.loads((root / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openai-codex"]
    assert {row["id"] for row in root_rows} == {"rotated", "canonical"}
    assert json.loads((profile / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openai-codex"] == [{"id": "stale-local"}]
    assert json.loads((root / "auth.json").read_text(encoding="utf-8"))["credential_pool"]["openrouter"] == [{"id": "other"}]


@pytest.mark.skipif(os.name == "nt", reason="Mission Control reasoning bridge uses Unix symlinks")
def test_mc_temporary_bridge_profile_resolves_canonical_codex_pool(tmp_path, monkeypatch):
    """MC sets HERMES_HOME to a temporary overlay, not the canonical profile path."""
    root = tmp_path / "canonical" / ".hermes"
    profile = root / "profiles" / "orion"
    profile.mkdir(parents=True)
    _store(root / "auth.json", {"openai-codex": [{"access_token": "canonical-access"}]})
    (profile / "auth.json").write_text(json.dumps({"providers": {}}), encoding="utf-8")
    bridge_home = tmp_path / "mission-control-hermes-bridge-test" / "profiles" / "orion"
    bridge_home.mkdir(parents=True)
    (bridge_home / "auth.json").symlink_to(profile / "auth.json")
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: root)
    monkeypatch.setenv("HERMES_HOME", str(bridge_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))

    # An unpinned bridge infers its empty temporary root: the original failure.
    monkeypatch.delenv("HERMES_ROOT", raising=False)
    assert get_default_hermes_root() == bridge_home.parent.parent
    assert not read_credential_pool("openai-codex")

    # MC pins HERMES_ROOT to the canonical install for the child process.
    monkeypatch.setenv("HERMES_ROOT", str(root))
    assert get_default_hermes_root() == root
    assert read_credential_pool("openai-codex")
    resolved = resolve_codex_runtime_credentials()
    assert resolved["source"] == "credential_pool"
    assert resolved["api_key"] == "canonical-access"
    assert json.loads((profile / "auth.json").read_text(encoding="utf-8")) == {"providers": {}}
