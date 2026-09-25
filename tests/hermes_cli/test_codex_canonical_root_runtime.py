import base64
import json
import os
import time

import pytest

import hermes_constants
from hermes_cli.auth import AuthError, read_credential_pool, write_credential_pool
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


@pytest.mark.skipif(os.name == "nt", reason="Mission Control reasoning bridge uses Unix symlinks")
@pytest.mark.parametrize("second_usable", [True, False])
def test_mc_bridge_skips_unrefreshable_first_grant(tmp_path, monkeypatch, second_usable):
    root = tmp_path / "canonical" / ".hermes"
    profile = root / "profiles" / "orion"
    profile.mkdir(parents=True)
    bridge_home = tmp_path / "mission-control-hermes-bridge-test" / "profiles" / "orion"
    bridge_home.mkdir(parents=True)
    (profile / "auth.json").write_text('{"providers": {}}', encoding="utf-8")
    (bridge_home / "auth.json").symlink_to(profile / "auth.json")

    def token(exp):
        payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).rstrip(b"=").decode()
        return f"h.{payload}.s"

    expired = token(int(time.time()) - 60)
    entries: list[dict[str, object]] = [{"access_token": expired, "last_status": "ok"}]
    if second_usable:
        entries.append({"access_token": token(int(time.time()) + 3600), "last_status": "ok"})
    else:
        entries.append({"access_token": "in-cooldown", "last_error_reset_at": time.time() + 3600})
    _store(root / "auth.json", {"openai-codex": entries})
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("HERMES_HOME", str(bridge_home))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))
    before = (root / "auth.json").read_text(encoding="utf-8")
    if second_usable:
        assert resolve_codex_runtime_credentials()["api_key"] == entries[1]["access_token"]
    else:
        with pytest.raises(AuthError) as exc:
            resolve_codex_runtime_credentials()
        assert exc.value.code == "codex_auth_missing_refresh_token"
    assert (root / "auth.json").read_text(encoding="utf-8") == before


def test_pool_refresh_failure_tries_next_grant_without_changing_cooldown(tmp_path, monkeypatch):
    root = tmp_path / "canonical" / ".hermes"
    root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "bridge" / "profiles" / "orion"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex-cli"))
    expired = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) - 60}).encode()).rstrip(b"=").decode()
    fresh = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 3600}).encode()).rstrip(b"=").decode()
    entries = [
        {"access_token": f"h.{expired}.s", "refresh_token": "broken-refresh", "last_status": "ok"},
        {"access_token": f"h.{fresh}.s", "last_status": "ok"},
    ]
    _store(root / "auth.json", {"openai-codex": entries})

    def failed_refresh(*args, **kwargs):
        raise AuthError("invalid grant", provider="openai-codex", code="codex_refresh_failed",
                        relogin_required=True)

    monkeypatch.setattr("hermes_cli.auth.refresh_codex_oauth_pure", failed_refresh)
    before = (root / "auth.json").read_text(encoding="utf-8")
    assert resolve_codex_runtime_credentials()["api_key"] == entries[1]["access_token"]
    assert (root / "auth.json").read_text(encoding="utf-8") == before
