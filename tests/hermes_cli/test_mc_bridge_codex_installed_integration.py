"""MC's temporary profile overlay must still borrow the canonical Codex grant."""
import base64
import json
import time

import pytest

from hermes_constants import get_default_hermes_root
from hermes_cli.auth import AuthError
from hermes_cli.auth_codex import resolve_codex_runtime_credentials


def _jwt(seconds):
    payload = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + seconds}).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    root = tmp_path / "canonical" / ".hermes"
    profile = root / "profiles" / "orion"
    profile.mkdir(parents=True)
    (profile / "auth.json").write_text('{"providers": {}}', encoding="utf-8")
    overlay = tmp_path / "mission-control-hermes-bridge" / "profiles" / "orion"
    overlay.mkdir(parents=True)
    (overlay / "auth.json").symlink_to(profile / "auth.json")
    monkeypatch.setenv("HERMES_HOME", str(overlay))
    monkeypatch.setenv("HERMES_ROOT", str(root))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-external-codex"))
    return root, profile, overlay


def _seed(root, entries):
    path = root / "auth.json"
    path.write_text(json.dumps({"version": 1, "providers": {},
                                "credential_pool": {"openai-codex": entries}}), encoding="utf-8")
    return path


def test_mc_bridge_skips_expired_grant_and_keeps_profile_isolated(bridge):
    root, profile, overlay = bridge
    auth = _seed(root, [{"access_token": _jwt(-60)}, {"access_token": _jwt(3600)}])
    before = auth.read_bytes()
    assert get_default_hermes_root() == root
    assert resolve_codex_runtime_credentials()["api_key"] == json.loads(before)["credential_pool"]["openai-codex"][1]["access_token"]
    assert auth.read_bytes() == before
    assert json.loads((profile / "auth.json").read_text()) == {"providers": {}}


def test_mc_bridge_all_unusable_reports_first_error(bridge):
    root, _, _ = bridge
    auth = _seed(root, [{"access_token": _jwt(-60)},
                        {"access_token": _jwt(3600), "last_error_reset_at": time.time() + 3600}])
    before = auth.read_bytes()
    with pytest.raises(AuthError) as error:
        resolve_codex_runtime_credentials()
    assert error.value.code == "codex_auth_missing_refresh_token"
    assert auth.read_bytes() == before


def test_mc_bridge_rotates_expired_pool_grant_at_canonical_root(bridge, monkeypatch):
    root, profile, _ = bridge
    auth = _seed(root, [{"access_token": _jwt(-60), "refresh_token": "old-refresh"}])
    fresh = _jwt(3600)
    calls = []

    def rotate(access, refresh, *, timeout_seconds):
        calls.append((access, refresh))
        return {"access_token": fresh, "refresh_token": "new-refresh", "last_refresh": "now"}

    monkeypatch.setattr("hermes_cli.auth_codex.refresh_codex_oauth_pure", rotate)
    assert resolve_codex_runtime_credentials()["api_key"] == fresh
    assert len(calls) == 1
    assert (root / "auth.lock").exists()
    stored = json.loads(auth.read_text())["credential_pool"]["openai-codex"][0]
    assert stored["refresh_token"] == "new-refresh"
    assert json.loads((profile / "auth.json").read_text()) == {"providers": {}}
    assert resolve_codex_runtime_credentials()["api_key"] == fresh
    assert len(calls) == 1


def test_mc_bridge_read_only_does_not_refresh_or_write(bridge, monkeypatch):
    root, profile, overlay = bridge
    auth = _seed(root, [{"access_token": _jwt(-60), "refresh_token": "old-refresh"}])
    before = auth.read_bytes()
    profile_before = (profile / "auth.json").read_bytes()
    root_before = set(root.iterdir())
    profile_before_files = set(profile.iterdir())
    overlay_before = set(overlay.iterdir())
    assert not (root / "auth.lock").exists()
    monkeypatch.setattr("hermes_cli.auth_codex.refresh_codex_oauth_pure", lambda *a, **kw: pytest.fail("read-only refresh"))
    assert resolve_codex_runtime_credentials(read_only=True)["source"] == "credential_pool"
    assert auth.read_bytes() == before
    assert (profile / "auth.json").read_bytes() == profile_before
    assert set(root.iterdir()) == root_before
    assert set(profile.iterdir()) == profile_before_files
    assert set(overlay.iterdir()) == overlay_before
