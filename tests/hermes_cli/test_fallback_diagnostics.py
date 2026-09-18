"""Fallback rung resolvability diagnostics (hermes_cli.fallback_diagnostics).

Covers the dead-rung defect class from the host-wide fallback audit: a
``fallback_providers`` rung that can never serve any turn must produce a
clear config-time warning naming the rung + machine-readable reason, while
a healthy chain produces none. Presence checks only — no credential values.
"""

import json
import time
from pathlib import Path

import pytest

from hermes_cli.fallback_diagnostics import (
    REASON_CREDENTIAL_IN_QUOTA_COOLDOWN,
    REASON_NO_CREDENTIAL_IN_SCOPE,
    REASON_RELOGIN_REQUIRED,
    REASON_VIRTUAL_MOA_WITHOUT_PRESET,
    diagnose_fallback_chain,
    format_rung_issue,
)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect HOME/HERMES_HOME into a per-test sandbox (same shape as
    test_fallback_cmd's local fixture — kept per-file by repo convention)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Blank provider credentials so scope checks are deterministic."""
    for name in (
        "OPENROUTER_API_KEY", "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
        "DEEPSEEK_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def zai_key(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key-zai-0001")


# ── Healthy chains ──────────────────────────────────────────────────────────


class TestHealthyChainSilent:
    """A healthy chain must produce ZERO warnings (constraint: no new noise)."""

    def test_env_keyed_provider_resolves(self, zai_key, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        assert issues == []

    def test_empty_chain(self, isolated_home):
        assert diagnose_fallback_chain({}) == []

    def test_missing_chain(self, isolated_home):
        assert diagnose_fallback_chain({"model": {"provider": "zai"}}) == []

    def test_inline_api_key_entry(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{
                "provider": "custom-x",
                "model": "m",
                "base_url": "https://api.example.com/v1",
                "api_key": "inline-key-0001",
            }],
        })
        assert issues == []

    def test_key_env_entry(self, monkeypatch, isolated_home):
        monkeypatch.setenv("MY_LANE_KEY", "key-0001")
        issues = diagnose_fallback_chain({
            "fallback_providers": [{
                "provider": "custom-x",
                "model": "m",
                "base_url": "https://api.example.com/v1",
                "key_env": "MY_LANE_KEY",
            }],
        })
        assert issues == []

    def test_anonymous_local_endpoint_needs_no_key(self, isolated_home):
        """base_url-pinned endpoint on a provider with no declared key surface."""
        issues = diagnose_fallback_chain({
            "fallback_providers": [{
                "provider": "my-local-gateway",
                "model": "m",
                "base_url": "http://127.0.0.1:8080/v1",
            }],
        })
        assert issues == []


# ── no_credential_in_scope ──────────────────────────────────────────────────


class TestNoCredentialInScope:
    def test_env_keyed_provider_without_key(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        assert len(issues) == 1
        assert issues[0].rung_index == 0
        assert issues[0].provider == "zai"
        assert issues[0].reason == REASON_NO_CREDENTIAL_IN_SCOPE

    def test_issue_names_env_hint_without_leaking_values(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        rendered = format_rung_issue(issues[0])
        assert "GLM_API_KEY" in rendered
        assert "test-key" not in rendered

    def test_legacy_fallback_model_dict_also_checked(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_model": {"provider": "zai", "model": "glm-4.7"},
        })
        assert len(issues) == 1
        assert issues[0].reason == REASON_NO_CREDENTIAL_IN_SCOPE


# ── credential_in_quota_cooldown / relogin_required ─────────────────────────


def _seed_pool(tmp_path, provider, entry):
    auth = {"version": 1, "credential_pool": {provider: [entry]}}
    (tmp_path / "auth.json").write_text(json.dumps(auth), encoding="utf-8")


@pytest.fixture
def pool_home(isolated_home):
    """isolated_home yields the tmp Path — reused for pool seeding.

    auth.json lives under ``$HERMES_HOME`` (the fixture's ``.hermes`` dir),
    which is where ``_auth_file_path()`` resolves under the sandbox.
    """
    return isolated_home / ".hermes"


class TestPoolStates:
    def _entry(self, now, **overrides):
        entry = {
            "id": "openrouter:0",
            "label": "main",
            "provider": "openrouter",
            "auth_type": "api_key",
            "source": "manual",
            "access_token": "sk-or-test-000000000001",
            "last_status": "exhausted",
            "last_status_at": now - 60,
            "last_error_code": 429,
            "last_error_reason": "rate_limit_exceeded",
            "last_error_reset_at": now + 3600,
        }
        entry.update(overrides)
        return entry

    def test_exhausted_pool_flags_cooldown(self, pool_home, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _seed_pool(pool_home, "openrouter", self._entry(time.time()))
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "openrouter", "model": "m"}],
        })
        assert len(issues) == 1
        assert issues[0].reason == REASON_CREDENTIAL_IN_QUOTA_COOLDOWN

    def test_expired_cooldown_is_healthy(self, pool_home, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _seed_pool(pool_home, "openrouter", self._entry(time.time(), last_error_reset_at=time.time() - 10))
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "openrouter", "model": "m"}],
        })
        assert issues == []

    def test_dead_pool_flags_relogin(self, pool_home, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        _seed_pool(pool_home, "openrouter", self._entry(
            time.time(), last_status="dead", last_error_reason="token_revoked"))
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "openrouter", "model": "m"}],
        })
        assert len(issues) == 1
        assert issues[0].reason == REASON_RELOGIN_REQUIRED

    def test_pool_exists_but_env_key_present_is_healthy(self, pool_home, monkeypatch):
        """An env key rescues a rung even when the pool is exhausted."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-rescue-0001")
        _seed_pool(pool_home, "openrouter", self._entry(time.time()))
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "openrouter", "model": "m"}],
        })
        assert issues == []


# ── virtual_moa_without_preset ──────────────────────────────────────────────


class TestVirtualMoaRung:
    def test_bare_moa_rung_with_default_preset_and_no_openrouter_key(
        self, isolated_home,
    ):
        """The audit's exact shape: no moa: config, no OPENROUTER_API_KEY.

        The built-in default preset pins the aggregator to openrouter, so the
        rung is unreachable and must say so via the moa reason.
        """
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "moa", "model": "default"}],
        })
        assert len(issues) == 1
        assert issues[0].reason == REASON_VIRTUAL_MOA_WITHOUT_PRESET
        assert issues[0].provider == "moa"
        assert "openrouter" in issues[0].detail

    def test_moa_rung_with_healthy_aggregator_is_clean(self, zai_key, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "moa", "model": "custom-preset"}],
            "moa": {"presets": {
                "custom-preset": {"aggregator": {"provider": "zai", "model": "glm-4.7"}},
            }},
        })
        assert issues == []

    def test_moa_rung_with_unreachable_aggregator_names_it(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "moa", "model": "custom-preset"}],
            "moa": {"presets": {
                "custom-preset": {"aggregator": {"provider": "zai", "model": "glm-4.7"}},
            }},
        })
        assert len(issues) == 1
        assert issues[0].reason == REASON_NO_CREDENTIAL_IN_SCOPE
        assert "zai" in issues[0].detail

    def test_healthy_ladder_only_flags_dead_rung(self, zai_key, isolated_home):
        """The live audit repro shape: zai healthy, moa/default dead."""
        issues = diagnose_fallback_chain({
            "fallback_providers": [
                {"provider": "zai", "model": "glm-4.7"},
                {"provider": "moa", "model": "default"},
            ],
        })
        assert len(issues) == 1
        assert issues[0].rung_index == 1
        assert issues[0].provider == "moa"


# ── rendering ───────────────────────────────────────────────────────────────


class TestRendering:
    def test_format_names_rung_index_provider_reason(self, isolated_home):
        issues = diagnose_fallback_chain({
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        rendered = format_rung_issue(issues[0])
        assert "fallback[0]" in rendered
        assert "zai" in rendered
        assert REASON_NO_CREDENTIAL_IN_SCOPE in rendered
