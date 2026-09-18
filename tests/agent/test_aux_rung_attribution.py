"""Aux error attribution: the 'No LLM provider configured' failure must name
the fallback rung / MoA runtime that produced it, not present a phantom
(task, provider) pair as a setup error (kanban t_b4d6e4b6, from the audit
card t_529e4ebf section 8)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    _client_rung_provenance,
    _explicit_provider_missing_error,
    _no_provider_configured_error,
    _rung_provenance_label,
    _set_client_rung_provenance,
    _try_main_fallback_chain,
)


class _FakeClient:
    """Minimal client stand-in that accepts attribute stamps."""

    _hermes_rung_index: int = 0
    _hermes_rung_entry: dict = {}


_RUNTIME_GLOBALS = (
    "_RUNTIME_MAIN_PROVIDER",
    "_RUNTIME_MAIN_MODEL",
    "_RUNTIME_MAIN_BASE_URL",
    "_RUNTIME_MAIN_API_KEY",
    "_RUNTIME_MAIN_API_MODE",
    "_RUNTIME_MAIN_AUTH_MODE",
)


@pytest.fixture(autouse=True)
def _restore_aux_runtime_state():
    """Snapshot/restore auxiliary_client's module-level runtime state.

    ``set_runtime_main()`` rewrites both the context-local runtime and the
    legacy compat mirrors + snapshot. Leaking those into other test files
    changes how *they* resolve the main runtime (test_auxiliary_main_first's
    vision tests patch the mirrors directly), so restore everything this
    file may touch.
    """
    saved_ctx = aux._RUNTIME_MAIN_CONTEXT.get()
    saved_snapshot = aux._RUNTIME_MAIN_COMPAT_SNAPSHOT
    saved_vars = {name: getattr(aux, name) for name in _RUNTIME_GLOBALS}
    try:
        yield
    finally:
        aux._RUNTIME_MAIN_CONTEXT.set(saved_ctx)
        aux._RUNTIME_MAIN_COMPAT_SNAPSHOT = saved_snapshot
        for name, value in saved_vars.items():
            setattr(aux, name, value)


# ── provenance stamping helpers ─────────────────────────────────────────────


class TestProvenanceHelpers:
    def test_label_matches_ladder_log_shape(self):
        label = _rung_provenance_label("fallback_providers", 2, {"provider": "moa", "model": "default"})
        assert label == "fallback_providers[2](moa/default)"

    def test_label_without_model(self):
        assert _rung_provenance_label("fallback_providers", 0, {"provider": "zai"}) == "fallback_providers[0](zai)"

    def test_empty_source_is_blank(self):
        assert _rung_provenance_label("", 0, {"provider": "zai"}) == ""

    def test_stamp_and_read_roundtrip(self):
        client = _FakeClient()
        _set_client_rung_provenance(client, "fallback_providers", 1, {"provider": "moa", "model": "default"})
        assert _client_rung_provenance(client) == "fallback_providers[1](moa/default)"
        assert client._hermes_rung_index == 1
        assert client._hermes_rung_entry["provider"] == "moa"

    def test_read_from_plain_client_is_blank(self):
        assert _client_rung_provenance(_FakeClient()) == ""
        assert _client_rung_provenance(None) == ""


# ── error builders ──────────────────────────────────────────────────────────


class TestNoProviderConfiguredError:
    def test_chain_client_names_the_rung_not_setup(self):
        client = _FakeClient()
        _set_client_rung_provenance(client, "fallback_providers", 1, {"provider": "moa", "model": "default"})
        err = _no_provider_configured_error("moa_aggregator", "openrouter", client=client)
        msg = str(err)
        assert "fallback_providers[1](moa/default)" in msg
        assert "Run: hermes setup" not in msg
        assert "did not configure" in msg

    def test_moa_runtime_attribution(self):
        aux.set_runtime_main("moa", "default", requested_provider="moa")
        try:
            err = _no_provider_configured_error("moa_aggregator", "openrouter", main_runtime=None)
            msg = str(err)
            assert "MoA runtime" in msg
            assert "preset 'default'" in msg
            assert "Run: hermes setup" not in msg
        finally:
            aux._RUNTIME_MAIN_CONTEXT.set(None)

    def test_plain_path_keeps_setup_hint(self):
        aux._RUNTIME_MAIN_CONTEXT.set(None)
        err = _no_provider_configured_error("compression", "auto", client=None, main_runtime={})
        assert "Run: hermes setup" in str(err)

    def test_moa_runtime_names_derived_aggregator(self):
        aux.set_runtime_main("moa", "default", requested_provider="moa")
        try:
            err = _no_provider_configured_error("moa_aggregator", "openrouter", main_runtime=None)
            # The derived (unconfigured) provider is named for the operator.
            assert "openrouter" in str(err)
        finally:
            aux._RUNTIME_MAIN_CONTEXT.set(None)


class TestExplicitProviderMissingError:
    def test_plain_path_keeps_env_hint(self):
        aux._RUNTIME_MAIN_CONTEXT.set(None)
        err = _explicit_provider_missing_error("compression", "deepseek", "DEEPSEEK_API_KEY", main_runtime={})
        msg = str(err)
        assert "DEEPSEEK_API_KEY" in msg
        assert "is set in config.yaml" in msg

    def test_moa_runtime_does_not_claim_config_yaml(self):
        """When the provider came from a MoA preset slot, 'is set in
        config.yaml' would be a lie."""
        aux.set_runtime_main("moa", "default", requested_provider="moa")
        try:
            err = _explicit_provider_missing_error("moa_aggregator", "openrouter", "OPENROUTER_API_KEY", main_runtime=None)
            msg = str(err)
            assert "is set in config.yaml" not in msg
            assert "MoA runtime" in msg
        finally:
            aux._RUNTIME_MAIN_CONTEXT.set(None)


# ── chain walkers stamp provenance ──────────────────────────────────────────


class TestChainStamping:
    def test_main_fallback_chain_stamps_resolved_client(self, monkeypatch):
        entry = {"provider": "zai", "model": "glm-4.7"}
        client = _FakeClient()
        monkeypatch.setattr(aux, "_read_main_provider", lambda: "anthropic")
        monkeypatch.setattr(aux, "load_config_readonly", lambda: {
            "fallback_providers": [entry]}, raising=False)
        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda cfg: [entry])
        monkeypatch.setattr(aux, "_is_provider_unhealthy", lambda p: False)
        monkeypatch.setattr(aux, "_resolve_fallback_entry", lambda e: (client, "glm-4.7"))
        monkeypatch.setattr(aux, "_task_minimum_context_length", lambda t: None)

        got, model, provider = _try_main_fallback_chain("compression", "anthropic")
        assert got is client
        assert _client_rung_provenance(got) == "fallback_providers[0](zai/glm-4.7)"


# ── end-to-end: the audit's live failure shape ───────────────────────────────


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Redirect HOME/HERMES_HOME into a per-test sandbox."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


class TestAuditReproShape:
    def test_moa_rung_session_gets_truthful_error(self, monkeypatch, isolated_home):
        """The cron-scheduler repro: session riding moa/default, aggregator
        openrouter has no credentials → error names the real origin."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        aux.set_runtime_main("moa", "default", requested_provider="moa",
                             base_url="moa://local", api_key="moa-virtual-provider")
        try:
            err = aux._no_provider_configured_error(
                "moa_aggregator", "openrouter", client=None, main_runtime=None)
            msg = str(err)
            assert "Run: hermes setup" not in msg
            assert "MoA runtime" in msg
            assert "openrouter" in msg
        finally:
            aux._RUNTIME_MAIN_CONTEXT.set(None)
