"""Config-validation + `hermes fallback list` surfacing of unreachable rungs.

Unit coverage for hermes_cli.fallback_diagnostics lives in
test_fallback_diagnostics.py; this file locks the operator-facing surfaces
(config validation output and the fallback list command) to the same
machine-readable reason codes.
"""

from pathlib import Path

import pytest
import yaml

from hermes_cli.config import print_config_warnings, validate_config_structure
from hermes_cli.fallback_cmd import cmd_fallback_list


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _write_config(home: Path, data: dict) -> None:
    (home / ".hermes" / "config.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8")


class TestStartupConfigWarnings:
    """print_config_warnings is the config-validation surface operators see at
    startup (CLI + gateway). Rung diagnostics must ride there, and
    validate_config_structure must stay purely structural."""

    def test_dead_rung_produces_warning_with_reason(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "fallback_providers": [{"provider": "moa", "model": "default"}],
        })
        print_config_warnings(None)
        err = capsys.readouterr().err
        assert "fallback_providers[0]" in err
        assert "virtual_moa_without_preset" in err

    def test_healthy_chain_produces_no_output(self, isolated_home, capsys, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-0001")
        _write_config(isolated_home, {
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        print_config_warnings(None)
        assert capsys.readouterr().err == ""

    def test_structural_validation_does_not_probe_credentials(self, isolated_home):
        """validate_config_structure stays structural: no rung diagnostics and
        no credential I/O, so hot callers (model switch, auth) keep its cost."""
        issues = validate_config_structure({
            "fallback_providers": [{"provider": "moa", "model": "default"}],
        })
        assert not [i for i in issues if "fallback_providers[" in i.message]

    def test_structural_issues_still_reported(self, isolated_home):
        issues = validate_config_structure({"custom_providers": {"name": "x"}})
        assert any(i.severity == "error" for i in issues)

    def test_diagnostics_never_break_startup(self, isolated_home, capsys):
        """Hostile chain shapes must not raise out of the startup writer.

        Malformed entries are dropped upstream by get_fallback_chain; the
        surviving unreachable rung must still be reported and nothing may
        raise.
        """
        _write_config(isolated_home, {
            "fallback_providers": [
                "not-a-dict",
                {"provider": "", "model": ""},
                {"provider": "moa"},                       # dropped: no model
                {"provider": "moa", "model": "default"},   # kept: unreachable
            ],
        })
        print_config_warnings(None)  # must not raise
        err = capsys.readouterr().err
        assert "fallback_providers" in err
        assert "virtual_moa_without_preset" in err


class TestFallbackListSurfaces:
    def test_list_prints_warning_for_unreachable_rung(self, isolated_home, capsys):
        _write_config(isolated_home, {
            "fallback_providers": [
                {"provider": "zai", "model": "glm-4.7"},
                {"provider": "moa", "model": "default"},
            ],
        })
        cmd_fallback_list(None)
        out = capsys.readouterr().out
        assert "moa" in out
        assert "virtual_moa_without_preset" in out or "no_credential_in_scope" in out

    def test_list_healthy_chain_has_no_warning(self, isolated_home, capsys, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-0001")
        _write_config(isolated_home, {
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        cmd_fallback_list(None)
        out = capsys.readouterr().out
        assert "⚠" not in out
