"""Config-validation + `hermes fallback list` surfacing of unreachable rungs.

Unit coverage for hermes_cli.fallback_diagnostics lives in
test_fallback_diagnostics.py; this file locks the operator-facing surfaces
(config validation output and the fallback list command) to the same
machine-readable reason codes.
"""

from pathlib import Path

import pytest
import yaml

from hermes_cli.config import validate_config_structure
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


class TestValidateConfigStructureSurfaces:
    def test_dead_rung_produces_warning_with_reason(self, isolated_home):
        issues = validate_config_structure({
            "fallback_providers": [{"provider": "moa", "model": "default"}],
        })
        rung_warnings = [
            i for i in issues
            if i.severity == "warning" and "fallback_providers[0]" in i.message
        ]
        assert len(rung_warnings) == 1
        assert "virtual_moa_without_preset" in rung_warnings[0].message

    def test_healthy_chain_produces_no_rung_warning(self, isolated_home, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-0001")
        issues = validate_config_structure({
            "fallback_providers": [{"provider": "zai", "model": "glm-4.7"}],
        })
        assert not [i for i in issues if "fallback_providers[" in i.message]

    def test_structural_issues_still_reported(self, isolated_home):
        issues = validate_config_structure({"custom_providers": {"name": "x"}})
        assert any(i.severity == "error" for i in issues)

    def test_diagnostics_never_break_validation(self, isolated_home, monkeypatch):
        """A hostile chain shape must not raise out of config validation."""
        issues = validate_config_structure({
            "fallback_providers": [
                "not-a-dict",
                {"provider": "", "model": ""},
                {"provider": "moa"},
            ],
        })
        assert isinstance(issues, list)


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
