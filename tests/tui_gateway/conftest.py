"""Shared fixtures for tests/tui_gateway kanban notify tests.

The kanban notify poller test creates tasks with a synthetic ``worker``
assignee.  Kanban ingress is strict since the assignee-validation
contract, so register the label explicitly as a resolver target without
weakening production validation.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _register_synthetic_assignee_lanes(monkeypatch):
    """Register kanban test-only labels as explicit resolver targets."""
    from hermes_cli import profiles

    synthetic = {"worker"}
    real_exists = profiles.profile_exists
    monkeypatch.setattr(
        profiles,
        "profile_exists",
        lambda name: str(name).strip().casefold() in synthetic or real_exists(name),
    )
