"""Shared fixtures for tests/plugins kanban dashboard tests.

Kanban task creation is strict since the assignee-validation contract:
an unresolved assignee label is rejected before mutation.  These plugin
tests intentionally use short synthetic labels ("alice", "worker", ...)
that are not real on-disk profiles, so register them explicitly as
resolver targets — mirroring ``tests/hermes_cli/conftest.py`` — without
weakening production validation.
"""

from __future__ import annotations

import pytest


_SYNTHETIC_ASSIGNEES = (
    "alice", "elias", "newbie", "ops", "orig", "researcher", "worker", "x",
)


@pytest.fixture(autouse=True)
def _register_synthetic_assignee_lanes(monkeypatch):
    """Register legacy test-only labels as explicit resolver targets."""
    from hermes_cli import profiles

    real_exists = profiles.profile_exists
    monkeypatch.setattr(
        profiles,
        "profile_exists",
        lambda name: str(name).strip().casefold() in _SYNTHETIC_ASSIGNEES
        or real_exists(name),
    )
