"""Canonical Kanban assignee resolution.

Assignees are persisted targets, not free-form labels.  This module is the
single authority for deciding whether an input names an on-disk Hermes
profile, a configured external lane, a configured alias, no assignee, or an
invalid value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


INVALID_ASSIGNEE = "invalid_assignee"


class AssigneeCategory(str, Enum):
    UNASSIGNED = "unassigned"
    PROFILE = "profile"
    EXTERNAL_LANE = "external_lane"
    ALIAS = "alias"
    INVALID = "invalid"


@dataclass(frozen=True)
class AssigneeResolution:
    """The validated meaning of an assignee input."""

    input_value: Any
    category: AssigneeCategory
    canonical: Optional[str]
    target_category: Optional[AssigneeCategory] = None
    diagnostic: Optional[str] = None

    @property
    def spawnable(self) -> bool:
        return self.target_category is AssigneeCategory.PROFILE or self.category is AssigneeCategory.PROFILE


class InvalidAssigneeError(ValueError):
    """Stable, actionable validation failure for unresolved assignees."""

    code = INVALID_ASSIGNEE

    def __init__(self, value: Any, diagnostic: Optional[str] = None):
        self.value = value
        self.diagnostic = diagnostic or invalid_assignee_diagnostic(value)
        self.resolution = AssigneeResolution(value, AssigneeCategory.INVALID, None, diagnostic=self.diagnostic)
        super().__init__(self.diagnostic)


def invalid_assignee_diagnostic(value: Any) -> str:
    return (
        f"{INVALID_ASSIGNEE}: assignee {value!r} does not resolve to an on-disk "
        "Hermes profile or a configured external lane; use an existing profile, "
        "configure kanban.external_lanes, or add kanban.assignee_aliases"
    )


def _load_config() -> dict:
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _external_lanes(config: dict) -> dict[str, str]:
    section = config.get("kanban")
    if not isinstance(section, dict):
        section = {}
    raw = section.get("external_lanes", [])
    entries: list[Any]
    if isinstance(raw, dict):
        entries = list(raw.keys())
    elif isinstance(raw, (list, tuple, set)):
        entries = list(raw)
    else:
        entries = []
    result: dict[str, str] = {}
    for entry in entries:
        if isinstance(entry, dict):
            entry = entry.get("name") or entry.get("id")
        if entry is None:
            continue
        name = str(entry).strip()
        if name:
            result[name.casefold()] = name
    return result


def configured_external_lanes(config: Optional[dict] = None) -> tuple[str, ...]:
    cfg = _load_config() if config is None else config
    return tuple(sorted(set(_external_lanes(cfg).values()), key=str.casefold))


def _assignee_aliases(config: dict) -> dict[str, str]:
    section = config.get("kanban")
    if not isinstance(section, dict):
        section = {}
    raw = section.get("assignee_aliases", {})
    if not isinstance(raw, dict):
        return {}
    return {
        str(alias).strip().casefold(): str(target).strip()
        for alias, target in raw.items()
        if str(alias).strip() and str(target).strip()
    }


def configured_assignee_aliases(config: Optional[dict] = None) -> dict[str, str]:
    cfg = _load_config() if config is None else config
    return _assignee_aliases(cfg)


def has_configured_assignee_targets(config: Optional[dict] = None) -> bool:
    cfg = _load_config() if config is None else config
    return bool(_external_lanes(cfg) or _assignee_aliases(cfg))


def configured_assignee_choices(config: Optional[dict] = None) -> tuple[AssigneeResolution, ...]:
    """Enumerate configured profiles, lanes, and aliases for CLI consumers."""
    cfg = _load_config() if config is None else config
    choices: list[AssigneeResolution] = []
    try:
        from hermes_cli.profiles import list_profiles
        profile_names = [info.name for info in list_profiles()]
    except Exception:
        profile_names = ["default"]
    for name in profile_names:
        try:
            choices.append(resolve_assignee(name, allow_unassigned=False, config=cfg))
        except InvalidAssigneeError:
            continue
    for name in configured_external_lanes(cfg):
        choices.append(resolve_assignee(name, allow_unassigned=False, config=cfg))
    for alias in sorted(configured_assignee_aliases(cfg)):
        try:
            choices.append(resolve_assignee(alias, allow_unassigned=False, config=cfg))
        except InvalidAssigneeError:
            continue
    return tuple(choices)


def _profile_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists, validate_profile_name
        name = normalize_profile_name(str(value))
        validate_profile_name(name)
        return name if profile_exists(name) else None
    except (TypeError, ValueError, OSError):
        return None


def _target(value: str, *, profiles: bool, lanes: dict[str, str]) -> Optional[tuple[AssigneeCategory, str]]:
    if profiles:
        profile = _profile_name(value)
        if profile is not None:
            return AssigneeCategory.PROFILE, profile
    lane = lanes.get(value.casefold())
    return (AssigneeCategory.EXTERNAL_LANE, lane) if lane is not None else None


def resolve_assignee(value: Any, *, allow_unassigned: bool = True, config: Optional[dict] = None) -> AssigneeResolution:
    if value is None or (isinstance(value, str) and value.strip().casefold() in {"", "none", "null", "-", "unassigned"}):
        if allow_unassigned:
            return AssigneeResolution(value, AssigneeCategory.UNASSIGNED, None)
        raise InvalidAssigneeError(value)
    text = str(value).strip()
    cfg = _load_config() if config is None else config
    lanes = _external_lanes(cfg)
    direct = _target(text, profiles=True, lanes=lanes)
    if direct is not None:
        category, canonical = direct
        return AssigneeResolution(value, category, canonical, category)
    alias_target = _assignee_aliases(cfg).get(text.casefold())
    if alias_target is not None:
        target = _target(alias_target, profiles=True, lanes=lanes)
        if target is not None:
            target_category, canonical = target
            return AssigneeResolution(value, AssigneeCategory.ALIAS, canonical, target_category)
    raise InvalidAssigneeError(value)


class AssigneeResolver:
    """Config-snapshot resolver used by a dispatcher tick."""

    def __init__(self, config: Optional[dict] = None):
        self.config = _load_config() if config is None else config

    def resolve(self, value: Any, *, allow_unassigned: bool = True) -> AssigneeResolution:
        return resolve_assignee(value, allow_unassigned=allow_unassigned, config=self.config)
