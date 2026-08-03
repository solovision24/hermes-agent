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
    """The validated meaning of an assignee input.

    ``canonical`` is the value safe to persist or dispatch.  For an alias,
    ``category`` remains ``alias`` while ``target_category`` identifies the
    canonical target kind.
    """

    input_value: Any
    category: AssigneeCategory
    canonical: Optional[str]
    target_category: Optional[AssigneeCategory] = None
    diagnostic: Optional[str] = None

    @property
    def spawnable(self) -> bool:
        return self.target_category is AssigneeCategory.PROFILE or (
            self.category is AssigneeCategory.PROFILE
        )


class InvalidAssigneeError(ValueError):
    """Stable, actionable validation failure for unresolved assignees."""

    code = INVALID_ASSIGNEE

    def __init__(self, value: Any, diagnostic: Optional[str] = None):
        self.value = value
        self.diagnostic = diagnostic or invalid_assignee_diagnostic(value)
        self.resolution = AssigneeResolution(
            value,
            AssigneeCategory.INVALID,
            None,
            diagnostic=self.diagnostic,
        )
        super().__init__(self.diagnostic)


def invalid_assignee_diagnostic(value: Any) -> str:
    """Return the stable operator/model-facing invalid-assignee message."""

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
    """Return case-folded external-lane lookup → configured canonical name.

    The documented form is a list of lane names.  Mapping keys and
    ``[{"name": ...}]`` entries are accepted too because both are common
    YAML representations of a named, non-spawnable lane.
    """

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
    """Return configured external lanes in their canonical spelling."""

    cfg = _load_config() if config is None else config
    return tuple(sorted(set(_external_lanes(cfg).values()), key=str.casefold))


def configured_assignee_aliases(config: Optional[dict] = None) -> dict[str, str]:
    """Return configured aliases mapped to their canonical targets."""
    cfg = _load_config() if config is None else config
    return _assignee_aliases(cfg)


def has_configured_assignee_targets(config: Optional[dict] = None) -> bool:
    """Whether the config declares any non-profile assignee targets.

    Older callers used arbitrary labels as assignees.  Keeping those labels
    readable when no registry is configured preserves compatibility for
    imported boards, while configured boards get strict validation.
    """
    cfg = _load_config() if config is None else config
    return bool(_external_lanes(cfg) or _assignee_aliases(cfg))


def configured_assignee_choices(config: Optional[dict] = None) -> tuple[AssigneeResolution, ...]:
    """Enumerate configured assignee inputs with their target types.

    This is intentionally typed rather than a bare list of strings so CLI and
    dashboard consumers can distinguish spawnable profiles from control-plane
    lanes and display-only aliases without reimplementing resolution rules.
    """
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
    for alias, target in sorted(configured_assignee_aliases(cfg).items()):
        try:
            choices.append(resolve_assignee(alias, allow_unassigned=False, config=cfg))
        except InvalidAssigneeError:
            continue
    return tuple(choices)


def _assignee_aliases(config: dict) -> dict[str, str]:
    section = config.get("kanban")
    if not isinstance(section, dict):
        section = {}
    raw = section.get("assignee_aliases", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for alias, target in raw.items():
        alias_text = str(alias).strip()
        target_text = str(target).strip() if target is not None else ""
        if alias_text and target_text:
            result[alias_text.casefold()] = target_text
    return result


def _profile_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        from hermes_cli.profiles import (
            normalize_profile_name,
            profile_exists,
            validate_profile_name,
        )

        name = normalize_profile_name(str(value))
        # normalize_profile_name intentionally only canonicalizes CLI input;
        # it does not validate path components.  Validate before calling
        # profile_exists/get_profile_dir so values such as ``..`` cannot be
        # treated as a profile rooted outside the profiles directory.
        validate_profile_name(name)
        return name if profile_exists(name) else None
    except (TypeError, ValueError, OSError):
        return None


def _target(
    value: str,
    *,
    profiles: bool,
    lanes: dict[str, str],
) -> Optional[tuple[AssigneeCategory, str]]:
    if profiles:
        profile = _profile_name(value)
        if profile is not None:
            return AssigneeCategory.PROFILE, profile
    lane = lanes.get(value.casefold())
    if lane is not None:
        return AssigneeCategory.EXTERNAL_LANE, lane
    return None


def resolve_assignee(
    value: Any,
    *,
    allow_unassigned: bool = True,
    config: Optional[dict] = None,
) -> AssigneeResolution:
    """Resolve *value* to a canonical, configured Kanban target.

    Profile names are normalized using Hermes' profile rules.  External lanes
    and aliases are matched case-insensitively but their configured target
    spelling is preserved.  Alias targets must resolve directly to a profile
    or external lane; alias chains and stale targets are invalid.
    """

    if value is None or (isinstance(value, str) and value.strip().casefold() in {
        "", "none", "null", "-", "unassigned",
    }):
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

    aliases = _assignee_aliases(cfg)
    alias_target = aliases.get(text.casefold())
    if alias_target is not None:
        target = _target(alias_target, profiles=True, lanes=lanes)
        if target is not None:
            target_category, canonical = target
            return AssigneeResolution(
                value,
                AssigneeCategory.ALIAS,
                canonical,
                target_category,
            )

    raise InvalidAssigneeError(value)


class AssigneeResolver:
    """Config-snapshot resolver used by a dispatcher tick."""

    def __init__(self, config: Optional[dict] = None):
        self.config = _load_config() if config is None else config

    def resolve(self, value: Any, *, allow_unassigned: bool = True) -> AssigneeResolution:
        return resolve_assignee(
            value, allow_unassigned=allow_unassigned, config=self.config,
        )
