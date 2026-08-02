"""Read-only conformance check for native Review handoff prerequisites.

The checker deliberately consumes the same human-readable command output a
user sees.  ``hermes profile list`` marks the active profile with ``◆``;
``hermes kanban assignees`` is the source of truth for whether a reviewer can
actually be spawned.  No board mutation is performed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Optional


_ACTIVE_PROFILE_RE = re.compile(r"^\s*◆(?P<name>\S+)")
_ASSIGNEE_RE = re.compile(r"^\s*(?P<name>\S+)\s+(?P<on_disk>yes|no)\s+(?:.*)$")


def parse_active_profile(profile_list_output: str) -> Optional[str]:
    """Return the profile marked active by ``hermes profile list``."""
    for line in profile_list_output.splitlines():
        match = _ACTIVE_PROFILE_RE.match(line)
        if match:
            return match.group("name")
    return None


def parse_assignees(assignees_output: str) -> dict[str, bool]:
    """Parse ``hermes kanban assignees`` into ``name -> on_disk`` values."""
    parsed: dict[str, bool] = {}
    for line in assignees_output.splitlines():
        match = _ASSIGNEE_RE.match(line)
        if match and match.group("name").upper() != "NAME":
            parsed[match.group("name")] = match.group("on_disk") == "yes"
    return parsed


def check_conformance(
    profile_list_output: str,
    assignees_output: str,
    *,
    reviewer: str = "orion",
) -> list[str]:
    """Return actionable conformance errors; an empty list means compliant."""
    active = parse_active_profile(profile_list_output)
    assignees = parse_assignees(assignees_output)
    errors: list[str] = []
    if active is None:
        errors.append("active profile marker ◆ was not found in hermes profile list")
    elif not assignees.get(active, False):
        errors.append(f"active profile {active!r} is not on disk in hermes kanban assignees")
    if reviewer not in assignees or not assignees[reviewer]:
        errors.append(f"reviewer {reviewer!r} is not on disk in hermes kanban assignees")
    return errors


def _run_hermes(hermes: str, *args: str) -> str:
    result = subprocess.run(
        [hermes, *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{' '.join([hermes, *args])} failed: {detail}")
    return result.stdout


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes", default="hermes", help="Hermes executable")
    parser.add_argument("--reviewer", default="orion")
    args = parser.parse_args(argv)
    try:
        profile_output = _run_hermes(args.hermes, "profile", "list")
        assignees_output = _run_hermes(args.hermes, "kanban", "assignees")
    except RuntimeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    errors = check_conformance(profile_output, assignees_output, reviewer=args.reviewer)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"PASS: active profile and reviewer {args.reviewer!r} are spawnable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
