"""Conformance checks for the native Review handoff prerequisites."""

from scripts.check_native_review_conformance import (
    check_conformance,
    parse_assignees,
    parse_active_profile,
)


PROFILE_LIST = """
 Profile          Model                        Gateway       Alias        Distribution
 ───────────────  ───────────────────────────  ────────────  ───────────  ────────────────────
  coder           claude-sonnet                stopped       —            —
 ◆orion           gpt-5                        running       orion        —
  reviewer        gpt-5                        stopped       reviewer     —
"""

ASSIGNEES = """
NAME                  ON DISK   COUNTS
coder                 yes       (idle)
orion                 yes       review=1
reviewer              yes       (idle)
"""


def test_parser_accepts_actual_profile_list_active_marker():
    assert parse_active_profile(PROFILE_LIST) == "orion"


def test_assignee_parser_validates_spawnable_profiles():
    assert parse_assignees(ASSIGNEES)["orion"] is True


def test_conformance_requires_active_profile_and_default_reviewer_on_disk():
    assert check_conformance(PROFILE_LIST, ASSIGNEES, reviewer="orion") == []

    errors = check_conformance(PROFILE_LIST, ASSIGNEES.replace("orion                 yes", "orion                 no"))
    assert any("reviewer 'orion' is not on disk" in error for error in errors)


def test_conformance_rejects_the_old_star_marker():
    output = PROFILE_LIST.replace("◆orion", "*orion")
    assert parse_active_profile(output) is None
    assert any("active profile marker" in error for error in check_conformance(output, ASSIGNEES))
