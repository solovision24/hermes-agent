"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ship a feature", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kb.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason


def test_tool_only_run_title_promotes_single_task_without_llm(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="run gen_scene.py", triage=True)

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        # The pre-check must promote BEFORE any LLM call; a call would raise.
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    assert outcome.child_ids is None
    assert "single script run" in outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    # Promoted out of triage as exactly ONE card. `recompute_ready` flips a
    # parent-free todo to ready immediately (the dispatcher would do the same
    # on its next tick), so the landing column is ready, not todo.
    assert task.status == "ready"
    assert task.assignee == "dev"
    assert "gen_scene.py" in (task.body or "")


def test_tool_only_run_body_prefix_promotes_single_task_without_llm(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="whatever",
            body="Run: python3 /tmp/gen_scene.py\n\nAcceptance criteria:\n- report output",
            triage=True,
        )

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert "python3 /tmp/gen_scene.py" in (task.body or "")


def test_prose_run_title_still_uses_llm_path(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="run the marketing campaign", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Run the campaign.",
        "assignee": None,
    })

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    # Prose titles keep the normal LLM fanout path (tightened here).
    assert outcome.ok, outcome.reason
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.title == "Tightened title"


def test_tool_only_run_title_extracts_command_prefix_not_prose(kanban_home):
    # The historical decompose-noise shape: a run title with trailing prose
    # (e.g. after a specify rewrite).  The promoted card must carry only the
    # matched script-run prefix as the command, never the prose.
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="Run gen_scene.py in /home/solo and verify change",
            triage=True,
        )

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert "gen_scene.py" in (task.body or "")
    # The trailing prose must not leak into the command block.
    assert "in /home/solo and verify change" not in (task.body or "")


def test_tool_only_run_body_prefix_preserves_full_command(kanban_home):
    # The coding gate's exact scope line: an env-prefixed cd chain plus the
    # script invocation.  The promoted card must preserve the FULL command.
    full_command = (
        'cd /home/solo && SOLODESIGNSTUDIO_API_KEY="$SOLODESIGNSTUDIO_API_KEY" '
        "python3 gen_scene.py"
    )
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="whatever",
            body=f"Run: {full_command}\n\nAcceptance criteria:\n- Implement",
            triage=True,
        )

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert full_command in (task.body or "")
    # The acceptance criteria must carry the read-only classification.
    assert "Do not modify repository files" in (task.body or "")


def test_tool_only_run_body_prefix_is_case_insensitive(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="whatever",
            body="run: python3 /tmp/gen_scene.py\n\nAcceptance criteria:\n- report",
            triage=True,
        )

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.status == "ready"
    assert "python3 /tmp/gen_scene.py" in (task.body or "")


def test_tool_only_run_decompose_creates_exactly_one_card(kanban_home):
    # Acceptance replay: a simple `run gen_scene.py` request must produce at
    # most one card and no overlapping Run cards.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="run gen_scene.py", triage=True)

    patches = _patch_list_profiles(["orchestrator", "dev"])
    for p in patches:
        p.start()
    try:
        with patch("agent.auxiliary_client.call_llm", side_effect=AssertionError("LLM must not be called")), \
             patch("hermes_cli.kanban_decompose._load_config",
                   return_value={"kanban": {"default_assignee": "dev"}}):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is False
    assert outcome.child_ids is None

    with kb.connect() as conn:
        all_tasks = kb.list_tasks(conn)
        task = kb.get_task(conn, tid)
    # The promoted card is the ONLY card on the board — no duplicate
    # run/verify children and no extra `Run:` wrapper cards.
    assert len(all_tasks) == 1
    assert all_tasks[0].id == tid
    assert task is not None
    assert task.status == "ready"
    assert task.assignee == "dev"
    assert "gen_scene.py" in (task.body or "")
    assert "Do not modify repository files" in (task.body or "")


