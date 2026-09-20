"""Crash-diagnosis + spawn-preflight contract (card t_04494161).

Three failure shapes used to be indistinguishable on the board:

* a worker that vanished — the dispatcher's bare ``pid N not alive`` verdict,
  with the real cause only in ``<task>.log``;
* a spawn that could never have worked at all (an unresolvable forced skill, an
  absent assignee profile, an unresolvable workspace) — discovered only by the
  child dying, and counted against the crash/retry budget every attempt;
* everything else that crashed.

These tests pin the observable contract: the run error names the real cause, a
spawn precondition fails once without consuming the crash budget and never
appears as ``pid N not alive``, repeated ticks are idempotent, and the four churn
classes are separately countable from the board stats surface.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kdg

BOARD = "default"
PROFILE = "probe-crashdiag"
# Deliberately not a real skill/bundle name on any host.
BOGUS_SKILL = "bogus-skill-crashdiag-04494161"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME (with one real-ish profile) + isolated kanban.db."""
    home = tmp_path / ".hermes"
    (home / "profiles" / PROFILE).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", BOARD)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACES_ROOT", str(home / "kanban" / "workspaces"))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board=BOARD)
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


@pytest.fixture
def spawnable(monkeypatch):
    """The isolated profile name must pass the dispatcher's profile guard."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _write_skill(home: Path, name: str) -> None:
    skill_dir = home / "profiles" / PROFILE / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Fixture skill for the crash-diagnostics suite.\n---\n\nFixture body.\n",
        encoding="utf-8",
    )


def _dead_grandchild():
    """A ``sleep`` that is NOT our child, so no reap status is ever recorded for
    it — the live ``pid N not alive`` shape (gone, exit status unknown)."""
    helper = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,time; p=subprocess.Popen(['sleep','300']); "
         "print(p.pid, flush=True); time.sleep(600)"],
        stdout=subprocess.PIPE, start_new_session=True, text=True,
    )
    try:
        pid = int(helper.stdout.readline().strip())
    except Exception:
        helper.kill()
        raise
    return helper, pid


@pytest.fixture
def dead_worker_pid():
    helpers = []
    pids = []

    def _make():
        helper, pid = _dead_grandchild()
        helpers.append(helper)
        pids.append(pid)
        return pid

    yield _make
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for helper in helpers:
        helper.kill()
        helper.wait(timeout=5)


def _recorder(pid):
    calls = []

    def _spawn(task, workspace, board=None):
        calls.append(task.id)
        return pid

    return calls, _spawn


def _tick(conn, spawn_fn, *, failure_limit=3):
    return kbd.dispatch_once(
        conn, spawn_fn=spawn_fn, ttl_seconds=300, failure_limit=failure_limit,
    )


def _rewind(conn, task_id):
    """Age the claim past the launch grace window so the reclaim can see it."""
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id = ?", (task_id,))
    conn.execute("UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id = ?", (task_id,))
    conn.commit()


def _await_worker_dead(pid, timeout=5.0):
    """Wait until the dispatcher's own liveness check agrees the worker is gone.

    SIGKILL is asynchronous, so a reclaim issued immediately after the kill can
    still observe the process (racing the test, not the product).
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not kbd._pid_alive(pid):
            return True
        time.sleep(0.02)
    return not kbd._pid_alive(pid)


def _task(conn, task_id):
    return conn.execute(
        "SELECT status, consecutive_failures, last_failure_error, worker_pid, claim_lock "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()


def _runs(conn, task_id):
    return conn.execute(
        "SELECT id, status, outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()


def _events(conn, task_id, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (task_id, kind),
    ).fetchall()


# --- worker log tail ---------------------------------------------------------

def test_worker_log_tail_missing_log_is_empty_and_never_raises(kanban_home):
    assert kbd._worker_log_tail("t_definitely_missing") == ""
    assert kbd._worker_log_tail("") == ""


def test_worker_log_tail_is_ansi_stripped_redacted_and_bounded(kanban_home):
    token = "ghp_" + "a1b2c3d4e5" * 4
    log = kb.worker_logs_dir() / "t_tail.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    long_line = "x" * 500
    log.write_text(
        "\n".join([
            "\x1b[32mline-one\x1b[0m",
            f"token: {token}",
            long_line,
        ]) + "\n",
        encoding="utf-8",
    )

    tail = kbd._worker_log_tail("t_tail", max_lines=1)
    assert "\x1b" not in tail and long_line not in tail
    assert len(tail) <= kbd._WORKER_LOG_TAIL_LINE_CHARS

    full = kbd._worker_log_tail("t_tail")
    assert "line-one" in full
    assert token not in full, "log tail must never carry a raw credential"
    assert len(full) <= kbd._WORKER_LOG_TAIL_CHARS


def test_worker_log_tail_force_redacts_when_display_redaction_disabled(
    kanban_home, monkeypatch,
):
    """Durable DB persistence must redact even when the user disables display redaction."""
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "false")
    token = "ghp_" + "a1b2c3d4e5" * 4
    log = kb.worker_logs_dir() / "t_force_redact.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"leaked: {token}\n", encoding="utf-8")

    assert token not in kbd._worker_log_tail("t_force_redact")
    assert token not in (kbd._display_safe_log_text(f"leaked: {token}") or "")


def test_worker_log_tail_omits_content_when_sanitizer_fails(kanban_home, monkeypatch):
    """If strip/redact cannot be guaranteed, omit the tail rather than persist raw secrets."""
    token = "ghp_" + "z9y8x7w6v5" * 4
    log = kb.worker_logs_dir() / "t_sanitizer_fail.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"secret={token}\n", encoding="utf-8")

    def _boom(text, *, force=False, **_kwargs):
        raise RuntimeError("redactor offline")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", _boom)
    assert kbd._display_safe_log_text(f"secret={token}") is None
    assert kbd._worker_log_tail("t_sanitizer_fail") == ""
    assert token not in kbd._compose_crash_error("pid 1 not alive", kbd._worker_log_tail("t_sanitizer_fail"))


def test_worker_log_tail_omits_content_when_redact_import_fails(kanban_home, monkeypatch):
    token = "sk-ant-" + "a" * 40
    log = kb.worker_logs_dir() / "t_import_fail.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"key={token}\n", encoding="utf-8")

    import builtins
    real_import = builtins.__import__

    def _block_redact(name, *args, **kwargs):
        if name == "agent.redact" or name.startswith("agent.redact."):
            raise ImportError("simulated redact import failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_redact)
    assert kbd._worker_log_tail("t_import_fail") == ""
    assert kbd._display_safe_log_text(f"key={token}") is None


# --- vanished worker: propagate the real failure -----------------------------

def test_vanished_worker_error_carries_log_tail_and_pid(conn, spawnable, dead_worker_pid):
    tid = kb.create_task(conn, title="killed mid run", assignee=PROFILE)
    pid = dead_worker_pid()
    calls, spawn = _recorder(pid)
    _tick(conn, spawn)
    assert calls == [tid]
    _rewind(conn, tid)

    log = kb.worker_logs_dir() / f"{tid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "session started\nstep 47/90\n"
        "\x1b[31mFATAL: provider stream closed unexpectedly\x1b[0m\n",
        encoding="utf-8",
    )
    os.kill(pid, signal.SIGKILL)
    assert _await_worker_dead(pid)

    assert kbd.detect_crashed_workers(conn) == [tid]
    run = _runs(conn, tid)[0]
    assert run["status"] == "crashed"
    assert run["error"].startswith(f"pid {pid} not alive — worker log tail:")
    assert "FATAL: provider stream closed unexpectedly" in run["error"]
    assert "\x1b" not in run["error"]

    metadata = json.loads(run["metadata"])
    assert metadata["pid"] == pid, "the PID must stay in metadata"
    assert "FATAL: provider stream closed unexpectedly" in metadata["worker_log_tail"]

    row = _task(conn, tid)
    assert "FATAL: provider stream closed unexpectedly" in row["last_failure_error"], (
        "the board/Mission Control error must name the real cause"
    )


def test_enriched_error_keeps_the_fingerprint_grouping(conn, spawnable, dead_worker_pid):
    """Systemic-crash grouping keys on the verdict; a varying tail must not split it."""
    tid = kb.create_task(conn, title="fingerprint", assignee=PROFILE)
    pid = dead_worker_pid()
    assert pid > 0
    calls, spawn = _recorder(pid)
    _tick(conn, spawn)
    assert calls == [tid]
    _rewind(conn, tid)
    log = kb.worker_logs_dir() / f"{tid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("some tail that differs run to run\n", encoding="utf-8")
    os.kill(pid, signal.SIGKILL)
    assert _await_worker_dead(pid)
    assert kbd.detect_crashed_workers(conn) == [tid]

    enriched = _runs(conn, tid)[0]["error"]
    assert enriched and enriched.startswith(f"pid {pid} not alive")
    assert kbd._error_fingerprint(enriched) == kbd._error_fingerprint(f"pid {pid} not alive")


# --- spawn preconditions -----------------------------------------------------

def test_bogus_forced_skill_fails_once_without_spawning_or_burning_budget(
    conn, spawnable, monkeypatch,
):
    tid = kb.create_task(
        conn, title="bogus forced skill", assignee=PROFILE, skills=[BOGUS_SKILL],
    )
    calls, spawn = _recorder(None)

    res = _tick(conn, spawn)
    assert calls == [], "a card that cannot start must not be spawned at all"
    assert [(t, check) for t, check, _ in res.spawn_precondition_failed] == [(tid, "forced_skills")]
    reason = res.spawn_precondition_failed[0][2]
    assert BOGUS_SKILL in reason and "Unknown skill(s)" in reason

    row = _task(conn, tid)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0, "a precondition must not consume the crash budget"
    assert BOGUS_SKILL in row["last_failure_error"]
    assert row["worker_pid"] is None and row["claim_lock"] is None

    runs = _runs(conn, tid)
    assert [r["outcome"] for r in runs] == ["precondition_failed"]
    assert json.loads(runs[0]["metadata"])["precondition"] == "forced_skills"

    events = _events(conn, tid, "spawn_precondition_failed")
    assert len(events) == 1 and json.loads(events[0]["payload"])["check"] == "forced_skills"

    # Idempotent: the next tick neither re-runs the card nor re-reports it.
    res2 = _tick(conn, spawn)
    assert res2.spawn_precondition_failed == []
    assert calls == []
    assert len(_runs(conn, tid)) == 1


def test_partial_skill_miss_still_spawns(conn, spawnable, kanban_home):
    """cli.py parity: a worker only dies when EVERY forced skill is missing."""
    _write_skill(kanban_home, "probe-ok-crashdiag")
    assert kbd.preflight_forced_skills(PROFILE, ["probe-ok-crashdiag"]) == []
    assert kbd.preflight_forced_skills(PROFILE, [BOGUS_SKILL]) == [BOGUS_SKILL]

    tid = kb.create_task(
        conn, title="partial skill miss", assignee=PROFILE,
        skills=["probe-ok-crashdiag", BOGUS_SKILL],
    )
    calls, spawn = _recorder(None)
    res = _tick(conn, spawn)
    assert res.spawn_precondition_failed == []
    assert calls == [tid], "one resolvable forced skill is enough to start the worker"


def test_missing_assignee_profile_is_reported_once_and_stays_claimable(conn, monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: False)
    ghost = "ghost-profile-crashdiag"
    tid = kb.create_task(conn, title="ghost assignee", assignee=ghost)
    calls, spawn = _recorder(None)

    res = _tick(conn, spawn)
    assert res.skipped_nonspawnable == [tid]
    assert calls == []

    row = _task(conn, tid)
    assert row["status"] == "ready", "a control-plane lane must still be able to claim it"
    assert row["consecutive_failures"] == 0
    assert ghost in row["last_failure_error"]

    events = _events(conn, tid, "spawn_precondition_failed")
    assert len(events) == 1
    assert json.loads(events[0]["payload"])["check"] == "assignee_profile"

    _tick(conn, spawn)
    assert len(_events(conn, tid, "spawn_precondition_failed")) == 1, "ticks must not spam"
    assert _runs(conn, tid) == []


def test_unresolvable_workspace_is_a_precondition_not_a_crash(conn, spawnable):
    tid = kb.create_task(
        conn, title="dir workspace without path", assignee=PROFILE,
        workspace_kind="dir",
    )
    calls, spawn = _recorder(None)
    res = _tick(conn, spawn)

    assert calls == []
    assert [(t, check) for t, check, _ in res.spawn_precondition_failed] == [(tid, "workspace")]
    row = _task(conn, tid)
    assert row["status"] == "blocked" and row["consecutive_failures"] == 0
    assert "workspace" in row["last_failure_error"]
    assert [r["outcome"] for r in _runs(conn, tid)] == ["precondition_failed"]


def test_relative_workspace_path_is_deterministic_precondition(conn, spawnable):
    tid = kb.create_task(
        conn, title="relative dir path", assignee=PROFILE,
        workspace_kind="dir", workspace_path="relative/not/absolute",
    )
    calls, spawn = _recorder(None)
    res = _tick(conn, spawn)

    assert calls == []
    assert [(t, check) for t, check, _ in res.spawn_precondition_failed] == [(tid, "workspace")]
    row = _task(conn, tid)
    assert row["status"] == "blocked" and row["consecutive_failures"] == 0
    assert [r["outcome"] for r in _runs(conn, tid)] == ["precondition_failed"]
    # Sticky: next tick must not re-burn or re-report.
    assert _tick(conn, spawn).spawn_precondition_failed == []
    assert len(_runs(conn, tid)) == 1


def test_transient_workspace_error_uses_spawn_failed_retry_budget(
    conn, spawnable, monkeypatch,
):
    """Git/FS operational failures must not sticky-block as spawn_precondition."""
    tid = kb.create_task(conn, title="transient wt", assignee=PROFILE)
    calls, spawn = _recorder(None)

    def _boom(task, board=None):
        raise RuntimeError("git worktree add failed: index.lock: File exists")

    monkeypatch.setattr(kbd._kbw, "resolve_workspace", _boom)
    res = _tick(conn, spawn, failure_limit=3)

    assert calls == []
    assert res.spawn_precondition_failed == []
    row = _task(conn, tid)
    assert row["status"] == "ready", "operational failure must stay retryable"
    assert row["consecutive_failures"] == 1
    assert "index.lock" in (row["last_failure_error"] or "")
    assert [r["outcome"] for r in _runs(conn, tid)] == ["spawn_failed"]


def test_empty_second_attempt_does_not_inherit_prior_log_tail(
    conn, spawnable, dead_worker_pid,
):
    """Append-only worker logs + a silent re-run must not re-attribute the prior cause."""
    tid = kb.create_task(conn, title="multi attempt log boundary", assignee=PROFILE)
    log = kb.worker_logs_dir() / f"{tid}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    # Attempt 1: worker wrote a real cause, then vanished.
    pid1 = dead_worker_pid()
    calls, spawn = _recorder(pid1)
    _tick(conn, spawn)
    assert calls == [tid]
    _rewind(conn, tid)
    prior_cause = "FATAL: provider stream closed on attempt-1"
    log.write_text(f"session started\n{prior_cause}\n", encoding="utf-8")
    os.kill(pid1, signal.SIGKILL)
    assert _await_worker_dead(pid1)
    assert kbd.detect_crashed_workers(conn) == [tid]
    assert prior_cause in _runs(conn, tid)[0]["error"]

    # Attempt 2: spawn again, write nothing, vanish — must not inherit attempt-1.
    pid2 = dead_worker_pid()
    calls2, spawn2 = _recorder(pid2)
    _tick(conn, spawn2)
    assert calls2 == [tid]
    run_meta = json.loads(_runs(conn, tid)[-1]["metadata"] or "{}")
    assert "worker_log_offset" in run_meta
    assert int(run_meta["worker_log_offset"]) == log.stat().st_size
    _rewind(conn, tid)
    os.kill(pid2, signal.SIGKILL)
    assert _await_worker_dead(pid2)
    assert kbd.detect_crashed_workers(conn) == [tid]

    second = _runs(conn, tid)[-1]
    assert second["status"] == "crashed"
    assert prior_cause not in (second["error"] or "")
    assert "worker log tail" not in (second["error"] or "")
    metadata = json.loads(second["metadata"] or "{}")
    assert prior_cause not in (metadata.get("worker_log_tail") or "")


# --- failure-class reporting -------------------------------------------------

def _insert_run(conn, task_id, *, outcome, error=None, metadata=None, ended=True, offset=0):
    now = int(time.time()) - offset
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, started_at, ended_at, outcome, error, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (task_id, PROFILE, outcome or "running", now - 5, now if ended else None, outcome, error,
         json.dumps(metadata) if metadata else None),
    )
    conn.commit()


def test_failure_classes_separate_the_four_churn_shapes(conn, spawnable):
    tid = kb.create_task(conn, title="classes", assignee=PROFILE)
    assert kdg.classify_run_failure("crashed", "pid 12 not alive", {"pid": 12}) == "worker_vanished"
    assert kdg.classify_run_failure("crashed", "pid 12 not alive — worker log tail: boom", {"pid": 12}) == "worker_vanished"
    assert kdg.classify_run_failure("crashed", "pid 12 exited with code 1", {"exit_kind": "nonzero_exit"}) == "worker_error"
    assert kdg.classify_run_failure(
        "crashed", "worker exited cleanly (rc=0) without calling kanban_complete",
        {"protocol_violation": True},
    ) == "protocol_violation"
    assert kdg.classify_run_failure("stale", "no heartbeat for 4000s after 9000s running") == "stale_lock"
    assert kdg.classify_run_failure("reclaimed", "orphaned running card (broken claim bookkeeping)") == "stale_lock"
    assert kdg.classify_run_failure("timed_out", "elapsed 700s > limit 600s") == "iteration_budget"
    assert kdg.classify_run_failure(
        "precondition_failed", "spawn precondition failed: Unknown skill(s): nope",
        {"precondition": "forced_skills"},
    ) == "spawn_precondition"
    assert kdg.classify_run_failure("completed", None) == ""
    assert kdg.classify_run_failure("", None) == "other"

    _insert_run(conn, tid, outcome="crashed", error="pid 999 not alive")
    _insert_run(conn, tid, outcome="crashed", error="protocol violation — nope", metadata={"protocol_violation": True})
    _insert_run(conn, tid, outcome="stale", error="no heartbeat ever after 9000s running")
    _insert_run(conn, tid, outcome="timed_out", error="elapsed 700s > limit 600s")
    _insert_run(conn, tid, outcome="completed", error=None)
    _insert_run(conn, tid, outcome="running", error=None, ended=False)

    counts = kdg.failure_class_counts(conn, task_id=tid)
    assert counts["worker_vanished"] == 1
    assert counts["protocol_violation"] == 1
    assert counts["stale_lock"] == 1
    assert counts["iteration_budget"] == 1
    assert counts["other"] == 0, "successful and in-flight runs carry no verdict"
    assert set(counts) == set(kdg.FAILURE_CLASSES)

    stats = kb.board_stats(conn)
    assert stats["failure_classes"]["worker_vanished"] == 1
    assert stats["failure_classes_recent"]["stale_lock"] == 1


def test_lifecycle_terminal_success_outcomes_are_not_failures(conn, spawnable):
    """Actual review-lane terminal successes must not land under ``other``."""
    tid = kb.create_task(conn, title="success outcomes", assignee=PROFILE)
    for outcome in ("completed", "review_requested", "submitted_for_review", "review"):
        assert kdg.classify_run_failure(outcome, None) == "", outcome
        _insert_run(conn, tid, outcome=outcome)

    counts = kdg.failure_class_counts(conn, task_id=tid)
    assert counts["other"] == 0
    assert sum(counts.values()) == 0
    assert "review_requested" in kdg._SUCCESS_RUN_OUTCOMES