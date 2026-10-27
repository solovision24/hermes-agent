"""Per-target snapshot and at-most-once receipt contract for opt-in delivery."""
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from cron import delivery_queue as queue
from cron import target_receipts as receipts

NOW = datetime(2026, 10, 27, 12, tzinfo=UTC)
HUMAN = {"kind": "human", "platform": "telegram", "chat_id": "123", "thread_id": "7"}
MACHINE = {"kind": "machine", "platform": "webhook", "chat_id": "hook-1"}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "receipts.db")
    monkeypatch.setattr(receipts, "_now", lambda: NOW)
    return tmp_path


def add(execution="exec", targets=None, release=None):
    return receipts.enqueue_targets(execution, "job", "brief", targets or [HUMAN, MACHINE],
        not_before=release or NOW, expires_at=NOW + timedelta(days=1))


def test_multi_target_snapshot_restart_and_duplicate_tick(store, monkeypatch):
    original = add(release=NOW + timedelta(hours=2))
    assert {r["kind"] for r in original} == {"human", "machine"}
    assert {r["status"] for r in original} == {"pending"}
    changed = {**HUMAN, "chat_id": "999"}
    assert add() == original  # duplicate tick cannot rewrite the already-snapshotted destination
    assert receipts.release_targets(Mock(), job_state=lambda _: "active") == 0
    monkeypatch.setattr(receipts, "_PROCESS_ID", "replacement-process")
    monkeypatch.setattr(receipts, "_now", lambda: NOW + timedelta(hours=3))
    send = Mock(side_effect=lambda target, *_: ("delivered", None))
    assert receipts.release_targets(send, job_state=lambda _: "active") == 2
    assert receipts.release_targets(send, job_state=lambda _: "active") == 0
    assert {call.args[0]["chat_id"] for call in send.call_args_list} == {"123", "hook-1"}
    assert changed["chat_id"] not in {call.args[0]["chat_id"] for call in send.call_args_list}
    assert {r["status"] for r in receipts.get_target_receipts("exec")} == {"delivered"}


def test_failed_and_ambiguous_target_never_claim_success_or_retry(store):
    add()
    outcomes = iter([("failed", "rejected"), None])
    send = Mock(side_effect=lambda *_: next(outcomes))
    assert receipts.release_targets(send, job_state=lambda _: "active") == 2
    assert receipts.release_targets(send, job_state=lambda _: "active") == 0
    assert {r["status"] for r in receipts.get_target_receipts("exec")} == {"failed", "unknown"}
    assert send.call_count == 2


def test_dead_owner_only_unknown_for_claimed_target(store, monkeypatch):
    add()
    with queue._transaction() as conn:
        conn.execute("UPDATE target_deliveries SET status='delivering',owner_pid=999999,owner_started_at=1 WHERE kind='human'")
    monkeypatch.setattr(receipts, "_owner_is_live", lambda *_: False)
    send = Mock(return_value=("delivered", None))
    assert receipts.release_targets(send, job_state=lambda _: "active") == 1
    assert {r["status"] for r in receipts.get_target_receipts("exec")} == {"unknown", "delivered"}
    send.assert_called_once()


def test_pause_delete_and_expiry(store, monkeypatch):
    add("paused", [HUMAN])
    add("deleted", [MACHINE])
    state = {"job": "paused"}
    send = Mock(return_value=("delivered", None))
    assert receipts.release_targets(send, job_state=lambda _: state["job"]) == 0
    assert receipts.get_target_receipts("paused")[0]["status"] == "pending"
    state["job"] = "deleted"
    assert receipts.release_targets(send, job_state=lambda _: state["job"]) == 2
    send.assert_not_called()
    deleted = receipts.get_target_receipts("paused")[0]
    assert deleted["status"] == "suppressed"
    assert deleted["content"] == "" and deleted["target_json"] == "{}"
    assert receipts.get_target_receipts("deleted")[0]["target_json"] == "{}"
    add("expired", [HUMAN])
    monkeypatch.setattr(receipts, "_now", lambda: NOW + timedelta(days=2))
    assert receipts.release_targets(send, job_state=lambda _: "paused") == 1
    expired = receipts.get_target_receipts("expired")[0]
    assert expired["status"] == "failed"
    assert expired["content"] == "" and expired["target_json"] == "{}"


def test_suppressed_and_expired_receipts_obey_terminal_bound(store, monkeypatch):
    monkeypatch.setattr(receipts, "MAX_TERMINAL_TARGETS", 0)
    add("deleted", [HUMAN])
    add("expired", [MACHINE])
    assert receipts.release_targets(Mock(), job_state=lambda _: "deleted", limit=1) == 1
    monkeypatch.setattr(receipts, "_now", lambda: NOW + timedelta(days=2))
    assert receipts.release_targets(Mock(), job_state=lambda _: "paused") == 1
    assert receipts.get_target_receipts("deleted")[0]["status"] == "suppressed"
    assert receipts.get_target_receipts("expired")[0]["status"] == "failed"
    with queue._transaction() as conn:
        assert conn.execute("SELECT COUNT(*) FROM target_deliveries").fetchone()[0] == 0


def test_active_claim_is_profile_scoped_and_release_uses_claimed_home(store, monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    monkeypatch.setattr(queue, "DELIVERY_DB", None)
    a, b = tmp_path / "a", tmp_path / "b"
    home_a = set_hermes_home_override(a)
    try:
        add("shared", [HUMAN])
        home_b = set_hermes_home_override(b)
        try:
            add("shared", [HUMAN])
            with queue._transaction() as conn:
                conn.execute("""UPDATE target_deliveries SET status='delivering',
                    owner_process_id=?,owner_pid=?""", (receipts._PROCESS_ID, 999999))
        finally:
            reset_hermes_home_override(home_b)

        def send(*_):
            # One gateway sends A while its served B has an abandoned identical key.
            switched = set_hermes_home_override(b)
            try:
                assert receipts.recover_targets() == 1
                assert receipts.get_target_receipts("shared")[0]["status"] == "unknown"
            finally:
                reset_hermes_home_override(switched)
            return "delivered", None

        assert receipts.release_targets(send, job_state=lambda _: "active") == 1
        assert receipts.get_target_receipts("shared")[0]["status"] == "delivered"
        assert receipts._ACTIVE == set()
    finally:
        reset_hermes_home_override(home_a)


def test_profile_isolation(store, monkeypatch, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    monkeypatch.setattr(queue, "DELIVERY_DB", None)
    a, b = tmp_path / "a", tmp_path / "b"
    token = set_hermes_home_override(a)
    try:
        add("shared", [HUMAN])
        other = set_hermes_home_override(b)
        try:
            assert receipts.get_target_receipts("shared") == []
            add("shared", [MACHINE])
        finally:
            reset_hermes_home_override(other)
        assert receipts.get_target_receipts("shared")[0]["kind"] == "human"
    finally:
        reset_hermes_home_override(token)


def test_bounds_atomicity_and_tombstones(store, monkeypatch):
    monkeypatch.setattr(receipts, "MAX_PENDING_TARGETS", 2)
    add("first")
    with pytest.raises(ValueError, match="capacity"):
        add("second", [HUMAN])
    assert receipts.get_target_receipts("second") == []
    with pytest.raises(ValueError, match="duplicate"):
        add("duplicate", [HUMAN, HUMAN])
    with pytest.raises(ValueError, match="duplicate"):
        add("ambiguous-kind", [HUMAN, {**HUMAN, "kind": "machine"}])
    with pytest.raises(ValueError, match="size limit"):
        receipts.enqueue_targets("large", "job", "x" * 65537, [HUMAN],
            not_before=NOW, expires_at=NOW + timedelta(days=1))
    monkeypatch.setattr(receipts, "MAX_TERMINAL_TARGETS", 0)
    receipts.release_targets(Mock(return_value=("delivered", None)), job_state=lambda _: "active")
    assert {r["status"] for r in receipts.get_target_receipts("first")} == {"delivered"}
    assert add("first") == receipts.get_target_receipts("first")
    assert receipts.release_targets(Mock(), job_state=lambda _: "active") == 0


def test_legacy_aggregate_queue_unaffected(store):
    queue.enqueue("legacy", {"id": "old"}, "old")
    assert receipts.get_target_receipts("legacy") == []
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once()
    assert queue.get_status("legacy")["status"] == "delivered"


def test_failure_lane_kind_and_exception_are_independent(store):
    receipts.enqueue_targets("failure", "job", "alarm", [HUMAN, MACHINE],
        not_before=NOW, expires_at=NOW + timedelta(days=1), for_failure=True)
    def send(target, content, for_failure):
        assert content == "alarm" and for_failure
        if target["kind"] == "human":
            raise TimeoutError("ambiguous")
        return "delivered", None
    assert receipts.release_targets(send, job_state=lambda _: "active") == 2
    assert {r["kind"]: r["status"] for r in receipts.get_target_receipts("failure")} == {
        "human": "unknown", "machine": "delivered"}


def test_conflicting_duplicate_does_not_change_pending_snapshot(store):
    add("immutable", [HUMAN])
    with pytest.raises(ValueError, match="different delivery payload"):
        receipts.enqueue_targets("immutable", "job", "replacement", [HUMAN],
            not_before=NOW, expires_at=NOW + timedelta(days=1))
    assert receipts.get_target_receipts("immutable")[0]["content"] == "brief"


def test_same_process_failed_terminal_write_is_fenced_not_replayed(store, monkeypatch):
    add("write-failure", [HUMAN])
    original = receipts._prune
    called = False

    def fail_after_send(conn, now):
        nonlocal called
        if called:
            raise OSError("disk full")
        called = True
        return original(conn, now)

    monkeypatch.setattr(receipts, "_prune", fail_after_send)
    send = Mock(return_value=("delivered", None))
    with pytest.raises(OSError, match="disk full"):
        receipts.release_targets(send, job_state=lambda _: "active")
    monkeypatch.setattr(receipts, "_prune", original)
    assert receipts.recover_targets() == 1
    assert receipts.get_target_receipts("write-failure")[0]["status"] == "unknown"
    assert receipts.release_targets(send, job_state=lambda _: "active") == 0
    send.assert_called_once()


def test_unwritable_store_rejects_before_claim_or_send(store, monkeypatch):
    original = receipts._prune
    monkeypatch.setattr(receipts, "_prune", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        add("write-rejected", [HUMAN])
    monkeypatch.setattr(receipts, "_prune", original)
    assert receipts.get_target_receipts("write-rejected") == []


def test_corrupt_store_fails_loud_without_sending(store):
    (store / "receipts.db").write_text("not sqlite", encoding="utf-8")
    with pytest.raises(Exception, match="cron/deliveries.db|database"):
        add("corrupt", [HUMAN])


def test_tombstone_retention_is_bounded_by_count_and_age(store, monkeypatch):
    monkeypatch.setattr(receipts, "MAX_TERMINAL_TARGETS", 0)
    monkeypatch.setattr(receipts, "MAX_TOMBSTONES", 2)
    for index in range(3):
        execution = f"exec-{index}"
        add(execution, [HUMAN])
        assert receipts.release_targets(Mock(return_value=("delivered", None)),
                                         job_state=lambda _: "active") == 1
    assert receipts.get_target_receipts("exec-0") == []
    assert receipts.get_target_receipts("exec-2")[0]["status"] == "delivered"
    monkeypatch.setattr(receipts, "_now", lambda: NOW + timedelta(days=31))
    receipts.recover_targets()
    assert receipts.get_target_receipts("exec-1") == []
    assert receipts.get_target_receipts("exec-2") == []
