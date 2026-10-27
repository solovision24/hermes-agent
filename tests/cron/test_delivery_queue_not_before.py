"""Durable bounded not-before handoff; no quiet-window policy is wired yet."""

from datetime import UTC, datetime
import sqlite3
import subprocess
import sys
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from cron import delivery_queue as queue


@pytest.fixture
def clock(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    now = [datetime(2026, 3, 8, 6, 30, tzinfo=UTC)]
    monkeypatch.setattr(queue, "_hermes_now", lambda: now[0])
    return now


def test_future_handoff_survives_restart_and_duplicate_drains(clock):
    expiry = datetime(2026, 3, 9, 12, tzinfo=UTC)
    release = datetime(2026, 3, 8, 8, tzinfo=UTC)
    queue.enqueue("exec-1", {"id": "job-1", "deliver": "telegram:123"}, "brief",
                  not_before=release, expires_at=expiry)
    send = Mock(return_value=None)
    assert queue.drain(send) == 0
    assert queue.get_status("exec-1")["status"] == "pending"
    queue.enqueue("exec-1", {"id": "other"}, "duplicate", not_before=release, expires_at=expiry)
    clock[0] = release
    assert queue.drain(send) == 1
    assert queue.drain(send) == 0
    send.assert_called_once_with({"id": "job-1", "deliver": "telegram:123"}, "brief", False)
    assert queue.get_status("exec-1")["status"] == "delivered"


def test_replacement_process_releases_persisted_receipt_without_agent(clock):
    release = datetime(2026, 3, 8, 8, tzinfo=UTC)
    queue.enqueue("restarted", {"id": "job-restarted"}, "saved payload",
                  not_before=release, expires_at=datetime(2026, 3, 9, tzinfo=UTC))
    script = """
import sys
from datetime import UTC, datetime
from pathlib import Path
from cron import delivery_queue as queue
queue.DELIVERY_DB = Path(sys.argv[1])
queue._hermes_now = lambda: datetime(2026, 3, 8, 8, tzinfo=UTC)
assert queue.drain(lambda job, content, failed: None) == 1
assert queue.drain(lambda job, content, failed: None) == 0
assert queue.get_status('restarted')['status'] == 'delivered'
"""
    result = subprocess.run([sys.executable, "-c", script, str(queue.DELIVERY_DB)],
                            check=False, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_eligibility_uses_instants_across_spring_gap_and_fall_fold(clock):
    zone = ZoneInfo("America/New_York")
    spring = datetime(2026, 3, 8, 3, 0, tzinfo=zone)
    queue.enqueue("spring", {"id": "spring"}, "a", not_before=spring,
                  expires_at=datetime(2026, 3, 9, tzinfo=zone))
    assert queue.claim_next() is None
    clock[0] = spring.astimezone(UTC)
    assert queue.claim_next()["execution_id"] == "spring"

    fall = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    clock[0] = datetime(2026, 11, 1, 5, 45, tzinfo=UTC)
    queue.enqueue("fall", {"id": "fall"}, "b", not_before=fall,
                  expires_at=datetime(2026, 11, 2, tzinfo=zone))
    assert queue.claim_next() is None
    clock[0] = fall.astimezone(UTC)
    assert queue.claim_next()["execution_id"] == "fall"


def test_deferred_pending_capacity_fails_loud_without_eviction(clock, monkeypatch):
    monkeypatch.setattr(queue, "MAX_DEFERRED_PENDING", 1)
    release = datetime(2026, 3, 9, tzinfo=UTC)
    expiry = datetime(2026, 3, 10, tzinfo=UTC)
    queue.enqueue("first", {"id": "first"}, "a", not_before=release, expires_at=expiry)
    with pytest.raises(ValueError, match="capacity"):
        queue.enqueue("second", {"id": "second"}, "b", not_before=release, expires_at=expiry)
    assert queue.get_status("first")["status"] == "pending"
    assert queue.get_status("second") is None


def test_expired_unclaimed_delivery_is_never_sent(clock):
    release = datetime(2026, 3, 9, tzinfo=UTC)
    expiry = datetime(2026, 3, 10, tzinfo=UTC)
    queue.enqueue("expired", {"id": "expired"}, "brief", not_before=release,
                  expires_at=expiry, for_failure=True)
    clock[0] = expiry
    send = Mock()
    assert queue.drain(send) == 0
    send.assert_not_called()
    row = queue.get_status("expired")
    assert row["status"] == "failed"
    assert "expired" in row["error"]
    assert row["content"] == ""


@pytest.mark.parametrize("release,expiry", [
    (datetime(2026, 3, 9), datetime(2026, 3, 10, tzinfo=UTC)),
    (datetime(2026, 3, 9, tzinfo=UTC), None),
    (datetime(2026, 3, 9, tzinfo=UTC), datetime(2026, 3, 8, tzinfo=UTC)),
])
def test_invalid_deferred_window_fails_before_persistence(clock, release, expiry):
    with pytest.raises(ValueError):
        queue.enqueue("invalid", {"id": "invalid"}, "brief",
                      not_before=release, expires_at=expiry)
    assert queue.get_status("invalid") is None


def test_legacy_immediate_delivery_unchanged(clock):
    queue.enqueue("immediate", {"id": "job"}, "brief")
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "job"}, "brief", False)


def test_microsecond_boundary_does_not_claim_early(clock):
    release = datetime(2026, 3, 8, 8, 0, 0, 900, UTC)
    queue.enqueue("micro", {"id": "micro"}, "brief", not_before=release,
                  expires_at=datetime(2026, 3, 9, tzinfo=UTC))
    clock[0] = release.replace(microsecond=899)
    assert queue.claim_next() is None
    clock[0] = release
    assert queue.claim_next()["execution_id"] == "micro"


def test_deferred_payload_size_is_bounded(clock, monkeypatch):
    monkeypatch.setattr(queue, "MAX_DEFERRED_PAYLOAD_BYTES", 2)
    with pytest.raises(ValueError, match="size limit"):
        queue.enqueue("large", {"id": "large"}, "abc",
                      not_before=datetime(2026, 3, 9, tzinfo=UTC),
                      expires_at=datetime(2026, 3, 10, tzinfo=UTC))
    assert queue.get_status("large") is None


def test_existing_schema_migrates_without_changing_old_pending_delivery(clock):
    db = queue.DELIVERY_DB
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE deliveries (
            execution_id TEXT PRIMARY KEY, job_json TEXT NOT NULL, content TEXT NOT NULL,
            for_failure INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL,
            owner_process_id TEXT, owner_pid INTEGER, owner_started_at INTEGER,
            created_at TEXT NOT NULL, finished_at TEXT, error TEXT)""")
        conn.execute("""INSERT INTO deliveries
            (execution_id, job_json, content, status, created_at)
            VALUES ('old', '{"id":"old"}', 'old brief', 'pending', '2026-03-08T06:00:00+00:00')""")
    send = Mock(return_value=None)
    assert queue.drain(send) == 1
    send.assert_called_once_with({"id": "old"}, "old brief", False)
    assert queue.get_status("old")["not_before"] is None
