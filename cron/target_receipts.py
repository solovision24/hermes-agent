"""Opt-in per-destination cron receipts in the existing profile-local delivery DB.

This is a storage/release primitive, not a quiet-hours policy or a scheduler hook.
A caller must provide concrete, already-resolved targets and a release instant. The
legacy aggregate queue and every default cron delivery remain unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from agent.redact import redact_sensitive_text
from cron.executions import _owner_is_live, _process_start_time
from hermes_time import now as _now

logger = logging.getLogger(__name__)
MAX_PENDING_TARGETS = 100
MAX_TARGETS_PER_EXECUTION = 32
MAX_PAYLOAD_BYTES = 65536
MAX_TERMINAL_TARGETS = 1000
MAX_TOMBSTONES = 10000
RECEIPT_RETENTION = timedelta(days=30)
MAX_DEFERRAL = timedelta(days=7)
_TERMINAL = ("delivered", "failed", "unknown", "suppressed")
_PROCESS_ID = uuid.uuid4().hex
_ACTIVE: set[tuple[str, str, str]] = set()


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Called from the aggregate queue's existing DB initializer."""
    conn.execute("""CREATE TABLE IF NOT EXISTS target_deliveries (
        execution_id TEXT NOT NULL,
        target_key TEXT NOT NULL,
        job_id TEXT NOT NULL,
        target_json TEXT NOT NULL,
        kind TEXT NOT NULL CHECK(kind IN ('human','machine')),
        content TEXT NOT NULL,
        for_failure INTEGER NOT NULL,
        not_before TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN
            ('pending','delivering','delivered','failed','unknown','suppressed')),
        owner_process_id TEXT,
        owner_pid INTEGER,
        owner_started_at INTEGER,
        created_at TEXT NOT NULL,
        finished_at TEXT,
        error TEXT,
        PRIMARY KEY (execution_id, target_key)
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS target_delivery_tombstones (
        execution_id TEXT NOT NULL,
        target_key TEXT NOT NULL,
        status TEXT NOT NULL,
        finished_at TEXT NOT NULL,
        PRIMARY KEY (execution_id, target_key)
    )""")
    from hermes_cli.sqlite_util import add_column_if_missing
    add_column_if_missing(conn, "target_deliveries", "owner_process_id", "owner_process_id TEXT")


def _iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("delivery instant must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _identity(target: dict) -> tuple[str, str]:
    """Use only a concrete destination; routing tokens or mutable home aliases are unsafe."""
    kind = target.get("kind")
    if kind not in ("human", "machine"):
        raise ValueError("target kind must be human or machine")
    platform, chat_id = target.get("platform"), target.get("chat_id")
    if not isinstance(platform, str) or not platform or not isinstance(chat_id, str) or not chat_id:
        raise ValueError("target requires concrete platform and chat_id")
    thread_id = target.get("thread_id")
    if thread_id is not None and not isinstance(thread_id, str):
        raise ValueError("target thread_id must be a string or null")
    # Preserve transport-specific routing metadata in the snapshot, but prevent
    # two logically identical destinations from having different receipt keys.
    key = json.dumps([platform.lower(), chat_id, thread_id], separators=(",", ":"))
    snapshot = json.dumps(target, sort_keys=True, ensure_ascii=False)
    if len(snapshot.encode("utf-8")) > 8192:
        raise ValueError("target snapshot exceeds size limit")
    return key, snapshot


def _prune(conn: sqlite3.Connection, now: datetime) -> None:
    cutoff = _iso(now - RECEIPT_RETENTION)
    conn.execute("DELETE FROM target_delivery_tombstones WHERE finished_at < ?", (cutoff,))
    conn.execute("""UPDATE target_deliveries SET content='', target_json='{}'
        WHERE status IN ('delivered','failed','unknown','suppressed')
          AND (content != '' OR target_json != '{}')""")
    excess = conn.execute("""SELECT COUNT(*) FROM target_deliveries
        WHERE status IN ('delivered','failed','unknown','suppressed')""").fetchone()[0] - MAX_TERMINAL_TARGETS
    if excess > 0:
        rows = conn.execute("""SELECT execution_id,target_key,status,finished_at
            FROM target_deliveries WHERE status IN ('delivered','failed','unknown','suppressed')
            ORDER BY finished_at,execution_id,target_key LIMIT ?""", (excess,)).fetchall()
        conn.executemany("""INSERT OR IGNORE INTO target_delivery_tombstones
            (execution_id,target_key,status,finished_at) VALUES (?,?,?,?)""",
            [(r["execution_id"], r["target_key"], r["status"], r["finished_at"]) for r in rows])
        conn.executemany("DELETE FROM target_deliveries WHERE execution_id=? AND target_key=?",
                         [(r["execution_id"], r["target_key"]) for r in rows])
    tombstone_excess = conn.execute("SELECT COUNT(*) FROM target_delivery_tombstones").fetchone()[0] - MAX_TOMBSTONES
    if tombstone_excess > 0:
        conn.execute("""DELETE FROM target_delivery_tombstones WHERE (execution_id,target_key) IN
            (SELECT execution_id,target_key FROM target_delivery_tombstones
             ORDER BY finished_at,execution_id,target_key LIMIT ?)""", (tombstone_excess,))


def enqueue_targets(execution_id: str, job_id: str, content: str, targets: list[dict], *,
                    not_before: datetime, expires_at: datetime,
                    for_failure: bool = False) -> list[dict]:
    """Atomically snapshot all destinations or reject the entire batch.

    Retried calls return the original rows, never mutate an existing snapshot.
    Idempotency survives terminal pruning for up to 30 days / 10,000 tombstones.
    """
    if not execution_id or not job_id or not targets or len(targets) > MAX_TARGETS_PER_EXECUTION:
        raise ValueError("execution, job, and 1..32 targets are required")
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("delivery content exceeds size limit")
    start, end = _iso(not_before), _iso(expires_at)
    if not start < end or expires_at.astimezone(UTC) - not_before.astimezone(UTC) > MAX_DEFERRAL:
        raise ValueError("expiry must follow release within seven days")
    identities = [_identity(t) for t in targets]
    if len({key for key, _ in identities}) != len(identities):
        raise ValueError("duplicate concrete target")
    from cron.delivery_queue import _transaction
    with _transaction() as conn:
        conn.execute("BEGIN IMMEDIATE")
        now = _now()
        _prune(conn, now)
        prior = conn.execute("""SELECT target_key FROM target_deliveries WHERE execution_id=?
            UNION SELECT target_key FROM target_delivery_tombstones WHERE execution_id=?""",
            (execution_id, execution_id)).fetchall()
        keys = {key for key, _ in identities}
        if prior:
            if {r["target_key"] for r in prior} != keys:
                raise ValueError("execution already has different target receipts")
            existing = get_target_receipts(execution_id, conn=conn)
            for row in existing:
                if row.get("status") not in _TERMINAL and (
                    row["job_id"] != job_id or row["content"] != content
                    or row["target_json"] != dict(identities)[row["target_key"]]
                    or bool(row["for_failure"]) != bool(for_failure)
                ):
                    raise ValueError("execution already has a different delivery payload")
            return existing
        pending = conn.execute("""SELECT COUNT(*) FROM target_deliveries
            WHERE status IN ('pending','delivering')""").fetchone()[0]
        if pending + len(targets) > MAX_PENDING_TARGETS:
            raise ValueError("target delivery queue at capacity; nothing was queued")
        conn.executemany("""INSERT INTO target_deliveries
            (execution_id,target_key,job_id,target_json,kind,content,for_failure,
             not_before,expires_at,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,'pending',?)""", [
            (execution_id, key, job_id, snapshot, target["kind"], content,
             int(for_failure), start, end, _iso(now))
            for target, (key, snapshot) in zip(targets, identities)
        ])
        return get_target_receipts(execution_id, conn=conn)


def get_target_receipts(execution_id: str, *, conn: Optional[sqlite3.Connection] = None) -> list[dict]:
    """Return independently truthful per-target statuses, including retained tombstones."""
    if conn is None:
        from cron.delivery_queue import _transaction
        with _transaction() as opened:
            return get_target_receipts(execution_id, conn=opened)
    rows = conn.execute("SELECT * FROM target_deliveries WHERE execution_id=? ORDER BY target_key",
                        (execution_id,)).fetchall()
    old = conn.execute("""SELECT execution_id,target_key,status,finished_at
        FROM target_delivery_tombstones WHERE execution_id=? ORDER BY target_key""",
        (execution_id,)).fetchall()
    return [dict(row) for row in (*rows, *old)]


def _default_job_state(job_id: str) -> str:
    from cron.jobs import get_job
    job = get_job(job_id)
    return "deleted" if job is None else "active" if job.get("enabled", True) else "paused"


def recover_targets(*, db_path: Optional[Path] = None) -> int:
    """A dead owner may have sent; mark only its claimed destination unknown."""
    from cron.delivery_queue import _transaction, queue_path
    path = db_path if db_path is not None else queue_path()
    with _transaction(path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("""SELECT execution_id,target_key,owner_process_id,owner_pid,owner_started_at
            FROM target_deliveries WHERE status='delivering'""").fetchall()
        changed = 0
        for row in rows:
            pid = row["owner_pid"]
            same_process = row["owner_process_id"] == _PROCESS_ID
            if same_process and (str(path), row["execution_id"], row["target_key"]) in _ACTIVE:
                continue
            if not same_process and pid is not None and _owner_is_live(int(pid), row["owner_started_at"]):
                continue
            conn.execute("""UPDATE target_deliveries SET status='unknown',finished_at=?,
                error='gateway exited during target send; outcome unknown, not retried'
                WHERE execution_id=? AND target_key=? AND status='delivering'""",
                (_iso(_now()), row["execution_id"], row["target_key"]))
            changed += 1
        _prune(conn, _now())
        return changed


def release_targets(send: Callable[[dict, str, bool], tuple[str, Optional[str]]], *,
                    job_state: Callable[[str], str] = _default_job_state,
                    limit: int = 20) -> int:
    """Release due rows once, with explicit transport evidence per destination.

    send(target, content, for_failure) returns (delivered|failed|unknown, error).
    An exception or missing confirmation is unknown, NEVER success and NEVER retried.
    Paused jobs remain pending; deleted jobs are suppressed without sending.
    A pause racing after claim cannot retract an already-started send.
    """
    from cron.delivery_queue import _transaction, queue_path
    path = queue_path()
    recover_targets(db_path=path)
    processed = 0
    for _ in range(max(0, limit)):
        state = "paused"
        with _transaction(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            now = _iso(_now())
            candidates = conn.execute("""SELECT * FROM target_deliveries WHERE status='pending'
                AND (not_before <= ? OR expires_at <= ?)
                ORDER BY not_before,created_at,execution_id,target_key LIMIT 100""",
                (now, now)).fetchall()
            selected = None
            for row in candidates:
                # Job lookup must use the same owning profile as this queue connection.
                state = job_state(row["job_id"])
                if state not in ("active", "paused", "deleted"):
                    raise ValueError("job state must be active, paused, or deleted")
                if state == "paused" and row["expires_at"] > now:
                    continue
                selected = row
                if state == "deleted" or row["expires_at"] <= now:
                    status = "suppressed" if state == "deleted" else "failed"
                    error = "job deleted before delivery" if state == "deleted" else "target delivery expired before send"
                    conn.execute("""UPDATE target_deliveries SET status=?,finished_at=?,error=?,
                        content='',target_json='{}'
                        WHERE execution_id=? AND target_key=? AND status='pending'""",
                        (status, now, error, row["execution_id"], row["target_key"]))
                    _prune(conn, _now())
                else:
                    conn.execute("""UPDATE target_deliveries SET status='delivering',
                        owner_process_id=?,owner_pid=?,owner_started_at=?
                        WHERE execution_id=? AND target_key=?
                        AND status='pending'""",
                        (_PROCESS_ID, os.getpid(), _process_start_time(os.getpid()),
                         row["execution_id"], row["target_key"]))
                    _ACTIVE.add((str(path), row["execution_id"], row["target_key"]))
                break
            if selected is None:
                break
        if state == "active" and selected["expires_at"] > now:
            try:
                outcome = send(json.loads(selected["target_json"]), selected["content"],
                               bool(selected["for_failure"]))
                status, error = outcome if isinstance(outcome, tuple) and len(outcome) == 2 else ("unknown", "no delivery confirmation")
                if status not in ("delivered", "failed", "unknown"):
                    status, error = "unknown", "unsupported delivery confirmation"
            except BaseException as exc:
                logger.warning("Cron target send returned an ambiguous outcome", exc_info=True)
                status, error = "unknown", f"{type(exc).__name__}: {exc}"
            try:
                safe_error = (redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
                              if error else None)
                with _transaction(path) as conn:
                    conn.execute("""UPDATE target_deliveries SET status=?,error=?,finished_at=?
                        WHERE execution_id=? AND target_key=? AND status='delivering'
                        AND owner_process_id=? AND owner_pid=?""",
                        (status, safe_error, _iso(_now()), selected["execution_id"],
                         selected["target_key"], _PROCESS_ID, os.getpid()))
                    _prune(conn, _now())
            finally:
                _ACTIVE.discard((str(path), selected["execution_id"], selected["target_key"]))
        processed += 1
    return processed
