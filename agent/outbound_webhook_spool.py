"""Optional, profile-local bounded SQLite spool for outbound lifecycle hooks.

This is deliberately separate from the gateway message delivery ledger. A successful HTTP
request can precede the acknowledgement transaction; receivers MUST dedupe delivery_id.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import stat
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
MAX_PENDING = 256
MAX_TERMINAL = 128
MAX_BODY_BYTES = 64 * 1024
MAX_ATTEMPTS = 5
RETENTION_SECONDS = 24 * 60 * 60
LEASE_SECONDS = 90  # exceeds the largest configured HTTP timeout (60s)
MAX_DB_PAGES = 8192  # 32 MiB at the explicitly selected 4096-byte page size
SCHEMA_VERSION = 1
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")


def target_key(target: Any) -> str:
    """Bind records to the exact configured endpoint, event set and matcher.

    Never store the signing key or URL. A URL edit changes the key; signing-key
    rotation is rejected when the worker verifies the original HMAC before POST.
    """
    identity = json.dumps([target.url, target.events, target.matcher], sort_keys=True)
    return hashlib.sha256(identity.encode()).hexdigest()


class Spool:
    def __init__(self, home: str | Path):
        self.directory = Path(home) / "outbound-webhooks"
        self.path = self.directory / "spool.db"
        self.scope = hashlib.sha256(str(Path(home).resolve()).encode()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        # ACLs cannot be made private with chmod on Windows. Fail closed there instead of
        # claiming confidentiality while inheriting a permissive home ACL.
        if os.name == "nt":
            raise PermissionError("durable outbound webhooks require a private filesystem ACL (unsupported on Windows)")
        try:
            self.directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        folder = self.directory.lstat()
        if not stat.S_ISDIR(folder.st_mode) or folder.st_uid != os.getuid() or folder.st_mode & 0o077:  # windows-footgun: ok — POSIX-only after nt guard
            raise PermissionError("outbound webhook spool directory is not private")
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        dbstat = self.path.lstat()
        if not stat.S_ISREG(dbstat.st_mode) or dbstat.st_uid != os.getuid() or dbstat.st_mode & 0o077:  # windows-footgun: ok — POSIX-only after nt guard
            raise PermissionError("outbound webhook spool file is not private")
        conn = sqlite3.connect(self.path, timeout=0.5, isolation_level=None)
        try:
            conn.execute("PRAGMA page_size=4096")
            if conn.execute("PRAGMA page_size").fetchone()[0] != 4096:
                raise ValueError("outbound webhook spool requires 4096-byte pages")
            if conn.execute(f"PRAGMA max_page_count={MAX_DB_PAGES}").fetchone()[0] > MAX_DB_PAGES:
                raise ValueError("outbound webhook spool exceeds disk cap")
            if conn.execute("PRAGMA journal_mode").fetchone()[0] != "delete":
                raise ValueError("outbound webhook spool requires bounded rollback journaling")
            conn.execute("PRAGMA synchronous=FULL")
            if conn.execute("PRAGMA secure_delete=ON").fetchone()[0] != 1:
                raise ValueError("outbound webhook spool requires secure deletion")
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise ValueError("unsupported outbound webhook spool schema")
            if version == 0 and conn.execute("SELECT 1 FROM sqlite_master WHERE name='deliveries'").fetchone():
                raise ValueError("unversioned outbound webhook spool schema")
            conn.execute("""CREATE TABLE IF NOT EXISTS deliveries (
                delivery_id TEXT PRIMARY KEY, target_key TEXT NOT NULL, scope TEXT NOT NULL,
                event TEXT NOT NULL,
                body BLOB, body_sha256 TEXT, signature TEXT,
                created REAL NOT NULL, next_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'pending',
                claim TEXT, lease_until REAL, reason TEXT)""")
            conn.execute("CREATE INDEX IF NOT EXISTS due_deliveries ON deliveries(state, next_at)")
            if version == 0:
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
            return conn
        except BaseException:
            conn.rollback()
            conn.close()
            raise

    @staticmethod
    def _prune(conn: sqlite3.Connection, now: float) -> None:
        conn.execute("DELETE FROM deliveries WHERE created < ?", (now - RETENTION_SECONDS,))

    @staticmethod
    def _prune_terminal(conn: sqlite3.Connection) -> None:
        conn.execute("""DELETE FROM deliveries WHERE delivery_id IN
            (SELECT delivery_id FROM deliveries WHERE state IN ('dead', 'delivered')
             ORDER BY created DESC LIMIT -1 OFFSET ?)""", (MAX_TERMINAL,))

    def enqueue(self, target: Any, delivery: dict[str, Any]) -> bool:
        """Commit before returning success. Disk errors are surfaced as a warning, never a send."""
        try:
            body = delivery["body"]
            identity = delivery["headers"]["X-Hermes-Delivery"]
            if (not isinstance(body, bytes) or len(body) > MAX_BODY_BYTES or
                    not isinstance(identity, str) or not _HEX32.fullmatch(identity) or
                    not isinstance(json.loads(body), dict)):
                logger.warning("outbound webhook durable admission refused: invalid or oversized payload")
                return False
            with contextlib.closing(self._connect()) as conn:
                now = time.time()
                conn.execute("BEGIN IMMEDIATE")
                with conn:
                    self._prune(conn, now)
                    self._prune_terminal(conn)
                    count = conn.execute("SELECT count(*) FROM deliveries WHERE state IN ('pending', 'claimed')").fetchone()[0]
                    if count >= MAX_PENDING:
                        logger.warning("outbound webhook durable spool full (%d pending); event dropped", MAX_PENDING)
                        return False
                    conn.execute("""INSERT INTO deliveries
                        (delivery_id, target_key, scope, event, body, body_sha256, signature, created, next_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                        identity, target_key(target), self.scope, delivery["event"], body,
                        hashlib.sha256(body).hexdigest(),
                        delivery["headers"].get("X-Hermes-Signature-256"), now, now,
                    ))
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as exc:
            logger.warning("outbound webhook durable admission failed (%s); event dropped", type(exc).__name__)
            return False

    def inspect(self) -> list[dict[str, Any]]:
        """Secret-free metadata only; never return retained payloads or signatures."""
        if not self.path.exists():
            return []
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute("SELECT delivery_id, event, state, attempts, created, reason FROM deliveries ORDER BY created").fetchall()
        return [dict(zip(("delivery_id", "event", "state", "attempts", "created", "reason"), row)) for row in rows]

    def _claim(self, now: float) -> tuple | None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._prune(conn, now)
                row = conn.execute("""SELECT delivery_id, target_key, scope, event, body, body_sha256, signature, attempts
                    FROM deliveries WHERE (state='pending' AND next_at <= ?)
                    OR (state='claimed' AND lease_until <= ?) ORDER BY created LIMIT 1""", (now, now)).fetchone()
                if row is None:
                    conn.commit()
                    return None
                claim = uuid.uuid4().hex
                conn.execute("""UPDATE deliveries SET state='claimed', claim=?, lease_until=?, attempts=attempts+1
                    WHERE delivery_id=?""", (claim, now + LEASE_SECONDS, row[0]))
                conn.commit()
                return (*row, claim)
            except BaseException:
                conn.rollback()
                raise

    def _finish(self, identity: str, claim: str, outcome: str, reason: str, now: float, attempts: int) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            with conn:
                if outcome == "retry" and attempts < MAX_ATTEMPTS:
                    conn.execute("""UPDATE deliveries SET state='pending', claim=NULL, lease_until=NULL,
                        next_at=?, reason=? WHERE delivery_id=? AND claim=?""",
                        (now + min(60, 2 ** (attempts - 1)), reason, identity, claim))
                else:
                    state = "delivered" if outcome == "ok" else "dead"
                    conn.execute("""UPDATE deliveries SET state=?, body=NULL, body_sha256=NULL, signature=NULL, claim=NULL,
                        lease_until=NULL, reason=? WHERE delivery_id=? AND claim=?""",
                        (state, reason, identity, claim))
                    self._prune_terminal(conn)

    def process(self, targets: dict[str, Any], *, now: float | None = None) -> bool:
        """Claim one row atomically across processes; a crashed claimant expires after the lease.

        No fallback to volatile delivery on a storage error: it could duplicate a send already
        completed before the error. A later worker may reclaim the lease instead.
        """
        now = time.time() if now is None else now
        try:
            row = self._claim(now)
        except (OSError, sqlite3.Error, ValueError) as exc:
            logger.warning("outbound webhook spool claim failed (%s)", type(exc).__name__)
            return False
        if row is None:
            return False
        identity, key, scope, event, body, body_sha256, signature, attempts, claim = row
        target = targets.get(key)
        outcome, reason = "terminal", "target_changed_or_removed"
        if attempts >= MAX_ATTEMPTS:
            outcome, reason = "terminal", "attempts_exhausted"
        elif scope != self.scope:
            outcome, reason = "terminal", "profile_mismatch"
        elif target is not None:
            try:
                if not isinstance(body, bytes) or len(body) > MAX_BODY_BYTES:
                    raise ValueError("invalid record")
                parsed = json.loads(body)
                expected = ("sha256=" + hmac.new(target.secret.encode(), body, hashlib.sha256).hexdigest()
                            if target.secret else None)
                if (not isinstance(parsed, dict) or
                        not hmac.compare_digest(hashlib.sha256(body).hexdigest(), body_sha256 or "") or
                        parsed.get("delivery_id") != identity or
                        parsed.get("hook_event_name") != event or
                        not isinstance(parsed.get("timestamp"), str) or
                        not isinstance(parsed.get("profile"), str) or
                        not isinstance(parsed.get("extra"), dict) or
                        not isinstance(identity, str) or not _HEX32.fullmatch(identity) or
                        (signature is not None and not isinstance(signature, str)) or
                        not hmac.compare_digest(signature or "", expected or "")):
                    raise ValueError("invalid record")
                from agent.outbound_webhooks import _build_delivery, _post_once
                delivery = _build_delivery(event, target, body, identity)
                outcome, reason = _post_once(delivery)
            except (ValueError, TypeError, UnicodeError, KeyError):
                outcome, reason = "terminal", "invalid_record"
            except (OSError, RuntimeError) as exc:
                outcome, reason = "retry", type(exc).__name__
        try:
            self._finish(identity, claim, outcome, reason, now, attempts + 1)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("outbound webhook spool acknowledgement failed (%s); duplicate possible", type(exc).__name__)
        return True
