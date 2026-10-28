"""Source-side atomic Kanban reply receipt, for a trusted owner-scoped bridge.

This is not an HTTP endpoint or an owner resolver. The caller must authenticate the
installation/owner and supply an authorization callback backed by CURRENT native
owner/profile binding; without one the write refuses. Never expose this function to
browser JSON or use the generic dashboard comment route for retries.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MAX_BODY_BYTES = 16_384
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_BOARD = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class ReplyConflict(ValueError):
    """Revision, identity, or idempotency collision (HTTP 409 at the bridge)."""


class ReplyUnauthorized(PermissionError):
    """Current native owner/profile authorization failed (HTTP 403)."""


@dataclass(frozen=True)
class ReplyScope:
    # These values MUST come from an authenticated server-side owner/context issuer.
    installation: str
    owner: str
    owner_incarnation: str
    profile: str
    board: str
    board_generation: str
    task_id: str
    task_generation: str
    thread_id: str


@dataclass(frozen=True)
class ReplyReceipt:
    native_namespace: str
    target_kind: str
    target_id: str
    target_generation: str
    native_reply_id: int
    native_revision: int
    idempotency_key: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS native_task_reply_board_identity (
 singleton INTEGER PRIMARY KEY CHECK(singleton = 1), generation TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS native_task_reply_generations (
 task_id TEXT PRIMARY KEY, generation TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS native_task_reply_deleted AFTER DELETE ON tasks
BEGIN
 UPDATE native_task_reply_generations SET generation = lower(hex(randomblob(16)))
 WHERE task_id = old.id;
END;
CREATE TABLE IF NOT EXISTS native_task_reply_receipts (
 installation TEXT NOT NULL, owner_incarnation TEXT NOT NULL,
 thread_id TEXT NOT NULL, expected_revision INTEGER NOT NULL,
 idempotency_key TEXT NOT NULL, body_hash TEXT NOT NULL,
 owner TEXT NOT NULL, profile TEXT NOT NULL, board TEXT NOT NULL,
 board_generation TEXT NOT NULL,
 task_id TEXT NOT NULL, task_generation TEXT NOT NULL,
 native_reply_id INTEGER NOT NULL, native_revision INTEGER NOT NULL,
 PRIMARY KEY (installation, owner_incarnation, thread_id, expected_revision, idempotency_key)
);
CREATE UNIQUE INDEX IF NOT EXISTS native_task_reply_unique_comment
 ON native_task_reply_receipts(native_reply_id);
"""


def install_schema(conn: sqlite3.Connection) -> None:
    """Install the scoped ledger and deletion tombstone on an initialized board."""
    conn.executescript(_SCHEMA)


def board_generation(conn: sqlite3.Connection) -> str:
    """Issue an incarnation in the physical board DB, never a display slug."""
    conn.execute("INSERT OR IGNORE INTO native_task_reply_board_identity VALUES (1, ?)", (uuid.uuid4().hex,))
    return conn.execute("SELECT generation FROM native_task_reply_board_identity WHERE singleton=1").fetchone()[0]


def _require_scope(scope: ReplyScope) -> None:
    if not isinstance(scope, ReplyScope) or not _BOARD.fullmatch(scope.board):
        raise ValueError("explicit canonical board slug required")
    for name in ("installation", "owner", "owner_incarnation", "profile", "board_generation", "task_id", "task_generation", "thread_id"):
        value = getattr(scope, name)
        if not isinstance(value, str) or not value or len(value) > 256:
            raise ValueError(f"invalid {name}")


def current_generation(conn: sqlite3.Connection, task_id: str) -> str:
    """Issue a generation only after caller has verified the board and owner.

    A trigger tombstones existing generations on task deletion; an old token never
    matches a recreated task even if its ID and creation timestamp are recycled.
    """
    row = conn.execute("SELECT id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise ReplyConflict("task missing")
    row = conn.execute(
        "SELECT generation FROM native_task_reply_generations WHERE task_id = ?", (task_id,)
    ).fetchone()
    if row is None:
        generation = uuid.uuid4().hex
        conn.execute(
            "INSERT OR IGNORE INTO native_task_reply_generations(task_id, generation) VALUES (?, ?)",
            (task_id, generation),
        )
        row = conn.execute(
            "SELECT generation FROM native_task_reply_generations WHERE task_id = ?", (task_id,)
        ).fetchone()
    return row[0]


def _revision(conn: sqlite3.Connection, task_id: str) -> int:
    # All native comment writers contribute to the revision, not just this bridge.
    return int(conn.execute(
        "SELECT COALESCE(MAX(id), 0) FROM task_comments WHERE task_id = ?", (task_id,)
    ).fetchone()[0])


def reply(
    conn: sqlite3.Connection, *, scope: ReplyScope, expected_revision: int,
    idempotency_key: str, body: str, canonical_board: str, canonical_db_path: Path,
    authorize: Callable[[ReplyScope], bool] | None,
) -> ReplyReceipt:
    """Atomically authorize, reconcile or insert an exact-byte comment and receipt.

    ``canonical_board`` and ``canonical_db_path`` come from the authenticated
    installation board registry, NOT the request or ambient board pointer.
    Callback must re-read current owner/incarnation, profile and source grant.
    An unknown response after commit can be reconciled by the identical call.
    """
    _require_scope(scope)
    if canonical_board != scope.board:
        raise ReplyUnauthorized("foreign board")
    main = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), None)
    if not main or Path(main).resolve() != canonical_db_path.resolve():
        raise ReplyUnauthorized("board DB is not the attested source")
    if authorize is None:
        raise ReplyUnauthorized("native owner resolver unavailable")
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise ValueError("invalid expected_revision")
    if not isinstance(idempotency_key, str) or not _KEY.fullmatch(idempotency_key):
        raise ValueError("invalid idempotency_key")
    if not isinstance(body, str) or not body.strip():
        raise ValueError("comment body is required")
    exact = body.encode("utf-8")
    if len(exact) > MAX_BODY_BYTES:
        raise ValueError("comment exceeds 16384 UTF-8 bytes")
    digest = hashlib.sha256(exact).hexdigest()
    if conn.in_transaction:
        raise ValueError("reply requires its own immediate transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        native_board = conn.execute(
            "SELECT generation FROM native_task_reply_board_identity WHERE singleton=1"
        ).fetchone()
        if native_board is None or native_board[0] != scope.board_generation:
            raise ReplyConflict("board generation changed")
        task = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (scope.task_id,)).fetchone()
        if task is None:
            raise ReplyConflict("task missing")
        generation = conn.execute(
            "SELECT generation FROM native_task_reply_generations WHERE task_id = ?", (scope.task_id,)
        ).fetchone()
        if generation is None or generation[0] != scope.task_generation:
            raise ReplyConflict("task generation changed")
        if task[0] != scope.profile or not authorize(scope):
            raise ReplyUnauthorized("current owner/profile binding denied")
        key = (scope.installation, scope.owner_incarnation, scope.thread_id, expected_revision, idempotency_key)
        prior = conn.execute(
            "SELECT body_hash, owner, profile, board, board_generation, task_id, task_generation, native_reply_id, native_revision "
            "FROM native_task_reply_receipts WHERE installation=? AND owner_incarnation=? "
            "AND thread_id=? AND expected_revision=? AND idempotency_key=?", key,
        ).fetchone()
        if prior is not None:
            if tuple(prior[:7]) != (digest, scope.owner, scope.profile, scope.board, scope.board_generation, scope.task_id, scope.task_generation):
                raise ReplyConflict("idempotency key reused with different identity or bytes")
            receipt = ReplyReceipt(scope.board, "task", scope.task_id, scope.task_generation, prior[7], prior[8], idempotency_key)
        else:
            collision = conn.execute(
                "SELECT 1 FROM native_task_reply_receipts WHERE installation=? AND owner_incarnation=? "
                "AND thread_id=? AND idempotency_key=?", (scope.installation, scope.owner_incarnation, scope.thread_id, idempotency_key),
            ).fetchone()
            if collision:
                raise ReplyConflict("idempotency key reused with different revision")
            if _revision(conn, scope.task_id) != expected_revision:
                raise ReplyConflict("stale native comment revision")
            # Preserve exact bytes; kanban_db.add_comment strips, so it is not a safe reply primitive.
            cursor = conn.execute(
                "INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, ?, ?, unixepoch())",
                (scope.task_id, scope.profile, body),
            )
            if cursor.lastrowid is None:
                raise sqlite3.DatabaseError("native comment insert has no row ID")
            reply_id = int(cursor.lastrowid)
            conn.execute(
                "INSERT INTO task_events(task_id, kind, payload, created_at) "
                "VALUES (?, 'commented', ?, unixepoch())",
                (scope.task_id, '{"source":"native_task_reply"}'),
            )
            conn.execute(
                "INSERT INTO native_task_reply_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*key, digest, scope.owner, scope.profile, scope.board, scope.board_generation,
                 scope.task_id, scope.task_generation, reply_id, reply_id),
            )
            receipt = ReplyReceipt(scope.board, "task", scope.task_id, scope.task_generation, reply_id, reply_id, idempotency_key)
        conn.commit()
        return receipt
    except BaseException:
        conn.rollback()
        raise
