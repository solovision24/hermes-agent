"""Real SQLite transaction and identity tests for the native task reply primitive."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from hermes_cli.native_task_replies import (
    ReplyConflict, ReplyScope, ReplyUnauthorized, current_generation, install_schema, reply,
)


def _connection(path):
    conn = sqlite3.connect(path, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, assignee TEXT, created_at INTEGER);
        CREATE TABLE IF NOT EXISTS task_comments (
          id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, author TEXT, body TEXT, created_at INTEGER);
        CREATE TABLE IF NOT EXISTS task_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, kind TEXT, payload TEXT, created_at INTEGER);
    """)
    install_schema(conn)
    return conn


@pytest.fixture
def board(tmp_path):
    path = tmp_path / "kanban.db"
    with _connection(path) as conn:
        conn.execute("INSERT INTO tasks VALUES ('t_one', 'chip', 1)")
        generation = current_generation(conn, "t_one")
    return path, ReplyScope("install-1", "owner-1", "inc-1", "chip", "default", "t_one", generation, "thread-1")


def _send(path, scope, *, body="hello", revision=0, key="key-1", authorize=None, board="default"):
    with _connection(path) as conn:
        return reply(conn, scope=scope, body=body, expected_revision=revision,
                     idempotency_key=key, canonical_board=board,
                     authorize=authorize if authorize is not None else lambda _: True)


def test_retry_and_response_loss_reconcile_same_receipt(board):
    path, scope = board
    first = _send(path, scope)
    second = _send(path, scope)
    assert first == second
    with _connection(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 1
        assert conn.execute("SELECT body FROM task_comments").fetchone()[0] == "hello"
    with pytest.raises(ReplyConflict):
        _send(path, scope, body="changed")
    with pytest.raises(ReplyConflict):
        _send(path, scope, revision=0, key="key-2")
    with pytest.raises(ReplyConflict):
        _send(path, scope, revision=1, key="key-1")


def test_concurrent_same_key_one_comment(board):
    path, scope = board
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _: _send(path, scope), range(8)))
    assert all(item == receipts[0] for item in receipts)
    with _connection(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 1


def test_foreign_owner_profile_board_and_reassignment(board):
    path, scope = board
    with pytest.raises(ReplyUnauthorized):
        _send(path, scope, authorize=lambda _: False)
    with pytest.raises(ReplyUnauthorized):
        _send(path, scope, board="foreign")
    with pytest.raises(ReplyUnauthorized):
        _send(path, ReplyScope(**{**scope.__dict__, "profile": "forge"}))
    with _connection(path) as conn:
        conn.execute("UPDATE tasks SET assignee='forge' WHERE id='t_one'")
    with pytest.raises(ReplyUnauthorized):
        _send(path, scope)


def test_recreation_same_id_and_timestamp_invalidates_receipt(board):
    path, scope = board
    old = _send(path, scope)
    with _connection(path) as conn:
        conn.execute("DELETE FROM tasks WHERE id='t_one'")
        conn.execute("INSERT INTO tasks VALUES ('t_one', 'chip', 1)")
        new_generation = current_generation(conn, "t_one")
    assert new_generation != scope.task_generation
    with pytest.raises(ReplyConflict):
        _send(path, scope)
    with _connection(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 1
    assert old.native_reply_id > 0


def test_partial_rollback_and_external_comment_revision(board):
    path, scope = board
    with _connection(path) as conn:
        conn.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON native_task_reply_receipts "
                     "BEGIN SELECT RAISE(ABORT, 'receipt write failed'); END")
    with pytest.raises(sqlite3.DatabaseError):
        _send(path, scope)
    with _connection(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM task_comments").fetchone()[0] == 0
        conn.execute("DROP TRIGGER fail_receipt")
        conn.execute("INSERT INTO task_comments(task_id,author,body,created_at) VALUES ('t_one','chip','other',0)")
    with pytest.raises(ReplyConflict):
        _send(path, scope)
    assert _send(path, scope, revision=1).native_revision > 1


def test_missing_resolver_and_body_bound(board):
    path, scope = board
    with _connection(path) as conn, pytest.raises(ReplyUnauthorized):
        reply(conn, scope=scope, body="hello", expected_revision=0,
              idempotency_key="key-1", canonical_board="default", authorize=None)
    with pytest.raises(ValueError, match="16384"):
        _send(path, scope, body="é" * 8193)


def test_real_native_board_schema_commit_and_reopen(tmp_path):
    from hermes_cli import kanban_db_connect

    path = tmp_path / "native-board.db"
    with kanban_db_connect.connect_closing(db_path=path) as conn:
        conn.execute("INSERT INTO tasks(id,title,status,created_at,assignee) "
                     "VALUES ('t_real','real task','ready',1,'chip')")
        conn.commit()
        install_schema(conn)
        generation = current_generation(conn, "t_real")
        conn.commit()
        scope = ReplyScope("install-1", "owner-1", "inc-1", "chip", "default", "t_real", generation, "thread-1")
        first = reply(conn, scope=scope, body="  exact bytes  ", expected_revision=0,
                      idempotency_key="idem", canonical_board="default", authorize=lambda _: True)
    with kanban_db_connect.connect_closing(db_path=path) as conn:
        second = reply(conn, scope=scope, body="  exact bytes  ", expected_revision=0,
                       idempotency_key="idem", canonical_board="default", authorize=lambda _: True)
        assert second == first
        assert conn.execute("SELECT body FROM task_comments WHERE id=?", (first.native_reply_id,)).fetchone()[0] == "  exact bytes  "
        assert conn.execute("SELECT kind FROM task_events WHERE task_id='t_real'").fetchone()[0] == "commented"
