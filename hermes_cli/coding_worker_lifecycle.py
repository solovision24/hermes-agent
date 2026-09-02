"""Durable task identity and closeout state for Kanban coding workers."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from hermes_cli import kanban_db as kb


class LifecycleConflict(RuntimeError):
    """A retry disagreed with identity already assigned to the task."""


_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?$")


def canonical_pr_url(value: str) -> str:
    """Return a credential-free canonical GitHub pull-request URL."""
    parsed = urlsplit(str(value).strip())
    if parsed.scheme.lower() != "https" or parsed.hostname != "github.com":
        raise ValueError("pull request URL must be an https://github.com URL")
    if parsed.username or parsed.password or parsed.port:
        raise ValueError("pull request URL must not contain credentials or a port")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 4 or parts[2] != "pull" or not parts[3].isdigit():
        raise ValueError("pull request URL must identify /owner/repo/pull/<number>")
    return urlunsplit(("https", "github.com", "/" + "/".join(parts), "", ""))


def canonical_head_sha(value: str) -> str:
    """Require an immutable full Git object id and normalize its case."""
    head = str(value).strip().lower()
    if not _SHA_RE.fullmatch(head):
        raise ValueError("head_sha must be a full 40- or 64-character hex digest")
    return head


@dataclass(frozen=True)
class CodingWorkerState:
    task_id: str
    workspace_path: str
    lease_token: Optional[str]
    lease_expires: Optional[int]
    phase: str
    pr_url: Optional[str]
    head_sha: Optional[str]
    review_submitted_at: Optional[int]
    wait_kind: Optional[str]
    operation_key: Optional[str]
    wait_started_at: Optional[int]
    wait_deadline: Optional[int]
    silent_checks: int
    operation_receipt: Optional[str]
    last_error: Optional[str]
    updated_at: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "CodingWorkerState":
        return cls(**{name: row[name] for name in cls.__dataclass_fields__})


def get_state(conn: sqlite3.Connection, task_id: str) -> Optional[CodingWorkerState]:
    row = conn.execute(
        "SELECT * FROM coding_worker_lifecycle WHERE task_id=?", (task_id,)
    ).fetchone()
    return CodingWorkerState.from_row(row) if row is not None else None


def allocate_workspace(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_path: str | Path,
    *,
    lease_token: str,
    now: int,
    ttl_seconds: int,
) -> CodingWorkerState:
    """Allocate once by task id; retries renew but cannot silently retarget."""
    lease_token = str(lease_token).strip()
    if not lease_token:
        raise ValueError("lease_token is required")
    resolved = str(Path(workspace_path).expanduser().resolve(strict=False))
    expires = int(now) + max(1, int(ttl_seconds))
    with kb.write_txn(conn):
        existing = get_state(conn, task_id)
        if existing is None:
            conn.execute(
                "INSERT INTO coding_worker_lifecycle "
                "(task_id, workspace_path, lease_token, lease_expires, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (task_id, resolved, lease_token, expires, int(now)),
            )
        else:
            if existing.workspace_path != resolved:
                raise LifecycleConflict(
                    f"task {task_id} is already allocated to {existing.workspace_path}"
                )
            if (
                existing.lease_token not in (None, lease_token)
                and (existing.lease_expires or 0) > int(now)
            ):
                raise LifecycleConflict(f"task {task_id} has a live foreign lease")
            conn.execute(
                "UPDATE coding_worker_lifecycle SET lease_token=?, lease_expires=?, "
                "phase=CASE WHEN wait_kind IS NULL AND phase IN "
                "('blocked','changes_requested','crashed','failed','gave_up',"
                "'rate_limited','reclaimed','spawn_failed','stale','timed_out') "
                "THEN 'allocated' ELSE phase END, "
                "updated_at=? WHERE task_id=?",
                (lease_token, expires, int(now), task_id),
            )
        state = get_state(conn, task_id)
    assert state is not None
    return state


def _operation_receipts(state: CodingWorkerState) -> dict:
    try:
        value = json.loads(state.operation_receipt or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def begin_pr_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    lease_token: str,
    operation_key: str,
    head_sha: str,
    now: int,
) -> tuple[CodingWorkerState, bool]:
    """Reserve the immutable PR head before asking the provider to create it.

    A retry receiving ``False`` must reconcile the provider by head instead of
    creating another PR. The eventual URL is bound by ``request_review`` in
    the same transaction as the native Review transition.
    """
    operation_key = str(operation_key).strip()
    if not operation_key:
        raise ValueError("operation_key is required")
    head = canonical_head_sha(head_sha)
    with kb.write_txn(conn):
        state = get_state(conn, task_id)
        if state is None:
            raise LifecycleConflict(f"task {task_id} has no workspace allocation")
        if state.lease_token != lease_token:
            raise LifecycleConflict(f"task {task_id} lease ownership changed")
        if state.head_sha not in (None, head):
            raise LifecycleConflict(
                f"task {task_id} review head is immutable ({state.head_sha})"
            )
        if state.wait_kind is not None:
            if state.wait_kind != "pr" or state.operation_key != operation_key:
                raise LifecycleConflict("a different remote operation is already durable")
            return state, False
        if state.pr_url is not None:
            # Native Changes Requested clears only the reviewed head. The
            # canonical PR remains bound to the task, so this revision must
            # reconcile/update that PR instead of creating another one.
            if state.phase != "allocated" or state.head_sha is not None:
                raise LifecycleConflict("canonical PR is not awaiting a new revision")
            conn.execute(
                "UPDATE coding_worker_lifecycle SET phase='pr_pending', head_sha=?, "
                "wait_kind='pr', operation_key=?, wait_started_at=?, "
                "last_error=NULL, updated_at=? WHERE task_id=?",
                (head, operation_key, int(now), int(now), task_id),
            )
            state = get_state(conn, task_id)
            assert state is not None
            return state, False
        conn.execute(
            "UPDATE coding_worker_lifecycle SET phase='pr_pending', head_sha=?, "
            "wait_kind='pr', operation_key=?, wait_started_at=?, "
            "last_error=NULL, updated_at=? WHERE task_id=?",
            (head, operation_key, int(now), int(now), task_id),
        )
        state = get_state(conn, task_id)
    assert state is not None
    return state, True


def begin_wait(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    kind: str,
    operation_key: str,
    now: int,
    timeout_seconds: int,
    active_chat_count: Optional[int] = None,
) -> tuple[CodingWorkerState, bool]:
    """Persist a bounded merge/deploy operation before its remote side effect.

    ``started`` is true only for the first caller. Retries resume the durable
    operation key and original deadline; completed receipts also remain
    idempotent after the active wait fields have been cleared.
    """
    if kind not in {"merge", "deploy"}:
        raise ValueError("wait kind must be 'merge' or 'deploy'")
    operation_key = str(operation_key).strip()
    if not operation_key:
        raise ValueError("operation_key is required")
    coordination_error: Optional[str] = None
    if kind == "deploy" and active_chat_count is None:
        try:
            from hermes_cli.active_sessions import active_session_registry_snapshot

            active_chat_count = len(active_session_registry_snapshot())
        except Exception as exc:
            active_chat_count = -1
            coordination_error = f"active chat coordination unavailable: {exc}"
    with kb.write_txn(conn):
        state = get_state(conn, task_id)
        if state is None:
            raise LifecycleConflict(f"task {task_id} has no workspace allocation")
        completed = _operation_receipts(state).get(kind)
        if isinstance(completed, dict):
            if completed.get("operation_key") != operation_key:
                raise LifecycleConflict(f"{kind} already completed under another key")
            return state, False
        if state.wait_kind is not None:
            if state.wait_kind != kind or state.operation_key != operation_key:
                raise LifecycleConflict("a different remote operation is already durable")
            return state, False
        if kind == "deploy" and int(active_chat_count or 0) != 0:
            conn.execute(
                "UPDATE coding_worker_lifecycle SET last_error=?, updated_at=? "
                "WHERE task_id=?",
                (
                    coordination_error or "deploy blocked by active chat sessions",
                    int(now), task_id,
                ),
            )
            blocked = get_state(conn, task_id)
            assert blocked is not None
            return blocked, False
        deadline = int(now) + max(1, int(timeout_seconds))
        conn.execute(
            "UPDATE coding_worker_lifecycle SET phase=?, wait_kind=?, "
            "operation_key=?, wait_started_at=?, wait_deadline=?, silent_checks=0, "
            "last_error=NULL, updated_at=? WHERE task_id=?",
            (
                f"{kind}_wait", kind, operation_key, int(now), deadline,
                int(now), task_id,
            ),
        )
        state = get_state(conn, task_id)
    assert state is not None
    return state, True


def observe_wait(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    operation_key: str,
    observation: str,
    now: int,
    receipt: Optional[dict] = None,
    error: Optional[str] = None,
) -> CodingWorkerState:
    """Record a provider observation without extending the durable deadline."""
    if observation not in {"pending", "silent", "connection_error", "complete"}:
        raise ValueError("invalid wait observation")
    with kb.write_txn(conn):
        state = get_state(conn, task_id)
        if state is None or state.operation_key != operation_key:
            raise LifecycleConflict("remote operation ownership changed")
        if state.wait_kind not in {"merge", "deploy"}:
            raise LifecycleConflict("task has no active merge/deploy wait")
        silent_checks = state.silent_checks + 1 if observation == "silent" else 0
        phase = state.phase
        wait_kind = state.wait_kind
        active_key: Optional[str] = state.operation_key
        encoded_receipt = state.operation_receipt
        last_error = error if observation == "connection_error" else None
        if observation == "complete":
            phase = "merged" if wait_kind == "merge" else "complete"
            receipts = _operation_receipts(state)
            receipts[str(wait_kind)] = {
                "operation_key": state.operation_key,
                "receipt": receipt or {},
            }
            encoded_receipt = json.dumps(receipts, sort_keys=True)
            wait_kind = None
            active_key = None
            silent_checks = 0
        elif (
            observation == "silent"
            and silent_checks >= 2
            and state.wait_deadline is not None
            and int(now) >= state.wait_deadline
        ):
            phase = "timed_out"
            last_error = f"{state.wait_kind} wait exceeded its durable deadline"
        conn.execute(
            "UPDATE coding_worker_lifecycle SET phase=?, wait_kind=?, operation_key=?, "
            "silent_checks=?, operation_receipt=?, last_error=?, updated_at=? "
            "WHERE task_id=?",
            (
                phase, wait_kind, active_key, silent_checks, encoded_receipt,
                last_error, int(now), task_id,
            ),
        )
        state = get_state(conn, task_id)
    assert state is not None
    return state
