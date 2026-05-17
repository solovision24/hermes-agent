"""Mirror Hermes Kanban tasks to SoLoRecall blocks.

Hermes Kanban is the source of truth by default. SoLoRecall-to-Hermes
imports/updates are performed only when two-way sync is explicitly enabled.

Runnable examples:

    python -m hermes_cli.solorecall_kanban_sync --dry-run
    python -m hermes_cli.solorecall_kanban_sync --apply --limit 25
    python -m hermes_cli.solorecall_kanban_sync --apply --daemon --interval 180

No hard deletes are performed. Audit reports and idempotent mapping state are
written under ``~/.hermes/reports/hermes-solorecall-sync/`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

from hermes_constants import get_default_hermes_root
from hermes_cli import kanban_db
from hermes_cli.config import load_config

BASE_URL = "https://api.solorecall.com"
REPORT_SUBDIR = "hermes-solorecall-sync"
DEFAULT_SYNC_MODE = "outbound_only"
SYNC_MODES = ("outbound_only", "two_way")
BLOCK_TYPE = "page"
MARKER_KEY = "hermes_kanban_sync"
MARKER_VERSION = 1

HERMES_TO_SOLORECALL = {
    "triage": "Triage",
    "todo": "Todo",
    "ready": "Ready",
    "running": "Running",
    "blocked": "Blocked",
    "done": "Done",
    "archived": "Archived",
}
SOLORECALL_TO_HERMES = {v: k for k, v in HERMES_TO_SOLORECALL.items()}
_STATUS_ALIASES = {
    "triage": "Triage",
    "todo": "Todo",
    "to do": "Todo",
    "ready": "Ready",
    "running": "Running",
    "active": "Running",
    "in progress": "Running",
    "blocked": "Blocked",
    "done": "Done",
    "complete": "Done",
    "completed": "Done",
    "archived": "Archived",
}


@dataclass
class SyncStats:
    solorecall_blocks_seen: int = 0
    solorecall_blocks_created: int = 0
    solorecall_blocks_would_create: int = 0
    solorecall_blocks_updated: int = 0
    solorecall_blocks_would_update: int = 0
    solorecall_blocks_archived_or_trashed_seen: int = 0
    hermes_tasks_seen: int = 0
    hermes_tasks_created: int = 0
    hermes_tasks_would_create: int = 0
    hermes_tasks_updated: int = 0
    hermes_tasks_would_update: int = 0
    inbound_skipped_outbound_only: int = 0
    no_hard_delete_skips: int = 0
    conflicts: int = 0
    errors: list[str] = field(default_factory=list)
    changed: bool = False


@dataclass
class SoLoRecallTask:
    block_id: str
    title: str
    status: str
    canonical_status: str
    body: str | None = None
    assignee: str | None = None
    priority: int | None = None
    hermes_task_id: str | None = None
    updated_at: str | None = None
    archived: bool = False
    in_trash: bool = False


class SoLoRecallError(RuntimeError):
    """Raised when SoLoRecall returns an API error."""


class SoLoRecallClient:
    def __init__(self, *, api_key: str | None = None, base_url: str = BASE_URL, timeout: int = 30) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("SOLORECALL_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise SoLoRecallError("SOLORECALL_API_KEY is required")

    def _request(self, method: str, path: str, *, json_body: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        try:
            resp = requests.request(method, url, headers=headers, json=json_body, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SoLoRecallError(f"SoLoRecall request failed: {exc}") from exc
        if resp.status_code >= 400:
            detail = _safe_error_detail(resp)
            raise SoLoRecallError(f"SoLoRecall {method} {path} returned HTTP {resp.status_code}: {detail}")
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError as exc:
            raise SoLoRecallError(f"SoLoRecall {method} {path} returned non-JSON response") from exc

    def list_blocks(self, *, limit: int = 100, offset: int = 0, block_type: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if block_type:
            params["type"] = block_type
        data = self._request("GET", "/api/v1/blocks", params=params)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("blocks", "data", "items", "results"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []

    def create_block(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._request("POST", "/api/v1/blocks", json_body=payload)
        return data if isinstance(data, dict) else {}

    def update_block(self, block_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = self._request("PATCH", f"/api/v1/blocks/{block_id}", json_body=payload)
        return data if isinstance(data, dict) else {}

    def archive_block(self, block_id: str) -> dict[str, Any]:
        data = self._request("POST", f"/api/v1/blocks/{block_id}/archive")
        return data if isinstance(data, dict) else {}


def _safe_error_detail(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return str(data.get("error") or data.get("message") or data)[:500]
        return str(data)[:500]
    except Exception:
        return (resp.text or "").strip()[:500]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_sync_mode(value: Any, *, default: str = DEFAULT_SYNC_MODE) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "outbound": "outbound_only",
        "outboundonly": "outbound_only",
        "hermes_to_solorecall": "outbound_only",
        "read_only_solorecall": "outbound_only",
        "two_way": "two_way",
        "twoway": "two_way",
        "bidirectional": "two_way",
        "solorecall_to_hermes": "two_way",
        "inbound_enabled": "two_way",
    }
    mode = aliases.get(raw, raw)
    return mode if mode in SYNC_MODES else default


def configured_sync_mode(*, explicit_mode: str | None = None, outbound_only: bool = False, enable_solorecall_import: bool = False) -> str:
    if outbound_only:
        return "outbound_only"
    if enable_solorecall_import:
        return "two_way"
    if explicit_mode:
        return normalize_sync_mode(explicit_mode)
    env_mode = os.environ.get("HERMES_SOLORECALL_KANBAN_SYNC_MODE")
    if env_mode:
        return normalize_sync_mode(env_mode)
    env_allow = os.environ.get("HERMES_SOLORECALL_KANBAN_ALLOW_INBOUND")
    if env_allow and env_allow.strip().lower() in {"1", "true", "yes", "on", "two_way"}:
        return "two_way"
    try:
        cfg = load_config() or {}
    except Exception:
        cfg = {}
    sr_cfg = cfg.get("solorecall_kanban_sync") if isinstance(cfg, dict) else {}
    if isinstance(sr_cfg, dict):
        return normalize_sync_mode(sr_cfg.get("mode"))
    return DEFAULT_SYNC_MODE


def _canonical_solorecall_status(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in SOLORECALL_TO_HERMES:
        return raw
    return _STATUS_ALIASES.get(raw.lower(), "Todo")


def _properties_for_task(task: kanban_db.Task) -> dict[str, Any]:
    return {
        MARKER_KEY: {"version": MARKER_VERSION, "task_id": task.id, "synced_at": _now_iso()},
        "title": task.title,
        "status": HERMES_TO_SOLORECALL.get(task.status, "Todo"),
        "hermes_task_id": task.id,
        "assignee": task.assignee,
        "priority": int(task.priority or 0),
        "created_by": task.created_by,
        "source": "Hermes Kanban",
    }


def _content_for_task(task: kanban_db.Task) -> dict[str, Any]:
    return {
        "body": task.body or "",
        "result": task.result or "",
        "workspace_kind": task.workspace_kind,
        "workspace_path": task.workspace_path,
        "status": task.status,
    }


def _payload_for_task(task: kanban_db.Task) -> dict[str, Any]:
    return {"type": BLOCK_TYPE, "properties": _properties_for_task(task), "content": _content_for_task(task)}


def _extract_title(block: dict[str, Any]) -> str:
    props = block.get("properties") if isinstance(block.get("properties"), dict) else {}
    for key in ("title", "name", "Title", "Name"):
        val = props.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            txt = val.get("plain_text") or val.get("content") or val.get("text")
            if txt:
                return str(txt).strip()
    content = block.get("content")
    if isinstance(content, dict) and content.get("title"):
        return str(content.get("title")).strip()
    return f"SoLoRecall block {block.get('id')}"


def _extract_body(block: dict[str, Any]) -> str | None:
    content = block.get("content")
    if isinstance(content, dict):
        for key in ("body", "text", "description"):
            if content.get(key):
                return str(content.get(key))
    if isinstance(content, str):
        return content
    props = block.get("properties") if isinstance(block.get("properties"), dict) else {}
    notes = props.get("body") or props.get("notes") or props.get("description")
    return str(notes) if notes else None


def parse_solorecall_task(block: dict[str, Any]) -> SoLoRecallTask | None:
    block_id = str(block.get("id") or "").strip()
    if not block_id:
        return None
    props = block.get("properties") if isinstance(block.get("properties"), dict) else {}
    marker = props.get(MARKER_KEY) if isinstance(props.get(MARKER_KEY), dict) else {}
    hermes_task_id = props.get("hermes_task_id") or marker.get("task_id")
    status = props.get("status") or props.get("Status") or "Todo"
    canonical = _canonical_solorecall_status(status)
    priority = props.get("priority")
    try:
        priority_int = int(priority) if priority is not None else None
    except (TypeError, ValueError):
        priority_int = None
    return SoLoRecallTask(
        block_id=block_id,
        title=_extract_title(block),
        status=str(status),
        canonical_status=canonical,
        body=_extract_body(block),
        assignee=str(props.get("assignee")).strip() if props.get("assignee") else None,
        priority=priority_int,
        hermes_task_id=str(hermes_task_id).strip() if hermes_task_id else None,
        updated_at=block.get("updated_at"),
        archived=bool(block.get("archived")),
        in_trash=bool(block.get("in_trash")),
    )


class MappingStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"task_to_block": {}, "block_to_task": {}}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update({k: v for k, v in loaded.items() if isinstance(v, dict)})
            except Exception:
                pass

    def block_for_task(self, task_id: str) -> str | None:
        val = self.data.get("task_to_block", {}).get(task_id)
        return str(val) if val else None

    def task_for_block(self, block_id: str) -> str | None:
        val = self.data.get("block_to_task", {}).get(block_id)
        return str(val) if val else None

    def remember(self, task_id: str, block_id: str) -> None:
        self.data.setdefault("task_to_block", {})[task_id] = block_id
        self.data.setdefault("block_to_task", {})[block_id] = task_id
        self.data["updated_at"] = _now_iso()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SoLoRecallKanbanSync:
    def __init__(
        self,
        *,
        solorecall: SoLoRecallClient,
        report_dir: Path | None = None,
        state_path: Path | None = None,
        board: str | None = None,
        db_path: Path | None = None,
    ) -> None:
        root = get_default_hermes_root()
        self.solorecall = solorecall
        self.report_dir = report_dir or (root / "reports" / REPORT_SUBDIR)
        self.state_path = state_path or (self.report_dir / "mapping.json")
        self.board = board
        self.db_path = db_path

    def run_once(self, *, apply: bool = False, limit: int = 100, mode: str = DEFAULT_SYNC_MODE) -> dict[str, Any]:
        stats = SyncStats()
        mode = normalize_sync_mode(mode)
        mapping = MappingStore(self.state_path)
        conn = kanban_db.connect(self.db_path, board=self.board)
        try:
            tasks = kanban_db.list_tasks(conn, limit=limit)
            blocks = self.solorecall.list_blocks(limit=limit, block_type=BLOCK_TYPE)
            sr_tasks: dict[str, SoLoRecallTask] = {}
            for block in blocks:
                parsed = parse_solorecall_task(block)
                if not parsed:
                    continue
                stats.solorecall_blocks_seen += 1
                sr_tasks[parsed.block_id] = parsed
                if parsed.hermes_task_id:
                    mapping.remember(parsed.hermes_task_id, parsed.block_id)
            by_task = {t.id: t for t in tasks}
            block_ids_seen = set(sr_tasks)

            for task in tasks:
                stats.hermes_tasks_seen += 1
                block_id = mapping.block_for_task(task.id)
                sr_task = sr_tasks.get(block_id) if block_id else None
                if block_id and block_id not in block_ids_seen:
                    stats.no_hard_delete_skips += 1
                    block_id = None
                    sr_task = None
                if sr_task and (sr_task.archived or sr_task.in_trash):
                    stats.solorecall_blocks_archived_or_trashed_seen += 1
                    stats.no_hard_delete_skips += 1
                    continue
                if not sr_task:
                    payload = _payload_for_task(task)
                    if apply:
                        created = self.solorecall.create_block(payload)
                        created_id = str(created.get("id") or "")
                        if created_id:
                            mapping.remember(task.id, created_id)
                        stats.solorecall_blocks_created += 1
                        stats.changed = True
                    else:
                        stats.solorecall_blocks_would_create += 1
                    continue
                desired = _payload_for_task(task)
                if self._needs_remote_update(task, sr_task):
                    # In explicit two-way mode, a mapped SoLoRecall block that differs
                    # from Hermes is treated as an inbound edit for this pass. Skipping
                    # the outbound patch here prevents the same run from overwriting the
                    # remote value and then importing the stale pre-patch snapshot.
                    if mode == "two_way":
                        stats.conflicts += 1
                    elif apply:
                        self.solorecall.update_block(sr_task.block_id, desired)
                        mapping.remember(task.id, sr_task.block_id)
                        stats.solorecall_blocks_updated += 1
                        stats.changed = True
                    else:
                        stats.solorecall_blocks_would_update += 1

            if mode == "two_way":
                for sr_task in sr_tasks.values():
                    if sr_task.archived or sr_task.in_trash:
                        stats.solorecall_blocks_archived_or_trashed_seen += 1
                        stats.no_hard_delete_skips += 1
                        continue
                    task_id = sr_task.hermes_task_id or mapping.task_for_block(sr_task.block_id)
                    hermes_status = SOLORECALL_TO_HERMES.get(sr_task.canonical_status, "todo")
                    if task_id and task_id in by_task:
                        task = by_task[task_id]
                        if task.status != hermes_status or task.title != sr_task.title or (sr_task.priority is not None and task.priority != sr_task.priority):
                            if apply:
                                self._update_hermes_task(conn, task.id, title=sr_task.title, status=hermes_status, priority=sr_task.priority)
                                kanban_db.add_comment(conn, task.id, "solorecall-sync", f"Synced inbound update from SoLoRecall block {sr_task.block_id}")
                                stats.hermes_tasks_updated += 1
                                stats.changed = True
                            else:
                                stats.hermes_tasks_would_update += 1
                        mapping.remember(task.id, sr_task.block_id)
                    elif not task_id:
                        if apply:
                            new_id = kanban_db.create_task(
                                conn,
                                title=sr_task.title,
                                body=sr_task.body,
                                assignee=sr_task.assignee,
                                created_by="solorecall-sync",
                                priority=sr_task.priority or 0,
                                triage=(hermes_status == "triage"),
                                idempotency_key=f"solorecall:{sr_task.block_id}",
                            )
                            if hermes_status != "ready":
                                self._update_hermes_task(conn, new_id, status=hermes_status)
                            mapping.remember(new_id, sr_task.block_id)
                            self.solorecall.update_block(sr_task.block_id, {"properties": {**{"hermes_task_id": new_id}, **_properties_for_task(kanban_db.get_task(conn, new_id))}})
                            stats.hermes_tasks_created += 1
                            stats.changed = True
                        else:
                            stats.hermes_tasks_would_create += 1
            else:
                inbound_candidates = [s for s in sr_tasks.values() if not s.hermes_task_id and not mapping.task_for_block(s.block_id)]
                stats.inbound_skipped_outbound_only += len(inbound_candidates)

            if apply:
                mapping.save()
            report = self._write_report(stats, mode=mode, apply=apply)
            return {"mode": mode, "apply": apply, "report": str(report), "state": str(self.state_path), "stats": asdict(stats)}
        except Exception as exc:
            stats.errors.append(str(exc))
            self._write_report(stats, mode=mode, apply=apply)
            raise
        finally:
            conn.close()

    def _needs_remote_update(self, task: kanban_db.Task, sr_task: SoLoRecallTask) -> bool:
        desired_status = HERMES_TO_SOLORECALL.get(task.status, "Todo")
        return any([
            sr_task.title != task.title,
            sr_task.canonical_status != desired_status,
            sr_task.assignee != task.assignee,
            sr_task.priority != int(task.priority or 0),
        ])

    def _update_hermes_task(self, conn: sqlite3.Connection, task_id: str, *, title: str | None = None, status: str | None = None, priority: int | None = None) -> None:
        fields: list[str] = []
        params: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if status is not None:
            if status not in kanban_db.VALID_STATUSES:
                raise ValueError(f"invalid inbound status {status!r}")
            fields.append("status = ?")
            params.append(status)
        if priority is not None:
            fields.append("priority = ?")
            params.append(int(priority))
        if not fields:
            return
        params.append(task_id)
        with kanban_db.write_txn(conn):
            conn.execute(f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?", params)

    def _write_report(self, stats: SyncStats, *, mode: str, apply: bool) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.report_dir / f"solorecall-kanban-sync-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        payload = {"created_at": _now_iso(), "mode": mode, "apply": apply, "stats": asdict(stats)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize Hermes Kanban with SoLoRecall blocks")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", default=True, help="preview changes without mutating SoLoRecall or Hermes (default)")
    action.add_argument("--apply", action="store_true", help="apply proposed changes")
    parser.add_argument("--limit", type=int, default=100, help="maximum Hermes tasks / SoLoRecall blocks to inspect per run")
    parser.add_argument("--mode", choices=SYNC_MODES, help="source-of-truth mode; default is outbound_only")
    parser.add_argument("--outbound-only", action="store_true", help="force Hermes -> SoLoRecall only")
    parser.add_argument("--enable-solorecall-import", action="store_true", help="explicitly enable SoLoRecall -> Hermes imports/updates")
    parser.add_argument("--board", help="Hermes Kanban board slug")
    parser.add_argument("--db-path", type=Path, help="explicit Hermes Kanban SQLite DB path")
    parser.add_argument("--report-dir", type=Path, help="directory for audit reports")
    parser.add_argument("--state-path", type=Path, help="mapping state JSON path")
    parser.add_argument("--base-url", default=BASE_URL, help="SoLoRecall API base URL")
    parser.add_argument("--daemon", action="store_true", help="run forever")
    parser.add_argument("--interval", type=int, default=180, help="daemon interval seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mode = configured_sync_mode(
        explicit_mode=args.mode,
        outbound_only=args.outbound_only,
        enable_solorecall_import=args.enable_solorecall_import,
    )
    apply_changes = bool(args.apply)
    client = SoLoRecallClient(base_url=args.base_url)
    engine = SoLoRecallKanbanSync(solorecall=client, report_dir=args.report_dir, state_path=args.state_path, board=args.board, db_path=args.db_path)
    while True:
        result = engine.run_once(apply=apply_changes, limit=args.limit, mode=mode)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not args.daemon:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
