"""Audited, identity-bound continuation of an existing PR; never a new build lane."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_pr_acceptance import _PR, _api
from hermes_cli.kanban_reconcile import authorize_local_owner, _required


def contract_version(conn, task_id: str) -> str:
    """Content + specification-event version: edit-away-and-back invalidates proof."""
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        raise ValueError("task not found")
    fields = ("id", "title", "body", "assignee", "completion_contract", "branch_name",
              "workspace_path", "workspace_kind", "skills", "goal_mode", "tenant")
    data = {key: row[key] for key in fields}
    data["parents"] = [r[0] for r in conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id=? ORDER BY parent_id", (task_id,))]
    data["revision_event"] = conn.execute(
        "SELECT coalesce(max(id),0) FROM task_events WHERE task_id=? "
        "AND kind IN ('edited','assigned','specified','linked','unlinked','review_scope_reconciled','pr_contract_reconciled')",
        (task_id,)).fetchone()[0]
    data["latest_run_id"] = _last_run(conn, task_id)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def require_idle(conn, task_id: str):
    row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None or row["status"] not in {"ready", "blocked", "review"}:
        raise ValueError("recovery requires an existing ready/blocked/review card")
    if any(row[k] is not None for k in ("claim_lock", "claim_expires", "worker_pid", "worker_started_at", "current_run_id")):
        raise ValueError("recovery refuses outstanding claim/worker/run state")
    if conn.execute("SELECT 1 FROM task_runs WHERE task_id=? AND (ended_at IS NULL OR status='running')", (task_id,)).fetchone():
        raise ValueError("recovery refuses an open run")
    return row


def _last_run(conn, task_id):
    row = conn.execute("SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return row[0] if row else None


def authorize_pr_remediation(conn, task_id: str, *, operator: str, confirm: str,
                             pr_url: str, head_sha: str, branch: str,
                             head_repository: str, assignee: str, reason: str) -> dict:
    actor = authorize_local_owner(conn, operator, task_id, confirm)
    reason = _required(reason, "reason")
    if not _PR.fullmatch(pr_url) or not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise ValueError("exact PR URL and lowercase full head SHA required")
    before = dict(require_idle(conn, task_id))
    if before["status"] == "review" or before["assignee"] != assignee or not assignee:
        raise ValueError("remediation must retain the current implementation owner")
    if before["branch_name"] and before["branch_name"] != branch:
        raise ValueError("task branch conflicts with remediation")
    contract = before["completion_contract"]
    match = _PR.fullmatch(pr_url)
    assert match is not None
    repo, number = match.groups()
    if contract not in {"local-only", repo, pr_url}:
        raise ValueError("PR conflicts with persisted contract")
    run_id = _last_run(conn, task_id)
    run = kb.get_run(conn, run_id) if run_id else None
    metadata = (run.metadata or {}) if run else {}
    if (not run or run.profile != assignee or metadata.get("published_pr") != pr_url
            or metadata.get("head_sha") != head_sha or metadata.get("branch") != branch):
        raise ValueError("latest implementation handoff must bind the same owner/PR/branch/head")
    version = contract_version(conn, task_id)
    pr = _api(f"repos/{repo}/pulls/{number}")
    if (pr.get("html_url") != pr_url or pr.get("state") != "open" or pr.get("merged")
            or pr.get("head", {}).get("sha") != head_sha
            or pr.get("head", {}).get("ref") != branch
            or pr.get("head", {}).get("repo", {}).get("full_name") != head_repository):
        raise ValueError("live PR owner/branch/head does not match remediation")
    payload = dict(task_id=task_id, actor=actor, reason=reason, pr_url=pr_url,
                   head_sha=head_sha, branch=branch, head_repository=head_repository,
                   assignee=assignee, contract_version=version, previous_run_id=run_id)
    with kb.write_txn(conn):
        if dict(require_idle(conn, task_id)) != before or contract_version(conn, task_id) != version or _last_run(conn, task_id) != run_id:
            raise ValueError("task changed during remediation authorization")
        kb._append_event(conn, task_id, "pr_remediation_authorized", payload)
        kb._insert_comment(conn, task_id, actor,
            f"Authorized SAME-PR remediation only: {pr_url}, branch {branch}, "
            f"head repository {head_repository}, pinned head {head_sha}, owner {assignee}. "
            "Continue this artifact; do not open a duplicate PR or change the full task specification. "
            + reason, int(time.time()))
    return payload


def remediation_allows(conn, task_id: str, pr_urls: set[str]) -> bool:
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_remediation_authorized' ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if not row:
        return False
    try:
        data = json.loads(row[0])
        task = require_idle(conn, task_id)
        matches = (task["status"] == "ready" and data["task_id"] == task_id
                and data["assignee"] == task["assignee"]
                and data["contract_version"] == contract_version(conn, task_id)
                and data["previous_run_id"] == _last_run(conn, task_id)
                and pr_urls == {data["pr_url"]})
        if not matches:
            return False
        match = _PR.fullmatch(data["pr_url"])
        if match is None:
            return False
        pr = _api(f"repos/{match[1]}/pulls/{match[2]}")
        return (pr.get("state") == "open" and not pr.get("merged")
                and pr.get("head", {}).get("sha") == data["head_sha"]
                and pr.get("head", {}).get("ref") == data["branch"]
                and pr.get("head", {}).get("repo", {}).get("full_name") == data["head_repository"])
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError):
        return False


def cmd_authorize_pr_remediation(args) -> int:
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        payload = authorize_pr_remediation(conn, args.task_id, operator=args.operator,
            confirm=args.confirm, pr_url=args.pr_url, head_sha=args.head_sha,
            branch=args.branch, head_repository=args.head_repository,
            assignee=args.assignee, reason=args.reason)
    print(json.dumps(payload))
    return 0


def cmd_recovery_version(args) -> int:
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        version = contract_version(conn, args.task_id)
    print(json.dumps(dict(task_id=args.task_id, contract_version=version)))
    return 0
