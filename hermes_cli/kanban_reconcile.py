"""Explicit local-owner recovery of an independently reviewed Kanban PR contract.

Not a CI waiver: this policy only covers a private GitHub repository whose exact
merged PR has no hosted check runs/statuses and whose required-check policy is
unavailable. It does not declare any check successful. Only an interactive local
OS owner (not a dispatched worker) can invoke this CLI-only operation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from urllib.parse import quote

from hermes_cli.kanban_pr_acceptance import _PR, _api
from hermes_cli.kanban_db_connect import write_txn

_SHA = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_POLICY = "private-merged-exact-pr-no-hosted-evidence-v2"
_PLAN_MESSAGE = "Upgrade to GitHub Pro or make this repository public to enable this feature."


def _plan_limitation(endpoint: str) -> dict:
    """Accept only GitHub's explicit plan response, never a generic permission 403.

    The ordinary acceptance reader deliberately redacts error messages. Keep this
    narrow policy probe separate; persist only the allowlisted message, not stderr.
    """
    from hermes_cli.kanban_pr_acceptance import _gh_env
    result = subprocess.run(
        ["gh", "api", endpoint, "--hostname", "github.com"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, env=_gh_env(None),
    )
    try:
        response = json.loads(result.stdout)
    except (ValueError, TypeError):
        raise ValueError("rules plan limitation is not verifiable") from None
    if (result.returncode == 0 or not isinstance(response, dict)
            or response.get("status") != "403" or response.get("message") != _PLAN_MESSAGE):
        raise ValueError("permission failure or readable rules are not a plan limitation")
    return {"endpoint": endpoint, "status": 403, "message": _PLAN_MESSAGE}


def _affirmative_review_history(repo: str, number: int, verdict: dict) -> None:
    latest = {}
    found = False
    # Explicit pages also work on owner machines with gh versions predating --slurp.
    for page_number in range(1, 101):
        page = _api(f"repos/{repo}/pulls/{number}/reviews?per_page=100&page={page_number}")
        if not isinstance(page, list) or any(not isinstance(review, dict) for review in page):
            raise ValueError("incomplete review history")
        for review in page:
            if review.get("id") == verdict["id"]:
                found = review == verdict
            if review.get("state") not in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
                continue
            uid = (review.get("user") or {}).get("id")
            if not isinstance(uid, int) or not isinstance(review.get("id"), int):
                raise ValueError("review history has missing identity")
            if uid not in latest or latest[uid]["id"] < review["id"]:
                latest[uid] = review
        if len(page) < 100:
            break
    else:
        raise ValueError("review history exceeds bounded recovery policy; use normal acceptance")
    if (not found or latest.get(verdict["user"]["id"], {}).get("id") != verdict["id"]
            or any(review["state"] == "CHANGES_REQUESTED" for review in latest.values())):
        raise ValueError("review approval is superseded or has unresolved changes requested")


def authorize_local_owner(conn: sqlite3.Connection, operator: str, task_id: str, confirm: str) -> str:
    """Use kernel uid + DB ownership, never a caller-supplied profile or dashboard author.

    This is a local machine owner boundary, not protection against another process
    running as the same OS account. No HTTP endpoint or worker tool exposes it.
    """
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise PermissionError("reconciliation requires a local POSIX owner shell")
    import pwd
    if os.environ.get("HERMES_KANBAN_TASK") or os.environ.get("HERMES_KANBAN_RUN_ID"):
        raise PermissionError("dispatched workers cannot reconcile contracts")
    if not sys.stdin.isatty():
        raise PermissionError("reconciliation requires an interactive owner terminal")
    db_path = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), None)
    if not db_path or os.stat(db_path).st_uid != os.geteuid():
        raise PermissionError("board database is not owned by this OS user")
    identity = pwd.getpwuid(os.geteuid()).pw_name
    if operator != identity or confirm != task_id:
        raise PermissionError("--operator must match the OS account and --confirm must match this task id")
    return identity


def _required(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise ValueError(f"{label} is required (maximum 2000 characters)")
    return value.strip()


def _snapshot(conn, task_id):
    return conn.execute(
        "SELECT status, current_run_id, completion_contract, body, goal_mode, claim_lock, "
        "worker_pid FROM tasks WHERE id=?", (task_id,),
    ).fetchone()


def _reconciliation_event(conn, task_id):
    row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='pr_contract_reconciled' "
                       "ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    return json.loads(row[0]) if row else None


def collect_merged_pr_evidence(pr_url: str, head_sha: str, merge_sha: str, review_url: str) -> dict:
    """Read real GitHub evidence, including independent exact-head approval.

    API reads use the interactive owner's gh login. The accepting worker's login
    remains unchanged for ordinary completion. No network is held under a DB lock.
    """
    match = _PR.fullmatch(pr_url)
    if not match or not _SHA.fullmatch(head_sha) or not _SHA.fullmatch(merge_sha):
        raise ValueError("exact GitHub PR URL and full lower-case head/merge SHAs are required")
    repo, number = match[1], int(match[2])
    review = re.fullmatch(rf"https://github\.com/{re.escape(repo)}/pull/{number}#pullrequestreview-([1-9][0-9]*)", review_url)
    if not review:
        raise ValueError("review evidence must be an exact affirmative PR review URL, not an issue comment")
    pr = _api(f"repos/{repo}/pulls/{number}")
    if (pr.get("html_url") != pr_url or pr.get("head", {}).get("sha") != head_sha
            or pr.get("merge_commit_sha") != merge_sha or not pr.get("merged")
            or not pr.get("merged_at") or pr.get("state") != "closed"):
        raise ValueError("PR URL/head/merge proof does not match a merged GitHub PR")
    if pr.get("base", {}).get("repo", {}).get("full_name") != repo:
        raise ValueError("PR base repository does not match its contract")
    review_endpoint = f"repos/{repo}/pulls/{number}/reviews/{review[1]}"
    verdict = _api(review_endpoint)
    reviewer = verdict.get("user") or {}
    author = pr.get("user") or {}
    if (verdict.get("html_url") != review_url or verdict.get("id") != int(review[1])
            or verdict.get("pull_request_url") != f"https://api.github.com/repos/{repo}/pulls/{number}"
            or verdict.get("commit_id") != head_sha or verdict.get("state") != "APPROVED"
            or not verdict.get("submitted_at") or reviewer.get("type") != "User"
            or not isinstance(reviewer.get("id"), int) or not isinstance(author.get("id"), int)
            or not reviewer.get("login") or not author.get("login")
            or reviewer["id"] == author["id"]
            or reviewer["login"].casefold() == author["login"].casefold()):
        raise ValueError("review must be an independent affirmative exact-head GitHub verdict")
    repo_data = _api(f"repos/{repo}")
    if repo_data.get("full_name") != repo or repo_data.get("private") is not True:
        raise ValueError("the no-hosted-evidence policy applies only to a private repository")
    if repo_data.get("permissions", {}).get("admin") is not True:
        raise ValueError("GitHub login must have verified repository admin permission")
    checks = _api(f"repos/{repo}/commits/{head_sha}/check-runs?per_page=100")
    statuses = _api(f"repos/{repo}/commits/{head_sha}/statuses?per_page=100")
    if checks.get("total_count") != 0 or statuses:
        raise ValueError("hosted evidence exists: use the normal PR acceptance gate")
    branch = quote(pr["base"]["ref"], safe="")
    # Independently require explicit plan denial for rules AND classic protection.
    rules = [_plan_limitation(f"repos/{repo}/rules/branches/{branch}?per_page=100"),
             _plan_limitation(f"repos/{repo}/branches/{branch}/protection")]
    # Re-read mutable PR state after all evidence calls; a moved head never inherits proof.
    current = _api(f"repos/{repo}/pulls/{number}")
    if (current.get("head", {}).get("sha") != head_sha or current.get("merge_commit_sha") != merge_sha
            or not current.get("merged") or current.get("base", {}).get("ref") != pr["base"]["ref"]):
        raise ValueError("PR evidence changed during collection")
    if _api(review_endpoint) != verdict:
        raise ValueError("review evidence changed during collection")
    _affirmative_review_history(repo, number, verdict)
    review_snapshot = {key: verdict[key] for key in
                       ("id", "html_url", "pull_request_url", "state", "commit_id", "submitted_at")}
    review_snapshot["user"] = {key: reviewer[key] for key in ("id", "login", "type")}
    review_snapshot["body_sha256"] = hashlib.sha256((verdict.get("body") or "").encode()).hexdigest()
    return {"pr_url": pr_url, "head_sha": head_sha, "merge_sha": merge_sha,
            "merged_at": pr["merged_at"], "review_url": review_url,
            "review": review_snapshot,
            "review_sha256": hashlib.sha256(json.dumps(review_snapshot, sort_keys=True).encode()).hexdigest(),
            "checks": "zero", "statuses": "zero", "rules": rules,
            "github_admin_verified": True, "policy": _POLICY}


def reconcile_pr_contract(conn: sqlite3.Connection, task_id: str, *, operator: str,
                          confirm: str, pr_url: str, head_sha: str, merge_sha: str,
                          review_url: str, reason: str) -> dict:
    """Bind an exact pre-existing PR to a local-only completion policy, audit first.

    This does not complete the card. The operator must subsequently use normal
    completion; that path still enforces parent dependencies and terminal evidence.
    """
    actor = authorize_local_owner(conn, operator, task_id, confirm)
    reason = _required(reason, "reason")
    before = _snapshot(conn, task_id)
    if before is None:
        raise ValueError("task not found")
    old = _reconciliation_event(conn, task_id)
    if before["completion_contract"] == "local-only" and old is not None:
        if old.get("policy") == _POLICY and all(old.get(key) == val for key, val in (("pr_url", pr_url), ("head_sha", head_sha),
               ("merge_sha", merge_sha), ("review_url", review_url), ("actor", actor))):
            return {"changed": False, "event": old, "status": before["status"]}
        raise ValueError("local-only contract was reconciled for different evidence")
    if before["completion_contract"] != pr_url or before["status"] not in {"ready", "blocked", "review"}:
        raise ValueError("task must be unclaimed and carry this exact persisted PR contract")
    if before["current_run_id"] is not None or before["claim_lock"] or before["worker_pid"]:
        raise ValueError("task still has an active claim")
    evidence = collect_merged_pr_evidence(pr_url, head_sha, merge_sha, review_url)
    from hermes_cli.kanban_db import _append_event
    payload = {**evidence, "actor": actor, "reason": reason,
               "original_contract": pr_url, "replacement_contract": "local-only",
               "hosted_ci_accepted": False}
    with write_txn(conn):
        current = _snapshot(conn, task_id)
        if current is None or tuple(current) != tuple(before):
            raise ValueError("task changed during evidence collection; retry")
        if _reconciliation_event(conn, task_id):
            raise ValueError("contract already reconciled; refresh task")
        conn.execute("UPDATE tasks SET completion_contract='local-only' WHERE id=?", (task_id,))
        _append_event(conn, task_id, "pr_contract_reconciled", payload)
    readback = _snapshot(conn, task_id)
    if readback["completion_contract"] != "local-only":
        raise RuntimeError("reconciled contract readback failed")
    return {"changed": True, "event": payload, "status": readback["status"]}


def reconcile_review_scope(conn: sqlite3.Connection, task_id: str, *, operator: str,
                           confirm: str, expected_body_sha256: str, amendment: dict,
                           deferred_child: str, reason: str) -> dict:
    """Record a narrow owner decision while retaining the complete original goal body.

    The normal goal judge remains active. A separate `request-review` must judge
    the *new* body and evidence; this operation never changes goal_mode/status.
    """
    actor = authorize_local_owner(conn, operator, task_id, confirm)
    reason = _required(reason, "reason")
    before = _snapshot(conn, task_id)
    if (before is None or before["status"] != "ready" or before["current_run_id"] is not None
            or before["claim_lock"] or before["worker_pid"] or not before["goal_mode"]):
        raise ValueError("goal-scope amendment requires an unclaimed ready goal-mode task")
    if not _SHA256.fullmatch(expected_body_sha256) or hashlib.sha256((before["body"] or "").encode()).hexdigest() != expected_body_sha256:
        raise ValueError("original body hash mismatch")
    if conn.execute("SELECT 1 FROM task_events WHERE task_id=? AND kind='review_scope_reconciled'",
                    (task_id,)).fetchone():
        raise ValueError("goal scope already amended; cumulative goal removal is forbidden")
    if not isinstance(amendment, dict) or set(amendment) != {"deferred_acceptance"}:
        raise ValueError("amendment must contain only deferred_acceptance exact original lines")
    deferred = amendment["deferred_acceptance"]
    original = before["body"] or ""
    lines = original.splitlines(keepends=True)
    if (not isinstance(deferred, list) or not deferred
            or any(not isinstance(item, str) or len(item.strip()) < 20 or "\n" in item or "\r" in item
                   for item in deferred) or len(set(deferred)) != len(deferred)
            or any(sum(line.rstrip("\r\n") == item for line in lines) != 1 for item in deferred)):
        raise ValueError("deferred acceptance must be unique complete original lines")
    retained = [line for line in lines if line.rstrip("\r\n") not in deferred]
    if (sum(len(item.strip()) for item in deferred) >= sum(len(line.strip()) for line in retained)
            or not any(len(line.strip()) >= 20 for line in retained)):
        raise ValueError("blanket or majority goal removal is forbidden")
    linked = conn.execute("SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (task_id, deferred_child)).fetchone()
    child = conn.execute("SELECT body, assignee, status FROM tasks WHERE id=?", (deferred_child,)).fetchone()
    if (not linked or child is None or not child["assignee"] or child["status"] in {"done", "archived"}
            or any(item not in (child["body"] or "").splitlines() for item in deferred)):
        raise ValueError("assigned directly dependent child must own every exact deferred acceptance line")
    revised_body = "".join(
        f"[Deferred acceptance to {deferred_child}: {line.rstrip(chr(13) + chr(10))}]\n"
        if line.rstrip("\r\n") in deferred else line for line in lines)
    revised_body += (f"\n\nOwner-bounded review scope; original body SHA256: {expected_body_sha256}.\n"
                     "All non-deferred requirements above remain operative and unchanged. "
                     "Only the explicitly marked acceptance belongs to the assigned child; "
                     "this review is not full-program delivery or release acceptance. "
                     "Normal goal judge, PR acceptance and independent review remain required.\n")
    from hermes_cli.kanban_db import _append_event
    with write_txn(conn):
        if tuple(_snapshot(conn, task_id) or ()) != tuple(before):
            raise ValueError("goal task changed during amendment")
        current_child = conn.execute("SELECT body, assignee, status FROM tasks WHERE id=?", (deferred_child,)).fetchone()
        if (tuple(current_child or ()) != tuple(child) or not conn.execute(
                "SELECT 1 FROM task_links WHERE parent_id=? AND child_id=?", (task_id, deferred_child)).fetchone()):
            raise ValueError("deferred child changed during amendment")
        conn.execute("UPDATE tasks SET body=? WHERE id=?", (revised_body, task_id))
        _append_event(conn, task_id, "review_scope_reconciled", {
            "actor": actor, "reason": reason, "original_body": before["body"],
            "original_body_sha256": expected_body_sha256, "revised_body": revised_body,
            "deferred_child": deferred_child, "goal_mode_preserved": True,
            "deferred_acceptance": deferred, "deferred_assignee": child["assignee"],
            "child_body_sha256": hashlib.sha256(child["body"].encode()).hexdigest(),
        })
    return {"changed": True, "body_sha256": hashlib.sha256(revised_body.encode()).hexdigest(),
            "status": _snapshot(conn, task_id)["status"]}
