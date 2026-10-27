"""Fail-closed native external-intake review provenance for owner recovery.

The board/scheduler is the authority, not a GitHub comment's shared account or
caller-supplied JSON. This v1 policy recognizes only the Orion review lane. The
local OS owner remains the trust boundary (not isolation against the same uid).
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import re


def github_timestamp(value: object) -> float:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", value):
        raise ValueError("missing or malformed GitHub evidence timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        raise ValueError("malformed GitHub evidence timestamp") from None


def native_review_snapshot(conn, task_id: str, review_task_id: str, run_id: int,
                           pr_url: str, head_sha: str, merge_sha: str) -> dict:
    """Read only supported board accessors, on the target's own connection.

    Require an ingested exact-head handoff followed by a real dispatched review
    claim/spawn/completion, never a named comment author or synthetic completion.
    The returned snapshot must be compared again under the reconciliation lock.
    """
    from hermes_cli import kanban_db as kb

    target = kb.get_task(conn, task_id)
    review_task = kb.get_task(conn, review_task_id)
    run = kb.get_run(conn, run_id)
    implementers = sorted({r.profile for r in kb.list_runs(conn, task_id) if r.profile})
    if (target is None or review_task is None or review_task_id == task_id
            or review_task.created_by != "github-webhook"
            or review_task.status not in {"done", "archived"}
            or run is None or run.task_id != review_task_id
            or run.profile != "orion" or run.profile in implementers
            or target.assignee == run.profile or not implementers
            or run.status != "done" or run.outcome != "completed" or run.error
            or not run.ended_at or run.ended_at < run.started_at):
        raise ValueError("native evidence requires an independent dispatched Orion review run")
    runs = kb.list_runs(conn, review_task_id)
    if not runs or max(r.id for r in runs) != run_id:
        raise ValueError("native review is superseded by a later run")
    metadata = run.metadata or {}
    if (metadata.get("review_outcome") != "approved_merged_source_only"
            or metadata.get("published_pr") != pr_url
            or metadata.get("reviewed_head") != head_sha
            or metadata.get("merge_commit") != merge_sha
            or not isinstance(metadata.get("reviewer_checks"), list)
            or not metadata["reviewer_checks"]
            or any(not isinstance(check, str) or not check.strip() for check in metadata["reviewer_checks"])
            or not isinstance(metadata.get("worker_session_id"), str)
            or not metadata["worker_session_id"].strip()):
        raise ValueError("native approval must bind exact PR/head/merge and reviewer session/checks")
    events = kb.list_events(conn, review_task_id)
    requests = [e for e in events if e.kind == "review_requested" and e.created_at <= run.started_at]
    if not requests:
        raise ValueError("missing native review request")
    request = max(requests, key=lambda e: e.id)
    intake = kb.get_run(conn, request.run_id) if request.run_id else None
    ingress = (intake.metadata or {}) if intake else {}
    repo, number = pr_url.removeprefix("https://github.com/").split("/pull/")
    if ((request.payload or {}).get("reviewer") != run.profile
            or intake is None or intake.task_id != review_task_id
            or intake.outcome != "review_requested" or intake.ended_at is None
            or ingress.get("adapter") != "github_pr_native_ingest"
            or ingress.get("repository") != repo or ingress.get("number") != int(number)
            or ingress.get("url") != pr_url or ingress.get("head_sha") != head_sha):
        raise ValueError("native intake must bind exact GitHub repository/PR/head")
    claimed = [e for e in events if e.kind == "claimed" and e.run_id == run_id
               and (e.payload or {}).get("source_status") == "review"]
    spawned = [e for e in events if e.kind == "spawned" and e.run_id == run_id
               and isinstance((e.payload or {}).get("pid"), int) and (e.payload or {}).get("pid", 0) > 0
               and (e.payload or {}).get("started_at")]
    completed = [e for e in events if e.kind == "completed" and e.run_id == run_id]
    if len(claimed) != 1 or len(spawned) != 1 or len(completed) != 1:
        raise ValueError("missing or ambiguous native claim/spawn/completion provenance")
    claim, spawn, complete = claimed[0], spawned[0], completed[0]
    if (not request.id < claim.id < spawn.id < complete.id
            or claim.created_at != run.started_at
            or not run.started_at <= spawn.created_at <= run.ended_at
            or complete.created_at != run.ended_at
            or (complete.payload or {}).get("summary") != run.summary
            or any(e.id > request.id and e.kind in {"changes_requested", "review_reopened"} for e in events)):
        raise ValueError("native review lifecycle is inconsistent or revoked")
    return {"kind": "native-orion-source-review-v1", "canonical_task_id": task_id,
            "review_task_id": review_task_id, "review_run_id": run_id,
            "implementers": implementers, "reviewer": run.profile,
            "started_at": run.started_at, "ended_at": run.ended_at,
            "summary": run.summary, "metadata": metadata,
            "intake_run_id": intake.id, "intake_metadata": ingress,
            "events": [asdict(e) for e in (request, claim, spawn, complete)]}


def corroborate_native_comment(pr: dict, comment: dict, review_url: str, comment_id: int,
                              native: dict, head_sha: str) -> dict:
    """A premerge comment corroborates time only; native run supplies authority."""
    repo, number = pr["html_url"].removeprefix("https://github.com/").split("/pull/")
    body = comment.get("body") or ""
    created = github_timestamp(comment.get("created_at"))
    updated = github_timestamp(comment.get("updated_at"))
    merged = github_timestamp(pr.get("merged_at"))
    canonical_link = f"https://hmc.sololink.cloud/#/tasks?task={native['canonical_task_id']}"
    # Match a whole canonical link in its task section, not a prefix sibling ID.
    section = re.search(r"(?m)^## Kanban Task\s*\n(.*?)(?=^## |\Z)", pr.get("body") or "", re.S | re.M)
    if (comment.get("id") != comment_id or comment.get("html_url") != review_url
            or comment.get("issue_url") != f"https://api.github.com/repos/{repo}/issues/{number}"
            or not section or not re.search(re.escape(canonical_link) + r"(?=\s|$)", section[1])
            or not re.search(r"(?<![0-9a-f])" + re.escape(head_sha) + r"(?![0-9a-f])", body)
            or not body.startswith(native["reviewer"].capitalize() + " review at immutable " + head_sha
                                   + ": approved for source merge")
            or not native["started_at"] <= created <= updated <= merged <= native["ended_at"]):
        raise ValueError("native review requires exact-task premerge approval corroboration during that run")
    return {key: comment[key] for key in ("id", "html_url", "issue_url", "created_at", "updated_at", "body")}
