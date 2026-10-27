"""Cron's owner-only quiet policy, separated from its transport dispatch."""
from __future__ import annotations

import logging
from datetime import timedelta

from cron.quiet_window import human_target, policy, release_at
import hermes_time

logger = logging.getLogger("cron.scheduler")


def defer_human_targets(job: dict, content: str, targets: list[dict],
                        user_cfg: dict | None, media_files: list,
                        for_failure: bool, errors: list[str]) -> set[tuple]:
    """Persist a whole human batch before any destination sends; fail closed on I/O."""
    try:
        configured = policy(user_cfg)
    except ValueError as exc:
        errors.append(f"invalid cron.quiet_window: {exc}; notification sent without deferral")
        logger.error("Job '%s': %s", job["id"], errors[-1])
        return set()
    if configured is None:
        return set()
    due = release_at(hermes_time.now(), configured)
    if due is None:
        return set()
    from gateway.warning_notifications import warning_notifications_enabled
    quiet = [dict(target, kind="human") for target in targets
             if human_target(target) and (not for_failure or
             warning_notifications_enabled(target["platform"], user_cfg))]
    if not quiet:
        return set()
    keys = {(t["platform"], t["chat_id"], t.get("thread_id")) for t in quiet}
    if media_files:
        errors.append("quiet-window delivery cannot retain media attachments")
        return keys
    if for_failure:
        for target in quiet:
            target["_incident_id"] = str(job.get("_failure_incident_id") or "")
    from cron.target_receipts import enqueue_targets
    from cron.scheduler_delivery import _redact_cron_payload
    try:
        execution_id = str(job.get("execution_id") or "")
        if not execution_id:
            raise ValueError("missing execution id for durable quiet-window delivery")
        if for_failure:
            execution_id += ":failure"
        rows = enqueue_targets(execution_id, job["id"],
                               _redact_cron_payload(content, "deferred delivery"), quiet,
                               not_before=due, expires_at=due + timedelta(days=1),
                               for_failure=for_failure)
        if for_failure:
            job["_quiet_deferred_failure"] = True
        state = {"execution_id": execution_id, "status": "deferred",
                 "release_at": due.isoformat(), "targets": len(rows)}
        job["last_quiet_window"] = state
        from cron.jobs import update_job
        update_job(job["id"], {"last_quiet_window": state})
        logger.info("Job '%s': %s human notification(s) deferred until %s",
                    job["id"], len(rows), due.isoformat())
    except Exception as exc:
        errors.append(f"quiet-window persistence failed: {exc}")
        logger.error("Job '%s': quiet-window persistence failed", job["id"], exc_info=True)
    return keys
