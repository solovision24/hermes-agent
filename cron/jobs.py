"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import copy
import json
import logging
import tempfile
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now

try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False

# =============================================================================
# Configuration
# =============================================================================

HERMES_DIR = get_hermes_home()
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
OUTPUT_DIR = CRON_DIR / "output"
PROFILES_DIR = HERMES_DIR / "profiles"
ONESHOT_GRACE_SECONDS = 120

# Legacy name -> agent_id mapping (for migrating OpenClaw-imported jobs)
_LEGACY_AGENT_NAMES = {
    "halo": "halo",
    "orion (cto)": "orion",
    "orion": "orion",
    "nova": "nova",
    "vector": "vector",
    "atlas": "atlas",
    "sterling": "sterling",
    "canon (docs)": "canon",
    "canon": "canon",
    "dev": "dev",
    "luma": "luma",
    "snip": "snip",
    "elon": "elon",
    "forge": "forge",
    "quill": "quill",
    "chip": "chip",
    "knox": "knox",
    "sentinel": "sentinel",
    "pulse": "pulse",
    "chase": "chase",
    "ledger": "ledger",
    "haven": "haven",
    "conductor": "conductor",
}


def _resolve_agent_id(name: Optional[str]) -> Optional[str]:
    """Infer agent_id from a job name like 'Orion — Daily Browser Daemon Restart'."""
    if not name:
        return None
    key = name.lower()
    for prefix, agent_id in _LEGACY_AGENT_NAMES.items():
        if prefix in key:
            if (PROFILES_DIR / agent_id / "SOUL.md").exists():
                return agent_id
    return None


def _normalize_skill_list(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    """Normalize legacy/single-skill and multi-skill inputs into a unique ordered list."""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)
    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _apply_skill_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a job dict with canonical `skills` and legacy `skill` fields aligned."""
    normalized = dict(job)
    skills = _normalize_skill_list(normalized.get("skill"), normalized.get("skills"))
    normalized["skills"] = skills
    normalized["skill"] = skills[0] if skills else None
    return normalized


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def ensure_dirs():
    """Ensure cron directories exist with secure permissions."""
    CRON_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _secure_dir(CRON_DIR)
    _secure_dir(OUTPUT_DIR)


# =============================================================================
# Schedule Parsing
# =============================================================================

def parse_duration(s: str) -> int:
    """Parse duration string into minutes. Examples: '30m' -> 30, '2h' -> 120, '1d' -> 1440."""
    s = s.strip().lower()
    match = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(f"Invalid duration: '{s}'. Use format like '30m', '2h', or '1d'")
    value = int(match.group(1))
    unit = match.group(2)[0]
    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    return value * multipliers[unit]


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """Parse schedule string into structured format."""
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()

    if schedule_lower.startswith("every "):
        duration_str = schedule[6:].strip()
        minutes = parse_duration(duration_str)
        return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}

    parts = schedule.split()
    if len(parts) >= 5 and all(re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]):
        if not HAS_CRONITER:
            raise ValueError("Cron expressions require 'croniter' package. Install with: pip install croniter")
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{schedule}': {e}")
        return {"kind": "cron", "expr": schedule, "display": schedule}

    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.astimezone()
            return {"kind": "once", "run_at": dt.isoformat(), "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"}
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")

    try:
        minutes = parse_duration(schedule)
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {"kind": "once", "run_at": run_at.isoformat(), "display": f"once in {original}"}
    except ValueError:
        pass

    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Duration: '30m', '2h', '1d' (one-shot)\n"
        f"  - Interval: 'every 30m', 'every 2h' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _ensure_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in Hermes configured timezone."""
    target_tz = _hermes_now().tzinfo
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz).astimezone(target_tz)
    return dt.astimezone(target_tz)


def _recoverable_oneshot_run_at(
    schedule: Dict[str, Any],
    now: datetime,
    *,
    last_run_at: Optional[str] = None,
) -> Optional[str]:
    """Return a one-shot run time if it is still eligible to fire."""
    if schedule.get("kind") != "once":
        return None
    if last_run_at:
        return None
    run_at = schedule.get("run_at")
    if not run_at:
        return None
    run_at_dt = _ensure_aware(datetime.fromisoformat(run_at))
    if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


def _compute_grace_seconds(schedule: dict) -> int:
    """Compute how late a job can be and still catch up instead of fast-forwarding."""
    MIN_GRACE = 120
    MAX_GRACE = 7200
    kind = schedule.get("kind")
    if kind == "interval":
        period_seconds = schedule.get("minutes", 1) * 60
        grace = period_seconds // 2
        return max(MIN_GRACE, min(grace, MAX_GRACE))
    if kind == "cron" and HAS_CRONITER:
        try:
            now = _hermes_now()
            cron = croniter(schedule["expr"], now)
            first = cron.get_next(datetime)
            second = cron.get_next(datetime)
            period_seconds = int((second - first).total_seconds())
            grace = period_seconds // 2
            return max(MIN_GRACE, min(grace, MAX_GRACE))
        except Exception:
            pass
    return MIN_GRACE


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """Compute the next run time for a schedule."""
    now = _hermes_now()
    if schedule["kind"] == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)
    elif schedule["kind"] == "interval":
        minutes = schedule["minutes"]
        if last_run_at:
            last = _ensure_aware(datetime.fromisoformat(last_run_at))
            next_run = last + timedelta(minutes=minutes)
        else:
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()
    elif schedule["kind"] == "cron":
        if not HAS_CRONITER:
            return None
        cron = croniter(schedule["expr"], now)
        next_run = cron.get_next(datetime)
        return next_run.isoformat()
    return None


# =============================================================================
# Job CRUD Operations
# =============================================================================

def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from storage, auto-migrating agent_id from job names."""
    ensure_dirs()
    if not JOBS_FILE.exists():
        return []
    try:
        with open(JOBS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            jobs = data.get("jobs", [])
    except json.JSONDecodeError:
        try:
            with open(JOBS_FILE, 'r', encoding='utf-8') as f:
                data = json.loads(f.read(), strict=False)
                jobs = data.get("jobs", [])
                if jobs:
                    save_jobs(jobs)
                    logger.warning("Auto-repaired jobs.json (had invalid control characters)")
        except Exception:
            return []
    except IOError:
        return []

    # Auto-migrate: infer agent_id from legacy job names (OpenClaw import)
    migrated = 0
    for job in jobs:
        if job.get("agent_id") is None:
            inferred = _resolve_agent_id(job.get("name"))
            if inferred:
                job["agent_id"] = inferred
                migrated += 1
    if migrated:
        save_jobs(jobs)
        logger.info("Auto-migrated %d job(s) to agent_id based on job name", migrated)

    return jobs


def save_jobs(jobs: List[Dict[str, Any]]):
    """Save all jobs to storage."""
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=str(JOBS_FILE.parent), suffix='.tmp', prefix='.jobs_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"jobs": jobs, "updated_at": _hermes_now().isoformat()}, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, JOBS_FILE)
        _secure_file(JOBS_FILE)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def create_job(
    prompt: str,
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new cron job."""
    parsed_schedule = parse_schedule(schedule)
    if repeat is not None and repeat <= 0:
        repeat = None
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1
    if deliver is None:
        deliver = "origin" if origin else "local"

    job_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()
    normalized_skills = _normalize_skill_list(skill, skills)
    normalized_model = str(model).strip() if isinstance(model, str) else None
    normalized_provider = str(provider).strip() if isinstance(provider, str) else None
    normalized_base_url = str(base_url).strip().rstrip("/") if isinstance(base_url, str) else None
    normalized_model = normalized_model or None
    normalized_provider = normalized_provider or None
    normalized_base_url = normalized_base_url or None

    if agent_id:
        if not (PROFILES_DIR / agent_id / "SOUL.md").exists():
            logger.warning(
                "Job '%s': agent_id '%s' not found in profiles -- job will run as default Hermes identity",
                name or job_id,
                agent_id,
            )
            agent_id = None

    label_source = (prompt or (normalized_skills[0] if normalized_skills else None)) or "cron job"
    job = {
        "id": job_id,
        "name": name or label_source[:50].strip(),
        "prompt": prompt,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": normalized_model,
        "provider": normalized_provider,
        "base_url": normalized_base_url,
        "agent_id": agent_id,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {"times": repeat, "completed": 0},
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": compute_next_run(parsed_schedule),
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "deliver": deliver,
        "origin": origin,
    }
    jobs = load_jobs()
    jobs.append(job)
    save_jobs(jobs)
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return _apply_skill_fields(job)
    return None


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """List all jobs, optionally including disabled ones."""
    jobs = [_apply_skill_fields(j) for j in load_jobs()]
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    return jobs


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a job by ID, refreshing derived schedule fields when needed."""
    jobs = load_jobs()
    for i, job in enumerate(jobs):
        if job["id"] != job_id:
            continue
        updated = _apply_skill_fields({**job, **updates})
        schedule_changed = "schedule" in updates
        if "skills" in updates or "skill" in updates:
            normalized_skills = _normalize_skill_list(updated.get("skill"), updated.get("skills"))
            updated["skills"] = normalized_skills
            updated["skill"] = normalized_skills[0] if normalized_skills else None
        if schedule_changed:
            updated_schedule = updated["schedule"]
            updated["schedule_display"] = updates.get(
                "schedule_display",
                updated_schedule.get("display", updated.get("schedule_display")),
            )
            if updated.get("state") != "paused":
                updated["next_run_at"] = compute_next_run(updated_schedule)
        if updated.get("enabled", True) and updated.get("state") != "paused" and not updated.get("next_run_at"):
            updated["next_run_at"] = compute_next_run(updated["schedule"])
        new_agent_id = updates.get("agent_id")
        if new_agent_id and not (PROFILES_DIR / new_agent_id / "SOUL.md").exists():
            logger.warning("Job '%s': agent_id '%s' not found -- clearing", job_id, new_agent_id)
            updated["agent_id"] = None
        jobs[i] = updated
        save_jobs(jobs)
        return _apply_skill_fields(jobs[i])
    return None


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pause a job without deleting it."""
    return update_job(job_id, {
        "enabled": False,
        "state": "paused",
        "paused_at": _hermes_now().isoformat(),
        "paused_reason": reason,
    })


def resume_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Resume a paused job and compute the next future run from now."""
    job = get_job(job_id)
    if not job:
        return None
    next_run_at = compute_next_run(job["schedule"])
    return update_job(job_id, {
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "next_run_at": next_run_at,
    })


def trigger_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Schedule a job to run on the next scheduler tick."""
    job = get_job(job_id)
    if not job:
        return None
    return update_job(job_id, {
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "next_run_at": _hermes_now().isoformat(),
    })


def remove_job(job_id: str) -> bool:
    """Remove a job by ID."""
    jobs = load_jobs()
    original_len = len(jobs)
    jobs = [j for j in jobs if j["id"] != job_id]
    if len(jobs) < original_len:
        save_jobs(jobs)
        return True
    return False


def mark_job_run(job_id: str, success: bool, error: Optional[str] = None):
    """Mark a job as having been run."""
    jobs = load_jobs()
    for i, job in enumerate(jobs):
        if job["id"] != job_id:
            continue
        now = _hermes_now().isoformat()
        job["last_run_at"] = now
        job["last_status"] = "ok" if success else "error"
        job["last_error"] = error if not success else None
        if job.get("repeat"):
            job["repeat"]["completed"] = job["repeat"].get("completed", 0) + 1
            times = job["repeat"].get("times")
            completed = job["repeat"]["completed"]
            if times is not None and times > 0 and completed >= times:
                jobs.pop(i)
                save_jobs(jobs)
                return
        job["next_run_at"] = compute_next_run(job["schedule"], now)
        if job["next_run_at"] is None:
            job["enabled"] = False
            job["state"] = "completed"
        elif job.get("state") != "paused":
            job["state"] = "scheduled"
        save_jobs(jobs)
        return
    save_jobs(jobs)


def advance_next_run(job_id: str) -> bool:
    """Preemptively advance next_run_at for a recurring job before execution."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] != job_id:
            continue
        kind = job.get("schedule", {}).get("kind")
        if kind not in ("cron", "interval"):
            return False
        now = _hermes_now().isoformat()
        new_next = compute_next_run(job["schedule"], now)
        if new_next and new_next != job.get("next_run_at"):
            job["next_run_at"] = new_next
            save_jobs(jobs)
            return True
        return False
    return False


def get_due_jobs() -> List[Dict[str, Any]]:
    """Get all jobs that are due to run now."""
    now = _hermes_now()
    raw_jobs = load_jobs()
    jobs = [_apply_skill_fields(j) for j in copy.deepcopy(raw_jobs)]
    due = []
    needs_save = False
    for job in jobs:
        if not job.get("enabled", True):
            continue
        next_run = job.get("next_run_at")
        if not next_run:
            recovered_next = _recoverable_oneshot_run_at(
                job.get("schedule", {}), now, last_run_at=job.get("last_run_at"))
            if not recovered_next:
                continue
            job["next_run_at"] = recovered_next
            next_run = recovered_next
            logger.info("Job '%s' had no next_run_at; recovering one-shot run at %s",
                        job.get("name", job["id"]), recovered_next)
            for rj in raw_jobs:
                if rj["id"] == job["id"]:
                    rj["next_run_at"] = recovered_next
                    needs_save = True
                    break
        next_run_dt = _ensure_aware(datetime.fromisoformat(next_run))
        if next_run_dt <= now:
            schedule = job.get("schedule", {})
            kind = schedule.get("kind")
            grace = _compute_grace_seconds(schedule)
            if kind in ("cron", "interval") and (now - next_run_dt).total_seconds() > grace:
                new_next = compute_next_run(schedule, now.isoformat())
                if new_next:
                    logger.info(
                        "Job '%s' missed its scheduled time (%s, grace=%ds). Fast-forwarding to next run: %s",
                        job.get("name", job["id"]), next_run, grace, new_next)
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["next_run_at"] = new_next
                            needs_save = True
                            break
                    continue
            due.append(job)
    if needs_save:
        save_jobs(raw_jobs)
    return due


def save_job_output(job_id: str, output: str):
    """Save job output to file."""
    ensure_dirs()
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(job_output_dir)
    timestamp = _hermes_now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = job_output_dir / f"{timestamp}.md"
    fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix='.tmp', prefix='output_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_file)
        _secure_file(output_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return output_file
