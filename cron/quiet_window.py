"""Explicit owner-configured notification window for cron human destinations.

Execution is never gated. Compare UTC instants projected into the owner's IANA
zone, rather than constructing nonexistent or ambiguous local wall instants.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def policy(config: dict | None) -> tuple[ZoneInfo, int, int] | None:
    """Return a validated policy or None when absent; malformed opt-ins fail visibly."""
    cron = (config or {}).get("cron", {})
    raw = cron.get("quiet_window") if isinstance(cron, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {"timezone", "start", "end"}:
        raise ValueError("cron.quiet_window requires exactly timezone, start and end")
    name = raw["timezone"]
    if not isinstance(name, str) or not name or name.startswith("/") or ".." in name:
        raise ValueError("cron.quiet_window.timezone must be an IANA zone")
    try:
        zone = ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError) as exc:
        raise ValueError("cron.quiet_window.timezone must be a valid IANA zone") from exc
    def minute(value: object) -> int:
        if (not isinstance(value, str) or len(value) != 5 or value[2] != ":"
                or not (value[:2] + value[3:]).isdigit()):
            raise ValueError("cron.quiet_window start/end must be HH:MM (24-hour)")
        hour, minute_part = int(value[:2]), int(value[3:])
        if hour > 23 or minute_part > 59:
            raise ValueError("cron.quiet_window start/end must be HH:MM (24-hour)")
        return hour * 60 + minute_part
    start, end = minute(raw["start"]), minute(raw["end"])
    if start == end:
        raise ValueError("cron.quiet_window start and end must differ")
    return zone, start, end


def release_at(instant: datetime, configured: tuple[ZoneInfo, int, int]) -> datetime | None:
    """Next UTC eligible minute, or None outside quiet time.

    Walking real UTC minutes resolves spring gaps at the first existing eligible
    minute and both fall folds by the actual projected wall time. A window never
    exceeds one local day; two UTC days cover the longest transition.
    """
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("instant must be timezone-aware")
    zone, start, end = configured
    def quiet(value: datetime) -> bool:
        wall = value.astimezone(zone)
        clock = wall.hour * 60 + wall.minute
        return start <= clock < end if start < end else clock >= start or clock < end
    at = instant.astimezone(UTC)
    if not quiet(at):
        return None
    candidate = at.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(48 * 60):
        if not quiet(candidate):
            return candidate
        candidate += timedelta(minutes=1)
    raise ValueError("quiet window did not end within 48 hours")


def human_target(target: dict) -> bool:
    """Machine destinations and Bot Chat retain their unchanged semantics."""
    return target.get("platform", "").lower() not in {
        "webhook", "wecom_callback", "bot-chat", "local", "api_server"}
