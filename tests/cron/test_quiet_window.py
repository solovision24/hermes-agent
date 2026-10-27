"""Owner opt-in quiet notification window; cron work remains independent of delivery."""
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cron import delivery_queue, quiet_window, scheduler, scheduler_delivery, target_receipts
from hermes_cli.config import validate_config_structure


def at(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def cfg(zone="America/New_York", start="22:00", end="07:00"):
    return {"cron": {"quiet_window": {"timezone": zone, "start": start, "end": end}}}


def test_absent_and_invalid_config():
    assert quiet_window.policy({}) is None
    assert quiet_window.policy({"cron": {"quiet_window": None}}) is None
    for invalid in (cfg("Mars/Base"), cfg(start="24:00"), cfg(start="12:00", end="12:00"),
                    {"cron": {"quiet_window": {"start": "22:00"}}}):
        assert any(i.severity == "error" and "cron.quiet_window" in i.message
                   for i in validate_config_structure(invalid))
        with pytest.raises(ValueError):
            quiet_window.policy(invalid)


def test_boundaries_non_hour_and_overnight():
    zone = quiet_window.policy(cfg("Asia/Kathmandu", "22:00", "07:00"))
    assert quiet_window.release_at(at(2026, 10, 27, 16, 15), zone) == at(2026, 10, 28, 1, 15)
    assert quiet_window.release_at(at(2026, 10, 27, 1, 15), zone) is None
    daytime = quiet_window.policy(cfg("UTC", "09:00", "17:00"))
    assert quiet_window.release_at(at(2026, 10, 27, 9), daytime) == at(2026, 10, 27, 17)
    assert quiet_window.release_at(at(2026, 10, 27, 17), daytime) is None


def test_dst_gap_and_fold_are_real_instants():
    spring = quiet_window.policy(cfg(start="01:00", end="02:30"))
    assert quiet_window.release_at(at(2026, 3, 8, 6, 30), spring) == at(2026, 3, 8, 7)
    fall = quiet_window.policy(cfg(start="01:00", end="02:00"))
    assert quiet_window.release_at(at(2026, 11, 1, 5, 30), fall) == at(2026, 11, 1, 7)
    assert quiet_window.release_at(at(2026, 11, 1, 6, 30), fall) == at(2026, 11, 1, 7)


def test_live_multitarget_deferral_restart_and_one_release(tmp_path, monkeypatch):
    from cron import jobs
    from gateway import media_policy
    from gateway import config as gateway_config

    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    clock = [at(2026, 10, 28, 3)]  # 23:00 EDT
    monkeypatch.setattr(target_receipts, "_now", lambda: clock[0])
    monkeypatch.setattr("hermes_time.now", lambda: clock[0])
    monkeypatch.setattr("cron.scheduler_delivery._sched.load_config", lambda: cfg())
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: object())
    monkeypatch.setattr(media_policy, "apply_media_policy_env", lambda _: None)
    job = {"id": "job", "name": "Brief", "execution_id": "run-1", "deliver": "telegram,webhook"}
    monkeypatch.setattr(jobs, "get_job", lambda _: job)
    monkeypatch.setattr(jobs, "update_job", lambda _id, values: job.update(values))
    targets = [{"platform": "telegram", "chat_id": "original", "thread_id": "topic"},
               {"platform": "webhook", "chat_id": "machine"}]
    monkeypatch.setattr(scheduler_delivery, "_resolve_delivery_targets", lambda *_a, **_k: targets)
    sent = []
    monkeypatch.setattr(scheduler_delivery, "_prepare_target_delivery", lambda _j, target, **_kw:
                        SimpleNamespace(where=target["chat_id"], live_adapter_ready=True))
    monkeypatch.setattr(scheduler_delivery, "_deliver_via_live_adapter",
                        lambda t, *_a, **_kw: sent.append(t.where) or True)
    assert scheduler_delivery._deliver_result(job, "brief", adapters={}, loop=None) is None
    assert sent == ["machine"]
    assert job["last_quiet_window"]["status"] == "deferred"
    receipt = target_receipts.get_target_receipts("run-1")[0]
    assert receipt["status"] == "pending"
    assert receipt["not_before"] == at(2026, 10, 28, 11).isoformat(timespec="microseconds")
    assert scheduler_delivery._deliver_result(job, "brief", adapters={}, loop=None) is None
    assert sent == ["machine", "machine"]
    targets[0]["chat_id"] = "changed"
    monkeypatch.setattr(target_receipts, "_PROCESS_ID", "replacement")
    clock[0] += timedelta(hours=8)
    assert scheduler.drain_delivery_queue({}, None) == 1
    assert sent[-1] == "original"
    assert target_receipts.get_target_receipts("run-1")[0]["status"] == "delivered"
    assert scheduler.drain_delivery_queue({}, None) == 0
    assert sent.count("original") == 1


def test_queue_failure_does_not_leak_human_delivery(tmp_path, monkeypatch):
    from gateway import config as gateway_config
    from gateway import media_policy
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    monkeypatch.setattr("cron.scheduler_delivery._sched.load_config", lambda: cfg())
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: object())
    monkeypatch.setattr(media_policy, "apply_media_policy_env", lambda _: None)
    monkeypatch.setattr(scheduler_delivery, "_resolve_delivery_targets", lambda *_a, **_k:
                        [{"platform": "telegram", "chat_id": "human"}])
    monkeypatch.setattr("hermes_time.now", lambda: at(2026, 10, 28, 3))
    monkeypatch.setattr(target_receipts, "enqueue_targets", Mock(side_effect=OSError("disk full")))
    send = Mock()
    monkeypatch.setattr(scheduler_delivery, "_prepare_target_delivery", send)
    error = scheduler_delivery._deliver_result({"id": "job", "execution_id": "run"}, "brief", adapters={})
    assert "persistence failed" in error
    send.assert_not_called()


def test_completed_one_shot_releases_but_paused_and_deleted_do_not(tmp_path, monkeypatch):
    from cron import jobs
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    monkeypatch.setattr(target_receipts, "_now", lambda: at(2026, 10, 28, 12))
    states = {"one": {"enabled": False, "state": "completed"},
              "pause": {"enabled": False, "state": "paused"}}
    monkeypatch.setattr(jobs, "get_job", lambda job_id: states.get(job_id))
    for job_id in ("one", "pause", "deleted"):
        target_receipts.enqueue_targets(job_id, job_id, "result",
            [{"kind": "human", "platform": "telegram", "chat_id": job_id}],
            not_before=at(2026, 10, 28, 11), expires_at=at(2026, 10, 29, 11))
    send = Mock(return_value=("delivered", None))
    assert target_receipts.release_targets(send) == 2
    send.assert_called_once()
    assert send.call_args.args[0]["chat_id"] == "one"
    assert target_receipts.get_target_receipts("pause")[0]["status"] == "pending"
    assert target_receipts.get_target_receipts("deleted")[0]["status"] == "suppressed"
    states["pause"]["enabled"] = True
    assert target_receipts.release_targets(send) == 1
    assert target_receipts.get_target_receipts("pause")[0]["status"] == "delivered"
