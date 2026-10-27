"""A deferred receipt keeps its conversation and never gets an untracked replay."""
import asyncio
from concurrent.futures import Future
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from cron import delivery_queue, scheduler, scheduler_delivery, target_receipts
from gateway.config import Platform, PlatformConfig


@pytest.fixture
def release_setup(tmp_path, monkeypatch):
    from cron import jobs
    from gateway import config as gateway_config

    clock = [datetime(2026, 10, 28, 3, tzinfo=UTC)]
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel.return_value = None
    job = {"id": "brief", "name": "Brief", "execution_id": "run-1",
           "deliver": "origin", "origin": {"platform": "telegram", "chat_id": "-1001234567890",
                                            "thread_id": "99"}}
    monkeypatch.setattr(delivery_queue, "DELIVERY_DB", tmp_path / "deliveries.db")
    monkeypatch.setattr(target_receipts, "_now", lambda: clock[0])
    monkeypatch.setattr("hermes_time.now", lambda: clock[0])
    monkeypatch.setattr(jobs, "get_job", lambda _id: job)
    monkeypatch.setattr(jobs, "update_job", lambda _id, updates: job.update(updates))
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: config)
    monkeypatch.setattr("cron.scheduler_delivery._sched.load_config", lambda: {
        "cron": {"quiet_window": {"timezone": "America/New_York", "start": "22:00",
                                   "end": "07:00"}, "wrap_response": False}})
    monkeypatch.setattr(scheduler_delivery, "_resolve_delivery_targets", lambda *_a, **_kw: [
        {"platform": "telegram", "chat_id": "-1001234567890", "thread_id": "99"}])
    assert scheduler_delivery._deliver_result(job, "brief", adapters={}, loop=None) is None
    assert target_receipts.get_target_receipts("run-1")[0]["status"] == "pending"
    clock[0] += timedelta(hours=8)
    return job, config


def _run_live(monkeypatch, response):
    sent = []

    async def dispatch(target, text, metadata, transport=None):
        sent.append((target.chat_id, target.thread_id, metadata.copy()))
        return response

    router = MagicMock()
    router._deliver_to_platform = dispatch
    monkeypatch.setattr("gateway.delivery.DeliveryRouter", lambda *_a, **_kw: router)

    def schedule(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr("cron.scheduler_delivery.safe_schedule_threadsafe", schedule,
                        raising=False)
    monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
    loop = MagicMock()
    loop.is_running.return_value = True
    return sent, loop


def test_changed_surface_and_mirror_do_not_move_deferred_topic(release_setup, monkeypatch):
    job, config = release_setup
    config.platforms[Platform.TELEGRAM].extra = {"cron_continuable_surface": "in_channel"}
    job["attach_to_session"] = True
    sent, loop = _run_live(monkeypatch, {"success": True, "message_id": "m-1"})
    adapter = MagicMock()
    adapter.supports_inchannel_continuable = True
    adapter.create_handoff_thread.side_effect = AssertionError("must not open a new thread")
    mirror_flags = []
    monkeypatch.setattr(scheduler_delivery, "_maybe_mirror_cron_delivery",
                        lambda *_a, **kw: mirror_flags.append(kw["enabled"]))
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 1
    assert len(sent) == 1
    assert sent[0][0:2] == ("-1001234567890", "99")
    assert sent[0][2]["thread_id"] == "99"
    assert mirror_flags == [False]
    assert target_receipts.get_target_receipts("run-1")[0]["status"] == "delivered"
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 0


def test_missing_live_transport_never_falls_back_or_reconnects(release_setup, monkeypatch):
    sent = []
    queued = []
    monkeypatch.setattr("tools.send_message_tool._send_to_platform",
                        lambda *_a, **_kw: sent.append(True))
    monkeypatch.setattr(scheduler_delivery, "_queue_for_live_reconnect",
                        lambda *_a, **_kw: queued.append(True))
    assert scheduler.drain_delivery_queue({}, None) == 1
    receipt = target_receipts.get_target_receipts("run-1")[0]
    assert receipt["status"] == "failed"
    assert "live adapter unavailable" in receipt["error"]
    assert sent == queued == []
    assert scheduler.drain_delivery_queue({}, None) == 0


def test_evidence_free_live_ack_is_unknown_without_replay(release_setup, monkeypatch):
    sent, loop = _run_live(monkeypatch, {"success": True})
    queued = []
    monkeypatch.setattr(scheduler_delivery, "_queue_for_live_reconnect",
                        lambda *_a, **_kw: queued.append(True))
    adapter = MagicMock()
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 1
    assert len(sent) == 1
    assert target_receipts.get_target_receipts("run-1")[0]["status"] == "unknown"
    assert queued == []
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 0


def test_live_rejection_does_not_trigger_standalone_or_reconnect(release_setup, monkeypatch):
    sent, loop = _run_live(monkeypatch, {"success": False, "error": "send_path_degraded"})
    standalone = []
    reconnect = []
    monkeypatch.setattr(scheduler_delivery, "_standalone_send",
                        lambda *_a: standalone.append(True))
    monkeypatch.setattr(scheduler_delivery, "_queue_for_live_reconnect",
                        lambda *_a: reconnect.append(True))
    adapter = MagicMock()
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 1
    assert len(sent) == 1
    assert target_receipts.get_target_receipts("run-1")[0]["status"] == "failed"
    assert standalone == reconnect == []
    assert scheduler.drain_delivery_queue({Platform.TELEGRAM: adapter}, loop) == 0
