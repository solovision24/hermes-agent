import asyncio
import sqlite3
from pathlib import Path
from urllib.error import URLError

import pytest

from gateway.config import Platform
from gateway.kanban_watchers import (
    _acquire_singleton_lock,
    _release_singleton_lock,
)
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from tools import operational_sender


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []
        self.outbound_attempts = []

    async def send(self, chat_id, text, metadata=None):
        self.outbound_attempts.append((chat_id, text, metadata))
        raise AssertionError("Halo adapter must not send Kanban Telegram notices")

    async def send_multiple_images(self, **kwargs):
        self.outbound_attempts.append(("send_multiple_images", kwargs))
        raise AssertionError("Halo adapter must not upload Kanban artifacts")

    async def send_video(self, **kwargs):
        self.outbound_attempts.append(("send_video", kwargs))
        raise AssertionError("Halo adapter must not upload Kanban artifacts")

    async def send_document(self, **kwargs):
        self.outbound_attempts.append(("send_document", kwargs))
        raise AssertionError("Halo adapter must not upload Kanban artifacts")

    async def handle_message(self, event):
        self.handled.append(event)

    def extract_local_files(self, text):
        return [], text

class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")

    def fake_api(_token, method, data):
        if method == "getMe":
            return {"ok": True, "result": {
                "id": operational_sender.EXPECTED_BOT_ID,
                "username": operational_sender.EXPECTED_USERNAME,
                "is_bot": True,
            }}
        delivery = {"message_id": 1,
                    "from": {"id": operational_sender.EXPECTED_BOT_ID,
                             "username": operational_sender.EXPECTED_USERNAME,
                             "is_bot": True},
                    "chat": {"id": operational_sender.DEFAULT_CHAT_ID}}
        runner.adapters[Platform.TELEGRAM].sent.append({
            "chat_id": operational_sender.DEFAULT_CHAT_ID,
            "text": data.get("text", ""),
            "metadata": {},
        })
        return {"ok": True, "result": delivery}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    # Most tests model the default gateway after its dispatcher acquired the
    # singleton lock. Tests for startup or non-owner gateways clear this.
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_replays_telegram_dm_topic_delivery_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "dm-topic-metadata.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm topic task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            thread_id="20197",
            delivery_mode="notify+wake",
            delivery_metadata={
                "chat_type": "dm",
                "direct_messages_topic_id": "20197",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "462",
                "thread_id": "20197",
            },
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    # Operational Telegram delivery intentionally strips topic/reply metadata.
    assert adapter.sent[0]["metadata"] == {}
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].source.thread_id == "20197"


def test_active_named_profile_subscription_is_delivered(tmp_path, monkeypatch):
    """A sub stamped with the gateway's own named profile uses self.adapters.

    Regression for #71340: on a standalone (non-multiplex) gateway running a
    named profile, _authorization_adapter() used to treat the active name as a
    multiplex secondary, find no _profile_adapters entry, fail closed, and
    rewind the claim forever — silent zero-delivery.
    """
    db_path = tmp_path / "actionable-block.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    reason = "AGE-39 — https://linear.example/AGE-39 — publishing verified."
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="approval", assignee="publisher")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="main",
        )
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert tid in message
    assert "blocked" in message


def test_non_dispatch_gateway_claims_only_its_profile_subscriptions(
    tmp_path, monkeypatch,
):
    """A profile gateway delivers its events while another gateway dispatches."""
    db_path = tmp_path / "cross-profile-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        foreign_tid = kb.create_task(
            conn, title="default-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=foreign_tid,
            platform="telegram",
            chat_id="default-chat",
            notifier_profile="default",
        )
        kb.complete_task(conn, foreign_tid, summary="default done")

        owned_tid = kb.create_task(
            conn, title="writer-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=owned_tid,
            platform="telegram",
            chat_id="writer-chat",
            notifier_profile="writer",
        )
        kb.complete_task(conn, owned_tid, summary="writer done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "writer"
    runner._kanban_dispatcher_lock_handle = None

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [delivery["chat_id"] for delivery in adapter.sent] == [operational_sender.DEFAULT_CHAT_ID]
    assert owned_tid in adapter.sent[0]["text"]
    assert len(_unseen_terminal_events_for(foreign_tid, "default-chat")) == 1


def test_legacy_subscription_requires_confirmed_dispatcher_lock_owner(
    tmp_path, monkeypatch,
):
    """Startup and lock-losing gateways cannot claim legacy notifications."""
    db_path = tmp_path / "legacy-lock-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="legacy-chat",
        )
        kb.complete_task(conn, task_id, summary="legacy done")
    finally:
        conn.close()

    startup_adapter = RecordingAdapter()
    startup_runner = _make_runner(startup_adapter)
    startup_runner._kanban_dispatcher_lock_handle = None
    asyncio.run(_run_one_notifier_tick(monkeypatch, startup_runner))
    assert startup_adapter.sent == []
    assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

    lock_path = tmp_path / ".dispatcher.lock"
    winner_handle, winner_state = _acquire_singleton_lock(lock_path)
    loser_handle, loser_state = _acquire_singleton_lock(lock_path)
    try:
        assert winner_state == "held"
        assert loser_state == "contended"

        loser_adapter = RecordingAdapter()
        loser_runner = _make_runner(loser_adapter)
        loser_runner._kanban_dispatcher_lock_handle = loser_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, loser_runner))
        assert loser_adapter.sent == []
        assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

        winner_adapter = RecordingAdapter()
        winner_runner = _make_runner(winner_adapter)
        winner_runner._kanban_dispatcher_lock_handle = winner_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, winner_runner))
        assert [item["chat_id"] for item in winner_adapter.sent] == [operational_sender.DEFAULT_CHAT_ID]
        assert task_id in winner_adapter.sent[0]["text"]
    finally:
        _release_singleton_lock(loser_handle)
        _release_singleton_lock(winner_handle)


def test_real_telegram_notifier_loop_uses_transport_boundary(monkeypatch, tmp_path):
    """The notifier loop must use the verified sender, never Halo.send()."""
    db_path = tmp_path / "transport-boundary.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="transport proof", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="delivered")
    finally:
        conn.close()

    calls = []
    def fake_api(_token, method, data):
        calls.append((method, data))
        if method == "getMe":
            return {"ok": True, "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": operational_sender.EXPECTED_USERNAME, "is_bot": True}}
        return {"ok": True, "result": {"message_id": 7,
                "from": {"id": operational_sender.EXPECTED_BOT_ID,
                         "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
                "chat": {"id": operational_sender.DEFAULT_CHAT_ID}}}
    monkeypatch.setattr(operational_sender, "_api_call", fake_api)

    class HaloForbidden:
        async def send(self, *_args, **_kwargs):
            raise AssertionError("Halo adapter must not send Kanban Telegram notices")
    runner = _make_runner(HaloForbidden())
    async def one_tick():
        original_sleep = asyncio.sleep
        async def stop_after_initial(delay):
            if delay == 5:
                return
            runner._running = False
            await original_sleep(0)
        monkeypatch.setattr(asyncio, "sleep", stop_after_initial)
        await runner._kanban_notifier_watcher(interval=1)
    asyncio.run(one_tick())

    assert calls[0][0] == "getMe"
    assert calls[1] == ("sendMessage", {"chat_id": operational_sender.DEFAULT_CHAT_ID,
                                          "text": calls[1][1]["text"]})
    assert "message_thread_id" not in calls[1][1]


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class ReportedFailureAdapter:
    """Adapter that REPORTS failure via SendResult(success=False) instead of
    raising — the exact contract the Telegram adapter uses for 'Not connected'
    and degraded-send paths."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not connected")


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "crashed" in adapter.sent[0]["text"].lower()

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "crashed" in adapter.sent[1]["text"].lower()


def test_notifier_wakeup_uses_subscription_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "chat-type-wakeup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm requester",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"

    # The wake must resume the creator's real DM session key — the whole bug
    # was that a hardcoded chat_type="group" made build_session_key() produce
    # a group-scoped key (a NEW session) instead of the ":dm:<chat_id>" shape
    # the original conversation runs under (#56580 / #68874).
    from gateway.session import build_session_key

    wake_key = build_session_key(adapter.handled[0].source)
    assert wake_key == "agent:main:telegram:dm:chat-dm"
    assert ":group:" not in wake_key


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_isolates_per_subscription_failure(tmp_path, monkeypatch):
    """One bad subscription must not block delivery for all others.

    Regression for #59269: when claim_unseen_events_for_sub raises for one
    subscription, the entire notifier tick used to abort — silently blocking
    delivery for every other subscription.
    """
    db_path = tmp_path / "isolation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Create two tasks with subscriptions and complete both. The BAD task is
    # created first: list_notify_subs() has no ORDER BY, so SQLite's natural
    # scan returns insertion order — the failing subscription must be
    # processed BEFORE the good one or this test passes even without the
    # per-subscription isolation (the good delivery happens before the tick
    # aborts). A deterministic-order shim below removes the reliance on the
    # scan order entirely.
    conn = kb.connect()
    try:
        tid_bad = kb.create_task(conn, title="bad task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_bad, platform="telegram", chat_id="chat-bad")
        kb.complete_task(conn, tid_bad, summary="done")

        tid_good = kb.create_task(conn, title="good task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_good, platform="telegram", chat_id="chat-good")
        kb.complete_task(conn, tid_good, summary="done")
    finally:
        conn.close()

    original_claim = kb.claim_unseen_events_for_sub

    def selective_claim(conn, task_id, **kwargs):
        if task_id == tid_bad:
            raise RuntimeError("simulated DB corruption for bad task")
        return original_claim(conn, task_id=task_id, **kwargs)

    monkeypatch.setattr(kb, "claim_unseen_events_for_sub", selective_claim)

    # Force the failing subscription to be iterated FIRST regardless of the
    # unordered SELECT's scan order.
    original_list = kb.list_notify_subs

    def bad_first(conn, task_id=None, **kwargs):
        subs = original_list(conn, task_id, **kwargs)
        return sorted(subs, key=lambda s: 0 if s["task_id"] == tid_bad else 1)

    monkeypatch.setattr(kb, "list_notify_subs", bad_first)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The good task must still be delivered despite the bad task failing.
    assert len(adapter.sent) == 1
    assert tid_good in adapter.sent[0]["text"]


def test_notifier_delivers_block_loop_detected_triage_ping(tmp_path, monkeypatch):
    """A `block_loop_detected` event must reach the subscriber as a triage ping.

    Regression for the silent-triage gap (PR #62712): kanban_db routes a task
    to `triage` after BLOCK_RECURRENCE_LIMIT re-blocks for the same cause and
    emits ONLY a `block_loop_detected` event — no `blocked`/`status` event.
    Before `block_loop_detected` joined TERMINAL_KINDS with its own message
    branch, that one transition (the whole point of which is to force human
    attention) produced zero notification and the task stalled in triage
    silently.
    """
    db_path = tmp_path / "block-loop.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "block_loop_detected must produce a notification"
    text = adapter.sent[0]["text"]
    assert "TRIAGE" in text
    assert tid in text
    assert "needs credentials" in text
    # Cursor advanced: the event is claimed and not re-delivered.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []


def test_notifier_subscription_survives_done_reopen_until_archive(
    tmp_path, monkeypatch,
):
    """Done is reversible; archive alone ends notification ownership."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "done-reopen-archive.db"))
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review continuation", assignee="worker",
                             session_id="origin-session")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="origin-chat",
                          thread_id="origin-thread", user_id="origin-user",
                          chat_type="group", notifier_profile="reviewer",
                          delivery_mode="notify+wake")
        assert kb.complete_task(conn, tid, summary="first completion")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.sent[0]["chat_id"] == operational_sender.DEFAULT_CHAT_ID
    assert adapter.sent[0]["metadata"] == {}
    assert adapter.handled[0].source.thread_id == "origin-thread"
    assert adapter.outbound_attempts == []

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        first_cursor = subs[0]["last_event_id"]
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.outbound_attempts == []

    with kb.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "status", {"status": "ready"})
        assert kb.complete_task(conn, tid, summary="corrected completion")
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    assert adapter.sent[1]["text"].endswith("→ ready")
    assert "corrected completion" in adapter.sent[2]["text"]
    assert adapter.outbound_attempts == []

    with kb.connect() as conn:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1 and subs[0]["last_event_id"] > first_cursor
        assert kb.archive_task(conn, tid)
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    assert adapter.outbound_attempts == []
    with kb.connect() as conn:
        assert kb.list_notify_subs(conn, tid) == []


def _wake_text(adapter):
    assert len(adapter.handled) == 1
    return getattr(adapter.handled[0], "text", "") or ""


def _review_handoff_task(*, delivery_mode="notify+wake"):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="implement the thing", assignee="worker",
                             session_id="agent:main:telegram:dm:chat-1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                          chat_type="dm", delivery_mode=delivery_mode)
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(conn, tid,
                                 summary="PR ready: https://example.invalid/pr/7",
                                 expected_run_id=run_id)
        return tid


def test_review_requested_wakes_the_origin_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-wake.db"))
    kb.init_db()
    tid = _review_handoff_task()
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert "ready for review" in adapter.sent[0]["text"]
    assert adapter.outbound_attempts == []
    wake = _wake_text(adapter)
    assert tid in wake
    assert "PR ready: https://example.invalid/pr/7" in wake


def test_block_loop_detected_wakes_the_origin_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "triage-wake.db"))
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loops forever", assignee="worker",
                             session_id="agent:main:telegram:dm:chat-1")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1",
                          chat_type="dm", delivery_mode="notify+wake")
        kb._append_event(conn, tid, "block_loop_detected",
                          {"reason": "needs credentials", "kind": "needs_input"})
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert tid in _wake_text(adapter)
    assert adapter.outbound_attempts == []


def test_review_requested_does_not_wake_a_notify_only_subscription(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-notify.db"))
    kb.init_db()
    _review_handoff_task(delivery_mode="notify")
    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))
    assert len(adapter.sent) == 1
    assert adapter.handled == []
    assert adapter.outbound_attempts == []


def _run_real_operational_tick(monkeypatch, runner):
    """Run one loop iteration without replacing the operational sender."""
    real_sleep = asyncio.sleep

    calls = 0

    async def stop_after_tick(delay):
        nonlocal calls
        if delay == 5:
            calls += 1
            if calls == 1:
                return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", stop_after_tick)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))


def _seed_event(db_path, monkeypatch, *, kind="blocked", session_id=None):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="operational matrix", assignee="worker",
                             session_id=session_id)
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="ignored",
                          thread_id="topic-1", delivery_mode="notify+wake")
        kb._append_event(
            conn, tid, kind,
            {"reason": "matrix"} if kind == "blocked" else {},
        )
    return tid


def _subscription_cursor(tid):
    with kb.connect() as conn:
        return kb.list_notify_subs(conn, tid)[0]["last_event_id"]


@pytest.mark.parametrize("identity", [
    {"id": operational_sender.EXPECTED_BOT_ID + 1,
     "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
    {"id": operational_sender.EXPECTED_BOT_ID,
     "username": operational_sender.EXPECTED_USERNAME, "is_bot": False},
])
def test_real_notifier_fails_closed_before_wrong_identity_send(tmp_path, monkeypatch, identity):
    tid = _seed_event(tmp_path / "wrong-identity.db", monkeypatch)
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")
    calls = []

    def fake_api(_token, method, data):
        calls.append((method, data))
        return {"ok": True, "result": identity} if method == "getMe" else pytest.fail("send must not run")

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _run_real_operational_tick(monkeypatch, runner)
    # The task-created event is cursor 1; the failed notification must
    # not advance beyond it.
    assert [method for method, _ in calls] == ["getMe"]
    assert _subscription_cursor(tid) == 1
    assert adapter.outbound_attempts == []


def test_real_notifier_missing_token_rewinds_without_halo_fallback(tmp_path, monkeypatch):
    tid = _seed_event(tmp_path / "missing-token.db", monkeypatch)
    monkeypatch.delenv("SOLO_HERMES_BOT_TOKEN", raising=False)
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    _run_real_operational_tick(monkeypatch, runner)
    assert _subscription_cursor(tid) == 1
    assert adapter.sent == []
    assert adapter.outbound_attempts == []


def test_real_notifier_rewinds_failed_transport_then_deduplicates(tmp_path, monkeypatch):
    tid = _seed_event(tmp_path / "retry.db", monkeypatch)
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")
    attempts = []

    def fake_api(_token, method, data):
        attempts.append((method, dict(data)))
        if method == "getMe":
            return {"ok": True, "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": operational_sender.EXPECTED_USERNAME, "is_bot": True}}
        if len([m for m, _ in attempts if m == "sendMessage"]) == 1:
            raise RuntimeError("transport down")
        return {"ok": True, "result": {"message_id": 42,
                "from": {"id": operational_sender.EXPECTED_BOT_ID,
                         "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
                "chat": {"id": operational_sender.DEFAULT_CHAT_ID}}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    first_adapter = RecordingAdapter()
    first = _make_runner(first_adapter)
    _run_real_operational_tick(monkeypatch, first)
    assert _subscription_cursor(tid) == 1
    assert first_adapter.outbound_attempts == []
    second_adapter = RecordingAdapter()
    second = _make_runner(second_adapter)
    _run_real_operational_tick(monkeypatch, second)
    assert _subscription_cursor(tid) >= 2
    assert second_adapter.outbound_attempts == []
    third_adapter = RecordingAdapter()
    third = _make_runner(third_adapter)
    _run_real_operational_tick(monkeypatch, third)
    assert len([m for m, _ in attempts if m == "sendMessage"]) == 2
    assert third_adapter.outbound_attempts == []
    sends = [data for method, data in attempts if method == "sendMessage"]
    assert sends[0]["chat_id"] == operational_sender.DEFAULT_CHAT_ID
    assert "message_thread_id" not in sends[0]


def test_real_notifier_review_events_send_and_preserve_creator_wake(tmp_path, monkeypatch):
    tid = _seed_event(tmp_path / "review-events.db", monkeypatch,
                      kind="review_requested", session_id="agent:main:telegram:dm:ignored")
    with kb.connect() as conn:
        kb._append_event(conn, tid, "changes_requested", {"reason": "please revise"})
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")

    operational_calls = []

    def fake_api(_token, method, data):
        if method == "getMe":
            return {"ok": True, "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": operational_sender.EXPECTED_USERNAME, "is_bot": True}}
        operational_calls.append((method, dict(data)))
        return {"ok": True, "result": {"message_id": 7,
                "from": {"id": operational_sender.EXPECTED_BOT_ID,
                         "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
                "chat": {"id": operational_sender.DEFAULT_CHAT_ID}}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    adapter = RecordingAdapter()
    _run_real_operational_tick(monkeypatch, _make_runner(adapter))
    assert adapter.outbound_attempts == []
    assert _subscription_cursor(tid) >= 2
    assert [method for method, _ in operational_calls] == ["sendMessage", "sendMessage"]
    assert all(call[1]["chat_id"] == operational_sender.DEFAULT_CHAT_ID for call in operational_calls)
    assert all("message_thread_id" not in call[1] for call in operational_calls)
    ping_text = "\n".join(call[1]["text"] for call in operational_calls).lower()
    assert "review requested" in ping_text
    # Installed safe rendering combines the event label with its routing
    # action ("review requested changes/block"). Assert the semantic event,
    # not wording that predates the preserved renderer.
    assert "review requested changes/block" in ping_text
    assert len(adapter.handled) == 1
    wake_text = adapter.handled[0].text.lower()
    assert "review requested" in wake_text
    assert "review requested changes" in wake_text


def test_real_notifier_rewinds_when_operational_response_has_thread(tmp_path, monkeypatch):
    """Telegram must reject a response that leaks topic/thread routing."""
    tid = _seed_event(tmp_path / "thread-proof.db", monkeypatch)
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")
    calls = []

    def fake_api(_token, method, data):
        calls.append(method)
        if method == "getMe":
            return {"ok": True, "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": operational_sender.EXPECTED_USERNAME, "is_bot": True}}
        return {"ok": True, "result": {"message_id": 9,
                "from": {"id": operational_sender.EXPECTED_BOT_ID,
                         "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
                "chat": {"id": operational_sender.DEFAULT_CHAT_ID},
                "message_thread_id": 77}}

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    adapter = RecordingAdapter()
    _run_real_operational_tick(monkeypatch, _make_runner(adapter))

    assert calls == ["getMe", "sendMessage"]
    assert _subscription_cursor(tid) == 1
    assert adapter.outbound_attempts == []


def test_real_notifier_rewinds_partial_multipart_batch_then_retries_and_deduplicates(
    tmp_path, monkeypatch,
):
    """A failed artifact in a real loop rewinds text and the whole batch.

    Replaying an already-uploaded artifact is intentional: the cursor is the
    idempotency boundary, and advancing it after a partial batch would lose
    the remaining artifact permanently.
    """
    db_path = tmp_path / "partial-multipart.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="artifact batch", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="ignored")
        kb._append_event(conn, tid, "completed", {"summary": "done",
                                                    "artifacts": [str(first), str(second)]})

    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "test-token")
    document_attempts = []

    def fake_api(_token, method, data):
        if method == "getMe":
            return {"ok": True, "result": {"id": operational_sender.EXPECTED_BOT_ID,
                    "username": operational_sender.EXPECTED_USERNAME, "is_bot": True}}
        return {"ok": True, "result": {"message_id": 10,
                "from": {"id": operational_sender.EXPECTED_BOT_ID,
                         "username": operational_sender.EXPECTED_USERNAME, "is_bot": True},
                "chat": {"id": operational_sender.DEFAULT_CHAT_ID}}}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (b'{"ok": true, "result": {"message_id": 11, '
                    b'"from": {"id": 8611668567, "username": "solo_hermes_bot", "is_bot": true}, '
                    b'"chat": {"id": "8148316720"}}}')

    def fake_urlopen(request, timeout=20):
        document_attempts.append(request.full_url)
        # The first file reaches Telegram; the second fails. The next tick
        # retries both files after the notifier rewinds the claimed event.
        if len(document_attempts) == 2:
            raise URLError("second document unavailable")
        return Response()

    monkeypatch.setattr(operational_sender, "_api_call", fake_api)
    monkeypatch.setattr(operational_sender, "urlopen", fake_urlopen)
    first_adapter = RecordingAdapter()
    _run_real_operational_tick(monkeypatch, _make_runner(first_adapter))
    # The task-creation event remains the durable cursor floor; the completed
    # event itself was rewound for retry.
    assert _subscription_cursor(tid) == 1
    assert len(document_attempts) == 2
    assert first_adapter.outbound_attempts == []

    second_adapter = RecordingAdapter()
    _run_real_operational_tick(monkeypatch, _make_runner(second_adapter))
    assert len(document_attempts) == 4
    assert _subscription_cursor(tid) >= 2
    assert second_adapter.outbound_attempts == []

    _run_real_operational_tick(monkeypatch, _make_runner(RecordingAdapter()))
    assert len(document_attempts) == 4
