"""Fork-local cron egress contract for t_e805dab0; no live credentials/network."""

import asyncio
import copy
import io
import json
import os
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs

import pytest
import yaml

from cron import operational_telegram as ops
from cron.scheduler_delivery import _deliver_result

ROOT_TOKEN = "8611668567:" + "synthetic_root_" * 3
PROFILE_TOKEN = "123456789:" + "synthetic_profile_" * 3


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("HERMES_ROOT", raising=False)
    (root / ".env").write_text(f"SOLO_HERMES_BOT_TOKEN={ROOT_TOKEN}\n")
    (root / "config.yaml").write_text("cron:\n  wrap_response: false\n")
    return root


def bot():
    return {"id": ops.BOT_ID, "username": ops.BOT_USERNAME, "is_bot": True}


class TelegramNetwork:
    def __init__(self):
        self.calls = []
        self.transform = lambda method, result: result
        self.error = None

    def __call__(self, request, timeout):
        credential, method = request.full_url.split("/bot", 1)[1].split("/", 1)
        assert credential == ROOT_TOKEN, "only the synthetic root credential may reach cron HTTP"
        assert timeout > 0
        if method == "sendDocument":
            envelope = ("Content-Type: " + request.headers["Content-type"] + "\r\n\r\n").encode()
            parts = BytesParser(policy=default).parsebytes(envelope + request.data).iter_parts()
            data = {part.get_param("name", header="content-disposition"):
                    part.get_payload(decode=True) for part in parts}
        else:
            data = {k: v[0] for k, v in parse_qs(request.data.decode(), keep_blank_values=True).items()}
        self.calls.append((method, data))
        if self.error:
            raise self.error
        sent = {"message_id": len(self.calls), "from": bot(),
                "chat": {"id": int(ops.CHAT_ID), "type": "private"}}
        if method == "sendDocument":
            sent["document"] = {"file_id": "synthetic-file"}
        result = {"ok": True, "result": bot() if method == "getMe" else sent}
        return io.BytesIO(json.dumps(self.transform(method, result)).encode())


@pytest.fixture
def network(home, monkeypatch):
    network = TelegramNetwork()
    monkeypatch.setattr(ops, "urlopen", network)
    return network


@pytest.fixture
def profile_lanes(monkeypatch):
    from cron import scheduler_delivery as delivery

    # Assert bypass at the earliest transport boundary: this also excludes relay
    # adapters, topic creation, and transcript seeding, before any event-loop work.
    prepare = Mock(side_effect=AssertionError("profile transport used"))
    standalone = AsyncMock(side_effect=AssertionError("profile standalone used"))
    mirror = Mock(side_effect=AssertionError("profile transcript seeded"))
    monkeypatch.setattr(delivery, "_prepare_target_delivery", prepare)
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone)
    monkeypatch.setattr("gateway.mirror.mirror_to_session", mirror)
    return prepare, standalone, mirror


def job(deliver="origin", **extra):
    return {"id": "synthetic-cron", "name": "test", "deliver": deliver,
            "attach_to_session": True,
            "origin": {"platform": "telegram", "chat_id": ops.CHAT_ID,
                       "thread_id": "111", "reply_to_message_id": "222",
                       "business_connection_id": "other-bot", "chat_type": "dm"},
            **extra}


@pytest.mark.parametrize("lane", ["live", "relay", "standalone"])
@pytest.mark.parametrize("destination", ["origin", "home", "fallback", "explicit"])
def test_canonical_dm_bypasses_profile_lanes_and_inherited_metadata(
    home, network, profile_lanes, monkeypatch, lane, destination,
):
    from gateway.config import Platform

    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", ops.CHAT_ID)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "333")
    monkeypatch.setenv("TELEGRAM_CRON_THREAD_ID", "444")
    j = job()
    if destination == "home":
        j["deliver"] = "telegram"
    elif destination == "fallback":
        j.pop("origin")
    elif destination == "explicit":
        j["deliver"] = f"telegram:{ops.CHAT_ID}"
    original = copy.deepcopy(j)
    adapter = Mock()
    adapter_key = Platform.TELEGRAM if lane == "live" else Platform.RELAY
    adapters = None if lane == "standalone" else {adapter_key: adapter}
    error = _deliver_result(j, "operational report", adapters=adapters, loop=Mock())
    assert error is None
    assert network.calls == [("getMe", {}), ("sendMessage", {
        "chat_id": ops.CHAT_ID, "text": "operational report"})]
    for spy in profile_lanes:
        spy.assert_not_called()
    assert adapter.mock_calls == []
    assert {k: j[k] for k in original} == original


@pytest.mark.parametrize("deliver,origin", [
    ("telegram:9999", None),
    (f"telegram:{ops.CHAT_ID}:77", None),
    ("origin", {"platform": "telegram", "chat_id": "9999"}),
    ("telegram", None),
])
def test_wrong_destination_refused_without_network_or_fallback(
    network, profile_lanes, monkeypatch, deliver, origin,
):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "9999")
    assert "destination verification failed" in _deliver_result(job(deliver, origin=origin), "report")
    assert network.calls == []
    for spy in profile_lanes:
        spy.assert_not_called()


@pytest.mark.parametrize("multiplex", [False, True])
def test_root_credential_across_named_scopes_and_direct_profile_sender(
    home, network, monkeypatch, multiplex,
):
    from agent.secret_scope import (
        build_profile_secret_scope, current_secret_scope, is_multiplex_active,
        reset_secret_scope, set_multiplex_active, set_secret_scope,
    )
    from cron.scheduler_provider import _profile_cron_scope
    from gateway.config import Platform, load_gateway_config
    from tools.send_message_tool import _send_to_platform

    homes = []
    for name in ("a", "b"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / ".env").write_text(
            f"TELEGRAM_BOT_TOKEN={PROFILE_TOKEN}{name}\nSOLO_HERMES_BOT_TOKEN=999:wrong_profile\n")
        (profile / "config.yaml").write_text("cron:\n  wrap_response: false\n")
        homes.append(profile)
    # A fallback daemon was launched in B with poisoned process credentials.
    monkeypatch.setenv("HERMES_HOME", str(homes[1]))
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", "999:wrong_process")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "888:wrong_process")
    before = dict(os.environ)
    files = {p: p.read_bytes() for p in home.rglob(".env")}
    direct = AsyncMock(return_value={"success": True})
    monkeypatch.setattr("tools.send_message_tool._send_telegram", direct)
    previous = is_multiplex_active()
    set_multiplex_active(multiplex)
    try:
        for profile in (homes[0], homes[1], homes[0]):
            with _profile_cron_scope(profile):
                scope = build_profile_secret_scope(profile)
                token = set_secret_scope(scope, profile_home=str(profile))
                try:
                    assert _deliver_result(job(), "report") is None
                    assert current_secret_scope() == scope
                    cfg = load_gateway_config().platforms[Platform.TELEGRAM]
                    asyncio.run(_send_to_platform(Platform.TELEGRAM, cfg, "different-dm", "direct"))
                    assert direct.call_args.args[0] == PROFILE_TOKEN + profile.name
                    assert direct.call_args.args[1] == "different-dm"
                finally:
                    reset_secret_scope(token)
    finally:
        set_multiplex_active(previous)
    assert len(network.calls) == 6
    assert dict(os.environ) == before
    assert {p: p.read_bytes() for p in files} == files


@pytest.mark.parametrize("credential", [None, "", "invalid", "999:bad/token"])
def test_missing_or_invalid_root_credential_never_uses_environment(
    home, network, profile_lanes, monkeypatch, credential,
):
    (home / ".env").write_text("" if credential is None else f"SOLO_HERMES_BOT_TOKEN={credential}\n")
    monkeypatch.setenv("SOLO_HERMES_BOT_TOKEN", ROOT_TOKEN)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", PROFILE_TOKEN)
    error = _deliver_result(job(), "report")
    assert "root" in error and ("missing" in error or "invalid" in error)
    assert network.calls == []
    for spy in profile_lanes:
        spy.assert_not_called()


@pytest.mark.parametrize("stage,field,value", [
    ("getMe", "id", 123), ("getMe", "username", "profile_bot"), ("getMe", "is_bot", False),
    ("sendMessage", "from", {"id": 123}),
    ("sendMessage", "chat", {"id": 999, "type": "private"}),
    ("sendMessage", "message_id", None),
    ("sendMessage", "message_thread_id", None),
    ("sendMessage", "direct_messages_topic", {"topic_id": 1}),
    ("sendMessage", "reply_to_message", {}),
    ("sendMessage", "business_connection_id", "foreign"),
])
def test_identity_and_send_proof_fail_closed(network, profile_lanes, stage, field, value):
    def transform(method, result):
        if method == stage:
            result["result"][field] = value
        return result

    network.transform = transform
    error = _deliver_result(job(), "report")
    assert ("identity verification" if stage == "getMe" else "delivery proof") in error
    assert len(network.calls) == (1 if stage == "getMe" else 2)
    for spy in profile_lanes:
        spy.assert_not_called()


@pytest.mark.parametrize("response", [{"ok": False, "description": ROOT_TOKEN},
                                      {"ok": True}, [], {"ok": True, "result": []}])
def test_absent_api_proof_is_failure(network, profile_lanes, response, caplog):
    network.transform = lambda method, result: response
    error = _deliver_result(job(), "report")
    assert error and ROOT_TOKEN not in error + caplog.text
    assert len(network.calls) == 1


@pytest.mark.parametrize("kind", ["http", "url", "timeout"])
def test_token_bearing_network_errors_are_safe(network, profile_lanes, caplog, kind):
    url = f"https://api.telegram.org/bot{ROOT_TOKEN}/getMe"
    errors = {"http": HTTPError(url, 401, url, {}, io.BytesIO(url.encode())),
              "url": URLError(url), "timeout": TimeoutError(url)}
    network.error = errors[kind]
    error = _deliver_result(job(), "report")
    assert error
    assert ROOT_TOKEN not in error + caplog.text
    assert url not in error + caplog.text
    assert len(network.calls) == 1


def test_lossless_long_text_and_document_after_common_redaction(home, network, profile_lanes):
    artifact = home / "cache" / 'report".pdf'
    artifact.parent.mkdir()
    payload = b"%PDF synthetic bytes\x00\xff"
    artifact.write_bytes(payload)
    synthetic_credential = "sk-" + "x" * 40
    text = ("astral \U0001f680 and spaces \n" * 900) + synthetic_credential
    from agent.redact import redact_sensitive_text
    from gateway.platforms.base import BasePlatformAdapter

    content = f"{text}\nMEDIA:{artifact}"
    _, cleaned = BasePlatformAdapter.extract_media(content)
    assert _deliver_result(job(), content) is None
    chunks = [data["text"] for method, data in network.calls if method == "sendMessage"]
    assert len(chunks) > 2
    assert "".join(chunks) == redact_sensitive_text(cleaned, force=True)
    assert synthetic_credential not in "".join(chunks)
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)
    assert network.calls[-1] == ("sendDocument", {"chat_id": ops.CHAT_ID.encode(), "document": payload})


@pytest.mark.parametrize("failure", ["policy", "upload", "proof"])
def test_media_failure_is_reported_without_adapter_fallback(home, network, profile_lanes, failure):
    artifact = home / "cache" / "report.pdf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"synthetic document")
    if failure == "policy":
        artifact = home / "auth.json"  # recognized MEDIA suffix; credential path is denied
        artifact.write_text("{}")
    else:
        def transform(method, result):
            if method == "sendDocument":
                if failure == "upload":
                    return {"ok": False}
                result["result"]["message_thread_id"] = 22
            return result
        network.transform = transform
    error = _deliver_result(job(), f"report\nMEDIA:{artifact}")
    assert error
    if failure == "policy":
        assert "media path policy" in error
        assert all(method != "sendDocument" for method, _ in network.calls)
    for spy in profile_lanes:
        spy.assert_not_called()


def test_partial_chunk_failure_stops_and_reports(network, profile_lanes):
    def transform(method, result):
        if len(network.calls) == 3:
            return {"ok": False, "description": "send failed"}
        return result
    network.transform = transform
    assert _deliver_result(job(), "X" * 13000)
    assert [method for method, _ in network.calls] == ["getMe", "sendMessage", "sendMessage"]


def test_inherited_topics_deduplicate_and_notify_setting_reaches_text_and_media(
    home, network, profile_lanes, monkeypatch,
):
    (home / "config.yaml").write_text("cron:\n  wrap_response: false\n  delivery:\n    notify: false\n")
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", ops.CHAT_ID)
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL_THREAD_ID", "different-inherited-topic")
    artifact = home / "cache" / "report.pdf"
    artifact.parent.mkdir()
    artifact.write_bytes(b"report")
    assert _deliver_result(job("origin,telegram"), f"report\nMEDIA:{artifact}") is None
    assert network.calls == [
        ("getMe", {}),
        ("sendMessage", {"chat_id": ops.CHAT_ID, "text": "report", "disable_notification": "true"}),
        ("sendDocument", {"chat_id": ops.CHAT_ID.encode(), "document": b"report",
                          "disable_notification": b"true"}),
    ]


def test_local_and_mixed_non_telegram_keep_existing_transport(home, network, monkeypatch):
    from gateway.config import Platform
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "cron": {"wrap_response": False},
            "platforms": {"discord": {"enabled": True, "token": "synthetic"}},
        }))
    send = AsyncMock(return_value={"success": True, "message_id": "non-telegram-proof"})
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    assert _deliver_result(job("local"), "local report") is None
    assert network.calls == [] and not send.called
    assert _deliver_result(job(f"telegram:{ops.CHAT_ID},discord:12345"), "mixed report") is None
    assert len(network.calls) == 2
    assert send.call_args.args[:4] == (Platform.DISCORD, send.call_args.args[1], "12345", "mixed report")
    network.calls.clear()
    assert "destination" in _deliver_result(job("telegram:999,discord:12345"), "mixed report")
    assert network.calls == [] and send.call_count == 2


@pytest.mark.parametrize("deliver", ["telegram,discord:12345", "telegram:,discord:12345"])
def test_missing_telegram_target_cannot_disappear_from_mixed_delivery(
    home, network, monkeypatch, deliver,
):
    (home / "config.yaml").write_text(yaml.safe_dump({
        "platforms": {"discord": {"enabled": True, "token": "synthetic"}},
    }))
    send = AsyncMock(return_value={"success": True, "message_id": "discord-proof"})
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
    assert "destination verification" in _deliver_result(job(deliver, origin=None), "report")
    assert network.calls == []
    send.assert_awaited_once()


@pytest.mark.parametrize("deliver", [f"origin,telegram:{ops.CHAT_ID}:111",
                                     f"telegram:{ops.CHAT_ID}:111,origin"])
def test_explicit_topic_refusal_survives_dedup_with_inherited_topic(network, profile_lanes, deliver):
    assert "destination verification" in _deliver_result(job(deliver), "report")
    # The separately requested origin is still valid; only its inherited topic
    # may be flattened. The explicit topic must leave a delivery failure.
    assert network.calls == [("getMe", {}), ("sendMessage", {"chat_id": ops.CHAT_ID, "text": "report"})]


@pytest.mark.parametrize("no_agent", [False, True])
@pytest.mark.parametrize("outcome", ["delivered", "failed", "silent"])
def test_real_scheduler_dispatch_for_ai_and_script(
    home, network, profile_lanes, monkeypatch, no_agent, outcome,
):
    from cron import executions, jobs, scheduler, scheduler_script

    response = "[SILENT]" if outcome == "silent" else "dispatch report"
    if outcome == "failed":
        (home / ".env").write_text("")
    (home / "config.yaml").write_text("cron:\n  wrap_response: false\n  preflight: false\n")
    # Work is synthetic; run_one_job, run_job, content composition, target
    # resolution, delivery, and durable outcome recording are all real.
    monkeypatch.setattr(scheduler, "_launch_external_cron_worker", lambda job: False)
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", lambda **kw: {
        "provider": "openrouter", "api_mode": "chat_completions",
        "base_url": "https://example.invalid/v1", "api_key": "synthetic-model-key",
        "source": "test", "requested_provider": None,
    })
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": response, "messages": []}
    agent_class = Mock(return_value=agent)
    monkeypatch.setattr("run_agent.AIAgent", agent_class)
    script = Mock(return_value=(True, response))
    monkeypatch.setattr(scheduler_script, "_run_job_script", script)
    with jobs.use_cron_store(home):
        j = jobs.create_job(
            prompt="Produce the synthetic report", schedule="every 1h",
            deliver=f"telegram:{ops.CHAT_ID}", no_agent=no_agent,
            script="synthetic.py" if no_agent else None,
        )
        assert scheduler.run_one_job(j)
        if no_agent:
            script.assert_called_once()
            agent_class.assert_not_called()
        else:
            agent_class.assert_called_once()
            agent.run_conversation.assert_called_once()
            script.assert_not_called()
        saved = jobs.get_job(j["id"])
        execution = executions.get_execution(j["execution_id"])
        if outcome == "failed":
            assert "root secret store" in saved["last_delivery_error"]
            assert execution["delivery_outcome"] == "failed"
            assert network.calls == []
        elif outcome == "silent":
            assert network.calls == []
            assert not saved.get("last_delivery_error")
        else:
            assert network.calls[-1] == ("sendMessage", {"chat_id": ops.CHAT_ID, "text": response})
            assert not saved.get("last_delivery_error")
            assert execution["delivery_outcome"] == "delivered"


@pytest.mark.parametrize("outcome", ["delivered", "failed", "suppressed"])
def test_external_worker_queue_drains_through_verified_boundary_once(
    home, network, profile_lanes, monkeypatch, outcome,
):
    from cron import delivery_queue, scheduler

    j = job(execution_id="synthetic-execution")
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", j["execution_id"])
    monkeypatch.setattr(delivery_queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
    assert _deliver_result(j, "queued report", for_failure=True) is None
    assert network.calls == []
    assert delivery_queue.get_status(j["execution_id"])["status"] == "pending"
    if outcome == "failed":
        (home / ".env").write_text("")
    elif outcome == "suppressed":
        (home / "config.yaml").write_text(
            "display:\n  suppress_warning_notifications: true\ncron:\n  wrap_response: false\n")
    assert scheduler.drain_delivery_queue({}, None) == 1
    assert delivery_queue.get_status(j["execution_id"])["status"] == outcome
    assert len(network.calls) == (2 if outcome == "delivered" else 0)
    assert scheduler.drain_delivery_queue({}, None) == 0
    # A replayed external worker sees the terminal receipt, never another send.
    error = _deliver_result(j, "queued report", for_failure=True)
    assert bool(error) == (outcome == "failed")
    assert len(network.calls) == (2 if outcome == "delivered" else 0)
