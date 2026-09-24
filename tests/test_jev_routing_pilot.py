"""Contract tests for the opt-in Codex-only recommendation pilot."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "jev_routing_pilot.py"
spec = importlib.util.spec_from_file_location("jev_routing_pilot", MODULE)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

MODELS = list(pilot.CODEX_CANDIDATES)


def case(**overrides):
    value = {"id": "labeled-1", "deidentified_summary": "Fix bounded CLI parsing defect",
             "candidates": {"intruder": "Untrusted fixture candidate"},
             "available_models": MODELS, "fallback_model": "gpt-6-sol",
             "current_model": "gpt-6-luna",
             "triage_candidates": {"chip": "Bounded fix", "forge": "Infrastructure"}}
    value.update(overrides)
    return value


def answer(route="gpt-6-luna", confidence=.95, options=MODELS):
    probs = {name: (1 if name == route else 0) for name in options}
    return {"answers": {
        "model_route": {"type": "choice", "choice": route, "confidence": confidence,
                        "probabilities": probs},
        "task_triage": {"type": "choice", "choice": "chip", "confidence": .91,
                        "probabilities": {"chip": .91, "forge": .09}},
        "escalate": {"type": "choice", "choice": "no", "confidence": .91,
                     "probabilities": {"yes": .09, "no": .91}}},
            "usage": {"input_tokens": 250}}


def forbidden(*_):
    raise AssertionError("network called")


def test_offline_never_calls_network():
    row = pilot.decide(case(), evaluator=forbidden)
    assert (row["route"], row["reason"]) == ("gpt-6-sol", "offline_fallback")


def test_holds_and_pin_precede_jev():
    with patch.object(pilot, "live_codex_models", return_value=MODELS):
        assert pilot.decide(case(user_selected_model="gpt-6-luna"), True, "key", forbidden)["route"] == "gpt-6-luna"
        for hold in ("safety_hold", "approval_required"):
            row = pilot.decide(case(**{hold: True}, user_selected_model="gpt-6-luna"), True, "key", forbidden)
            assert (row["route"], row["reason"]) == (None, "approval_or_safety_hold")
        assert pilot.decide(case(user_selected_model="missing"), True, "key", forbidden)["reason"] == "selected_unavailable"


def test_malformed_live_input_fails_closed():
    for item in (case(deidentified_summary="   "), case(triage_candidates=["chip"]),
                 case(deidentified_summary="x" * 4001)):
        row = pilot.decide(item, True, "key", forbidden, MODELS)
        assert (row["route"], row["reason"]) == (None, "invalid_input")


def test_exact_trio_and_live_account_availability():
    assert MODELS == ["gpt-6-astra", "gpt-6-sol", "gpt-6-luna"]
    allowed, _, _ = pilot.policy(case(available_models=MODELS + ["intruder"]))
    assert list(allowed) == MODELS
    allowed, _, reason = pilot.policy(case(available_models=MODELS), ["gpt-6-luna", "intruder"])
    assert list(allowed) == ["gpt-6-luna"] and reason == "fallback_unavailable"
    row = pilot.decide(case(fallback_model="gpt-6-luna"), True, "key", forbidden, [])
    assert (row["route"], row["reason"]) == (None, "fallback_unavailable")
    row = pilot.decide(case(fallback_model="gpt-6-luna", available_models=MODELS), True, "key",
                       lambda body, _: (answer(options=["gpt-6-luna"]), 100), ["gpt-6-luna"])
    assert row["route"] == "gpt-6-luna"


def test_payload_only_declared_summary_and_verified_candidates():
    c = case(private_data="must not leak")
    body = pilot.payload(c, pilot.policy(c, ["gpt-6-luna"])[0])
    assert body["state"] == {"task": c["deidentified_summary"], "current_model": "gpt-6-luna"}
    assert list(body["questions"]["model_route"]["criteria"]) == ["gpt-6-luna"]
    assert "private_data" not in str(body) and "intruder" not in str(body)


def test_recommendation_probabilities_triage_escalation_cost():
    row = pilot.decide(case(), True, "key", lambda *_: (answer(), 120.5), MODELS)
    assert row["route"] == "gpt-6-luna" and row["source"] == "jev_recommendation"
    assert row["probabilities"]["gpt-6-luna"] == 1
    assert row["triage"]["choice"] == "chip" and row["escalation_advice"]["choice"] == "no"
    assert row["estimated_usd"] == 250 * pilot.PRICE_PER_MILLION_INPUT / 1_000_000


def test_low_confidence_invalid_choice_and_outage_fallback():
    low = pilot.decide(case(), True, "key", lambda *_: (answer(confidence=.5), 80), MODELS)
    assert (low["route"], low["reason"]) == ("gpt-6-sol", "low_confidence")
    invalid = pilot.decide(case(), True, "key", lambda *_: (answer(route="intruder"), 80), MODELS)
    assert (invalid["route"], invalid["reason"]) == ("gpt-6-sol", "invalid_response")
    def outage(*_):
        raise TimeoutError("offline")
    assert pilot.decide(case(), True, "key", outage, MODELS)["reason"] == "api_error_fallback"


def test_benchmark_catalog_once_and_labeled_metrics():
    calls = []
    def catalog():
        calls.append(1)
        return MODELS
    report = pilot.benchmark([case(expected_model="gpt-6-luna", expected_triage="chip"),
                              case(id="labeled-2", expected_model="gpt-6-sol")],
                             True, "key", lambda *_: (answer(), 100), catalog)
    assert calls == [1]
    assert report["model_accuracy"] == .5 and report["triage_accuracy"] == 1
    assert report["regressions"] == ["labeled-2"]


def test_catalog_failure_never_uses_fixture_or_cached_models():
    report = pilot.benchmark([case()], True, "key", forbidden,
                             lambda: (_ for _ in ()).throw(OSError("catalog down")))
    assert report["results"][0]["route"] is None
    assert report["results"][0]["reason"] == "fallback_unavailable"


def test_live_catalog_reads_raw_account_response_not_synthetic(monkeypatch):
    from hermes_cli import auth, codex_models
    import httpx
    monkeypatch.setattr(auth, "resolve_codex_runtime_credentials", lambda: {"api_key": "opaque"})
    monkeypatch.setattr(codex_models, "_extract_chatgpt_account_id", lambda _: "account-id")
    def fetch(url, headers, timeout):
        assert headers == {"Authorization": "Bearer opaque", "ChatGPT-Account-Id": "account-id"}
        assert "codex/models" in url and timeout == 10
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                return {"models": [{"slug": "gpt-6-sol"}, {"slug": "gpt-6-astra", "visibility": "hidden"},
                                   {"slug": "gpt-5.6-sol"}]}
        return Response()
    monkeypatch.setattr(httpx, "get", fetch)
    assert pilot.live_codex_models() == ["gpt-6-sol", "gpt-5.6-sol"]
    assert list(pilot.policy(case(), pilot.live_codex_models())[0]) == ["gpt-6-sol"]
