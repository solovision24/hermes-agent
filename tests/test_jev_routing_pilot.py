"""Contract tests for the opt-in recommendation-only pilot."""
import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "jev_routing_pilot.py"
spec = importlib.util.spec_from_file_location("jev_routing_pilot", MODULE)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def case(**overrides):
    value = {"id": "historical-1", "deidentified_summary": "Fix bounded CLI parsing defect",
             "candidates": {"small": "Low cost, simple fixes", "large": "Complex integration work"},
             "available_models": ["small", "large"], "fallback_model": "large",
             "current_model": "small",
             "triage_candidates": {"chip": "Bounded fix", "forge": "Infrastructure"}}
    value.update(overrides)
    return value


def answer(route="small", confidence=.95):
    return {"answers": {
        "model_route": {"type": "choice", "choice": route, "confidence": confidence,
                        "probabilities": {"small": .9, "large": .1}},
        "task_triage": {"type": "choice", "choice": "chip", "confidence": .91,
                        "probabilities": {"chip": .91, "forge": .09}},
        "escalate": {"type": "choice", "choice": "no", "confidence": .91,
                     "probabilities": {"yes": .09, "no": .91}}},
            "usage": {"input_tokens": 250}}


def test_offline_never_calls_network():
    def forbidden(*_):
        raise AssertionError("network called")
    row = pilot.decide(case(), evaluator=forbidden)
    assert (row["route"], row["reason"]) == ("large", "offline_fallback")


def test_user_pin_and_safety_hold_override_even_in_live_mode():
    def forbidden(*_):
        raise AssertionError("network called")
    assert pilot.decide(case(user_selected_model="small"), True, "key", forbidden)["route"] == "small"
    assert pilot.decide(case(approval_required=True), True, "key", forbidden)["route"] is None
    assert pilot.decide(case(user_selected_model="missing"), True, "key", forbidden)["reason"] == "selected_unavailable"


def test_hold_overrides_valid_user_pin_in_live_mode():
    def forbidden(*_):
        raise AssertionError("network called")
    safety = pilot.decide(case(user_selected_model="small", safety_hold=True), True, "key", forbidden)
    assert (safety["route"], safety["reason"]) == (None, "approval_or_safety_hold")
    approval = pilot.decide(case(user_selected_model="small", approval_required=True), True, "key", forbidden)
    assert (approval["route"], approval["reason"]) == (None, "approval_or_safety_hold")


def test_malformed_live_input_fails_closed_without_network_call():
    def forbidden(*_):
        raise AssertionError("network called")
    empty = pilot.decide(case(deidentified_summary="   "), True, "key", forbidden)
    assert (empty["route"], empty["reason"]) == (None, "invalid_input")
    overlong = pilot.decide(case(deidentified_summary="x" * 4001), True, "key", forbidden)
    assert (overlong["route"], overlong["reason"]) == (None, "invalid_input")
    bad_triage = pilot.decide(case(triage_candidates=["chip", "forge"]), True, "key", forbidden)
    assert (bad_triage["route"], bad_triage["reason"]) == (None, "invalid_input")


def test_payload_contains_only_declared_summary_and_available_candidates():
    c = case(available_models=["small"], private_data="must not leak")
    body = pilot.payload(c, pilot.policy(c)[0])
    assert body["state"] == {"task": c["deidentified_summary"], "current_model": "small"}
    assert list(body["questions"]["model_route"]["criteria"]) == ["small"]
    assert "private_data" not in str(body)


def test_recommendation_probabilities_triage_escalation_and_cost():
    row = pilot.decide(case(), True, "key", lambda *_: (answer(), 120.5))
    assert row["route"] == "small" and row["source"] == "jev_recommendation"
    assert row["probabilities"]["small"] == .9
    assert row["triage"]["choice"] == "chip"
    assert row["escalation_advice"]["choice"] == "no"
    assert row["estimated_usd"] == 250 * pilot.PRICE_PER_MILLION_INPUT / 1_000_000


def test_fail_closed_low_confidence_malformed_and_outage():
    low = pilot.decide(case(), True, "key", lambda *_: (answer(confidence=.5), 80))
    assert (low["route"], low["reason"]) == ("large", "low_confidence")
    assert low["triage"]["accepted"] and low["escalation_advice"]["accepted"]
    malformed = answer(route="undeclared")
    row = pilot.decide(case(), True, "key", lambda *_: (malformed, 80))
    assert (row["route"], row["reason"]) == ("large", "invalid_response")
    def outage(*_):
        raise TimeoutError("offline")
    assert pilot.decide(case(), True, "key", outage)["reason"] == "api_error_fallback"


def test_benchmark_reports_accuracy_and_regressions():
    report = pilot.benchmark([case(expected_model="small", expected_triage="chip"),
                              case(id="historical-2", expected_model="large")],
                             True, "key", lambda *_: (answer(), 100))
    assert report["model_accuracy"] == .5
    assert report["triage_accuracy"] == 1
    assert report["regressions"] == ["historical-2"]
