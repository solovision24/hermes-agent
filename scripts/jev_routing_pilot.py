#!/usr/bin/env python3
"""Opt-in, offline-by-default Jev routing benchmark. Never changes a Hermes route."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
PRICE_PER_MILLION_INPUT = 0.042  # https://docs.typesafe.ai/models (2026-09-24)
CONFIDENCE_FLOOR = 0.80


def policy(case):
    """Hard constraints precede any remote judgment."""
    candidates = case["candidates"]
    if not isinstance(candidates, dict) or not candidates or any(
        not isinstance(k, str) or not k or not isinstance(v, str) for k, v in candidates.items()
    ):
        raise ValueError("candidates must be a nonempty model-to-description map")
    allowed = {name: desc for name, desc in candidates.items()
               if name in case.get("available_models", list(candidates))}
    if case.get("approval_required") or case.get("safety_hold"):
        return allowed, None, "approval_or_safety_hold"
    selected = case.get("user_selected_model")
    if selected:
        return allowed, selected if selected in allowed else None, "user_selected" if selected in allowed else "selected_unavailable"
    fallback = case.get("fallback_model")
    if fallback not in allowed:
        return allowed, None, "fallback_unavailable"
    return allowed, fallback, "eligible"


def payload(case, allowed):
    """Only caller-supplied de-identified task summary crosses the API boundary."""
    summary = case.get("deidentified_summary")
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 4000:
        raise ValueError("deidentified_summary must be 1-4000 characters")
    teams = case.get("triage_candidates", {})
    if not isinstance(teams, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in teams.items()):
        raise ValueError("triage_candidates must be a name-to-description map")
    current = case.get("current_model")
    state = {"task": summary}
    if current in allowed:
        state["current_model"] = current
    questions = {
        "model_route": {"type": "choice", "instructions": "Select the least expensive capable model for this task. Consider complexity and required capabilities, not user identity.", "criteria": allowed},
    }
    if current in allowed:
        questions["escalate"] = {"type": "choice", "instructions": "Does this task require escalation from its current model because of complexity, failure or safety uncertainty?", "criteria": {"yes": "Escalation is warranted; recommendation only", "no": "Current model is adequate"}}
    if teams:
        questions["task_triage"] = {"type": "choice", "instructions": "Which team should review this task? This is advisory, not an assignment.", "criteria": teams}
    return {"model": MODEL, "state": state, "questions": questions}


def evaluate(body, api_key, timeout=3):
    request = Request(ENDPOINT, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + api_key, "Content-Type": "application/json",
    }, method="POST")
    start = time.monotonic()
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, (time.monotonic() - start) * 1000


def valid_choice(answers, question, options):
    answer = answers.get(question)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        return None
    probabilities = answer.get("probabilities")
    confidence = answer.get("confidence")
    if (answer.get("choice") not in options or not isinstance(probabilities, dict)
        or set(probabilities) != set(options) or not isinstance(confidence, (int, float))
        or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1
        or any(not isinstance(p, (float, int)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values())
        or abs(sum(probabilities.values()) - 1) > .03):
        return None
    return answer


def decide(case, live=False, api_key=None, evaluator=evaluate):
    allowed, fallback, reason = policy(case)
    row = {"id": case.get("id"), "route": fallback, "source": "policy", "reason": reason,
           "triage": None, "escalation_advice": None, "latency_ms": 0, "input_tokens": 0,
           "estimated_usd": 0, "confidence": None, "probabilities": None}
    if reason != "eligible" or not live:
        if reason == "eligible":
            row["reason"] = "offline_fallback"
        return row
    if not api_key or api_key.startswith("PASTE_"):
        row["reason"] = "missing_credential"
        return row
    try:
        body = payload(case, allowed)
    except ValueError:
        row["route"] = None
        row["reason"] = "invalid_input"
        return row
    try:
        result, elapsed = evaluator(body, api_key)
        row["latency_ms"] = round(elapsed, 2)
        usage = result.get("usage", {})
        tokens = usage.get("input_tokens", 0)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("invalid token count")
        row["input_tokens"] = tokens
        row["estimated_usd"] = round(tokens * PRICE_PER_MILLION_INPUT / 1_000_000, 9)
        answers = result["answers"]
        route = valid_choice(answers, "model_route", allowed)
        if route is None:
            row["reason"] = "invalid_response"
            return row
        row["confidence"] = route["confidence"]
        row["probabilities"] = route["probabilities"]
        triage = valid_choice(answers, "task_triage", case.get("triage_candidates", {})) if case.get("triage_candidates") else None
        if triage:
            row["triage"] = {"choice": triage["choice"], "confidence": triage["confidence"],
                             "probabilities": triage["probabilities"],
                             "accepted": triage["confidence"] >= CONFIDENCE_FLOOR}
        escalation = valid_choice(answers, "escalate", {"yes", "no"})
        if escalation:
            row["escalation_advice"] = {"choice": escalation["choice"], "confidence": escalation["confidence"],
                                         "probabilities": escalation["probabilities"],
                                         "accepted": escalation["confidence"] >= CONFIDENCE_FLOOR}
        if route["confidence"] < CONFIDENCE_FLOOR:
            row["reason"] = "low_confidence"
            return row
        row["route"] = route["choice"]
        row["source"] = "jev_recommendation"
        row["reason"] = "advisory_only"

    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        row["route"] = fallback
        row["source"] = "policy"
        row["reason"] = "api_error_fallback"
    return row


def benchmark(cases, live=False, api_key=None, evaluator=evaluate):
    rows = [decide(case, live, api_key, evaluator) for case in cases]
    labeled = [(case, row) for case, row in zip(cases, rows) if case.get("expected_model")]
    triage_labeled = [(case, row) for case, row in zip(cases, rows) if case.get("expected_triage")]
    return {"mode": "live_advisory" if live else "offline", "cases": len(rows),
            "model_accuracy": (sum(c["expected_model"] == r["route"] for c, r in labeled) / len(labeled)) if labeled else None,
            "triage_accuracy": (sum(r["triage"] is not None and r["triage"]["accepted"] and c["expected_triage"] == r["triage"]["choice"] for c, r in triage_labeled) / len(triage_labeled)) if triage_labeled else None,
            "model_labeled": len(labeled), "triage_labeled": len(triage_labeled),
            "total_estimated_usd": round(sum(r["estimated_usd"] for r in rows), 9),
            "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 2) if rows else None,
            "regressions": [r["id"] for c, r in labeled if c["expected_model"] != r["route"]],
            "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path, help="JSON list of de-identified cases; never pass raw requests")
    parser.add_argument("--live", action="store_true", help="Explicit opt-in to send fixture summaries to TypeSafe")
    args = parser.parse_args()
    cases = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if isinstance(cases, dict) and isinstance(cases.get("cases"), list):
        defaults = cases.get("defaults", {})
        if not isinstance(defaults, dict):
            parser.error("defaults must be an object")
        cases = [{**defaults, **item} for item in cases["cases"]]
    if not isinstance(cases, list):
        parser.error("fixtures must contain a JSON list")
    print(json.dumps(benchmark(cases, args.live, os.environ.get("TYPESAFE_API_KEY")), indent=2))


if __name__ == "__main__":
    main()
