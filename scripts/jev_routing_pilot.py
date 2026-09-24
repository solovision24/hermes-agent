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
CODEX_CANDIDATES = {
    "gpt-6-astra": "Codex GPT-6 Astra; assess task fit and quality",
    "gpt-6-sol": "Codex GPT-6 Sol; assess task fit and quality",
    "gpt-6-luna": "Codex GPT-6 Luna; assess task fit and quality",
}


def live_codex_models():
    """Read only the account-scoped live catalog; no cache or synthetic IDs."""
    import httpx
    from hermes_cli.auth import resolve_codex_runtime_credentials
    from hermes_cli.codex_models import _extract_chatgpt_account_id

    token = resolve_codex_runtime_credentials().get("api_key")
    account = _extract_chatgpt_account_id(token) if token else None
    if not account:
        return []
    response = httpx.get(
        "https://chatgpt.com/backend-api/codex/models?client_version=1.0.0",
        headers={"Authorization": f"Bearer {token}", "ChatGPT-Account-Id": account},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    entries = data.get("models") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    return [entry["slug"] for entry in entries if isinstance(entry, dict)
            and isinstance(entry.get("slug"), str)
            and str(entry.get("visibility", "")).lower() not in {"hide", "hidden"}]


def policy(case, catalog_models=None):
    """Hard constraints precede any remote judgment."""
    available = case.get("available_models", []) if catalog_models is None else catalog_models
    if not isinstance(available, (list, tuple, set)) or any(not isinstance(x, str) for x in available):
        raise ValueError("available_models must be a list of model IDs")
    allowed = {name: desc for name, desc in CODEX_CANDIDATES.items() if name in available}
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
        "model_route": {"type": "choice", "instructions": "Choose the best-fit Codex GPT-6 model for task complexity and required capabilities, not user identity. Prices and measured model quality are not supplied; do not infer a cost ranking.", "criteria": allowed},
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


def decide(case, live=False, api_key=None, evaluator=evaluate, catalog_models=None):
    if live and catalog_models is None:
        try:
            catalog_models = live_codex_models()
        except Exception:
            catalog_models = []
    allowed, fallback, reason = policy(case, catalog_models)
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


def benchmark(cases, live=False, api_key=None, evaluator=evaluate, catalog_fetcher=live_codex_models):
    catalog_models = None
    if live:
        try:
            catalog_models = catalog_fetcher()
        except Exception:
            catalog_models = []
    rows = [decide(case, live, api_key, evaluator, catalog_models) for case in cases]
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
