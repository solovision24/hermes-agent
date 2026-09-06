"""Deterministic, per-turn adaptive model routing.

Routing is resolved once before an acting-model call.  The returned decision is
metadata as well as a target, so callers can audit why a model was selected.
This module deliberately does not participate in provider fallback chains,
provider routing, delegation, or MoA.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


CATEGORIES = ("simple_chat", "simple_coding", "standard_coding", "complex_coding", "architecture_coding", "research", "ops", "creative")
CODING_CATEGORIES = CATEGORIES[1:5]


@dataclass(frozen=True)
class AdaptiveRoute:
    category: str
    level: str
    model: str | None
    provider: str | None
    reason: str
    enabled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"category": self.category, "level": self.level, "model": self.model,
                "provider": self.provider, "reason": self.reason, "enabled": self.enabled}


def classify_message(message: Any) -> tuple[str, str, str]:
    """Classify one user turn without an LLM or network call."""
    text = message if isinstance(message, str) else str(message)
    lower = text.lower()
    if re.search(r"\b(migration|migrate|schema design|architecture|architect|redesign|breaking change|production rollout|security model)\b", lower):
        return "architecture_coding", "complex", "explicit architecture/high-risk language"
    if re.search(r"\b(research|compare sources|literature review|find papers|investigate)\b", lower):
        return "research", "standard", "research language"
    if re.search(r"\b(deploy|deployment|docker|kubernetes|terraform|pipeline|infrastructure|server|production ops|incident)\b", lower):
        return "ops", "complex" if len(text) > 240 else "standard", "operations language"
    if re.search(r"\b(image|illustration|song|music|creative|poem|logo|design concept)\b", lower):
        return "creative", "standard", "creative language"
    coding = re.search(r"\b(code|coding|bug|fix|implement|refactor|test|script|function|api|endpoint|class|typescript|python|javascript|sql|compile|error|traceback|debug)\b", lower)
    if coding:
        if re.search(r"\b(multi-file|multiple files|integration|debug|race condition|concurrency|regression|root cause)\b", lower) or len(text) > 600:
            return "complex_coding", "complex", "multi-file/integration/debugging signal"
        if re.search(r"\b(implement|feature|refactor|endpoint|api|bug|fix)\b", lower) or len(text) > 140:
            return "standard_coding", "standard", "bounded coding signal"
        return "simple_coding", "simple", "small coding signal"
    return "simple_chat", "simple", "default conversational route"


def resolve_route(config: Mapping[str, Any] | None, message: Any, *, current_model: str = "", current_provider: str = "") -> AdaptiveRoute:
    cfg = (config or {}).get("adaptive_model_routing") or {}
    if not isinstance(cfg, Mapping) or cfg.get("enabled", False) is not True:
        return AdaptiveRoute("simple_chat", "simple", None, None, "adaptive routing disabled", False)
    category, level, reason = classify_message(message)
    routes = cfg.get("routes") or {}
    target = routes.get(category) if isinstance(routes, Mapping) else None
    if isinstance(target, Mapping) and isinstance(target.get("levels"), Mapping):
        target = target["levels"].get(level) or target["levels"].get("standard")
    if not isinstance(target, Mapping):
        target = routes.get(level) if isinstance(routes, Mapping) else None
    if not isinstance(target, Mapping):
        return AdaptiveRoute(category, level, current_model or None, current_provider or None, reason)
    return AdaptiveRoute(category, level, target.get("model") or current_model or None,
                         target.get("provider") or current_provider or None, reason)
