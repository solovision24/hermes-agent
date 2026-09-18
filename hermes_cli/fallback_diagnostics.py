"""Config-time resolvability diagnostics for fallback chain rungs.

A ``fallback_providers`` (or legacy ``fallback_model``) rung that can never
serve any turn silently converts a recoverable provider problem into a
misleading, non-transient config error: the ladder lands on the rung, the
rung's provider dies with a setup-shaped error, and the operator sees a
message about a ``(task, provider)`` pair they never configured.

This module answers one question per rung: **can this rung resolve in the
current profile's own credential scope?**  It performs presence/entitlement
checks only — credential *values* are never read into memory beyond the
boolean the underlying resolver needs, and never logged.

Every diagnostic is warning-only: nothing here changes ladder order,
removes rungs, or blocks a config from loading.  The ladders themselves
(already skipping unresolvable entries at runtime) stay the single source
of behavioral truth.

Machine-readable reasons (``FallbackRungIssue.reason``):
    ``virtual_moa_without_preset`` — a virtual ``moa`` rung whose preset
        cannot be resolved in config, or whose resolved aggregator depends
        on a provider with no reachable credential (the built-in default
        preset pins the aggregator to openrouter, so a bare ``moa`` rung
        silently depends on OPENROUTER_API_KEY);
    ``no_credential_in_scope`` — the provider has no reachable API key,
        OAuth singleton, or credential-pool entry in this profile's scope;
    ``credential_in_quota_cooldown`` — the only reachable credentials are
        pool entries in a temporary exhaustion cooldown (self-healing —
        the rung will serve again after the reset time);
    ``relogin_required`` — the only reachable credentials are pool entries
        in a terminal auth state (revoked/invalidated token); a human must
        re-authenticate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes_cli.fallback_config import get_fallback_chain


# Reasons for an unreachable rung. Stable identifiers — operators and tests
# match on these, so do not reword them.
REASON_VIRTUAL_MOA_WITHOUT_PRESET = "virtual_moa_without_preset"
REASON_NO_CREDENTIAL_IN_SCOPE = "no_credential_in_scope"
REASON_CREDENTIAL_IN_QUOTA_COOLDOWN = "credential_in_quota_cooldown"
REASON_RELOGIN_REQUIRED = "relogin_required"

# The built-in default MoA preset (hermes_cli.moa_config) pins its aggregator
# here. A bare virtual-moa rung with no ``moa:`` config section therefore
# silently depends on this provider — surfaced via REASON_VIRTUAL_MOA_WITHOUT_PRESET.
_DEFAULT_MOA_AGGREGATOR_PROVIDER = "openrouter"


@dataclass
class FallbackRungIssue:
    """One unreachable fallback rung. Warning-only; never mutates config."""

    rung_index: int          # 0-based index into the effective chain
    provider: str            # rung provider as written in config
    model: str               # rung model as written in config
    reason: str              # one of the REASON_* constants
    detail: str              # human-readable explanation (safe to log)


def _has_inline_or_env_key(entry: Dict[str, Any]) -> bool:
    """True when the rung entry itself pins a usable key (inline or key_env)."""
    try:
        from hermes_cli.fallback_config import resolve_entry_api_key

        return bool(resolve_entry_api_key(entry))
    except Exception:
        return False


def _entry_carries_base_url(entry: Dict[str, Any]) -> bool:
    """True when the rung pins its own endpoint (base_url)."""
    return bool(str(entry.get("base_url") or "").strip())


def _provider_profile(provider: str):
    """Return the provider's ProviderDef (env var names, auth type), or None."""
    try:
        from hermes_cli.providers import get_provider

        return get_provider(provider, allow_network=False)
    except Exception:
        return None


def _oauth_provider_logged_in(provider: str) -> bool:
    """Best-effort presence check for OAuth-backed providers.

    Presence/entitlement only: reads stored auth state, never token values.
    Returns False on any lookup failure (the caller then reports the rung as
    unreachable — a conservative verdict that costs a warning, not behavior).
    """
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.auth import (
            get_minimax_oauth_auth_status,
            get_nous_auth_status,
            get_qwen_auth_status,
            get_xai_oauth_auth_status,
        )

        cfg = PROVIDER_REGISTRY.get(provider)
        if cfg is None:
            return False
        auth_type = (cfg.auth_type or "").strip().lower()
        if auth_type == "oauth_device_code" and provider == "nous":
            status = get_nous_auth_status() or {}
            return bool(status.get("logged_in"))
        if provider == "xai-oauth":
            status = get_xai_oauth_auth_status() or {}
            return bool(status.get("logged_in"))
        if provider == "minimax-oauth":
            status = get_minimax_oauth_auth_status() or {}
            return bool(status.get("logged_in"))
        if provider == "qwen-oauth":
            status = get_qwen_auth_status() or {}
            return bool(status.get("logged_in"))
        return False
    except Exception:
        return False


def _codex_has_usable_auth() -> bool:
    """Presence check for openai-codex: singleton tokens or a pool row.

    Codex is a single-use-refresh OAuth provider reading the canonical root
    store (``_codex_auth_file_path``) — never the profile shadow.
    """
    try:
        from hermes_cli.auth import (
            _codex_auth_file_path,
            _load_auth_store,
            _read_codex_tokens,
        )

        try:
            _read_codex_tokens(_lock=False)
            return True
        except Exception:
            pass
        auth_path = _codex_auth_file_path()
        store = _load_auth_store(auth_path)
        pool = store.get("credential_pool")
        if isinstance(pool, dict):
            entries = pool.get("openai-codex")
            if isinstance(entries, list) and entries:
                return True
        return False
    except Exception:
        return False


def _pool_entries_for(provider: str) -> List[Any]:
    """Load the provider's credential pool via the shared loader.

    Returns pooled credential records (dataclass ``PooledCredential``) with
    their persisted rotation status. Empty when no pool exists in scope.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        if pool is None or not pool.has_credentials():
            return []
        return pool.entries()
    except Exception:
        return []


def _pool_recovery_hint(entries: List[Any]) -> str:
    """Human-readable earliest recovery time for cooldown-bound pool entries.

    Renders `` (self-heals at 04:10, in ~6h)`` when a reset time is known,
    or "" when the pool carries no recoverable timestamp. Timestamps only —
    never credential material.
    """
    import time as _time

    try:
        from agent.credential_pool import _exhausted_until

        now = _time.time()
        untils = [
            u for u in (_exhausted_until(e) for e in entries)
            if u is not None and u > now
        ]
        if not untils:
            return ""
        soonest = min(untils)
        remaining = soonest - now
        if remaining >= 3600:
            scale = f"{remaining / 3600:.1f}h"
        elif remaining >= 60:
            scale = f"{remaining / 60:.0f}m"
        else:
            scale = f"{remaining:.0f}s"
        return (
            f" (self-heals at {_time.strftime('%H:%M', _time.localtime(soonest))}, "
            f"in ~{scale})"
        )
    except Exception:
        return ""


def _classify_pool_state(entries: List[Any]) -> str:
    """Map pool entries to the worst reachable state.

    Returns one of: ``ok`` (at least one entry outside cooldown), a
    REASON_* cooldown constant, or ``""`` (no entries at all — the caller
    treats that as no-credential-in-scope).
    """
    import time as _time

    try:
        from agent.credential_pool import STATUS_DEAD, STATUS_EXHAUSTED, _exhausted_until

        now = _time.time()
        saw_exhausted = False
        saw_dead = False
        for entry in entries:
            status = str(getattr(entry, "last_status", None) or "").strip().lower()
            if status in ("", "ok"):
                return "ok"
            if status == STATUS_EXHAUSTED:
                until = _exhausted_until(entry)
                # A cooldown that already expired means the entry can serve.
                if until is not None and until <= now:
                    return "ok"
                saw_exhausted = True
                continue
            if status == STATUS_DEAD:
                saw_dead = True
                continue
            # Unknown persisted status: treat as available (never invent
            # warnings the runtime would not hit).
            return "ok"
        if saw_dead:
            return REASON_RELOGIN_REQUIRED
        if saw_exhausted:
            return REASON_CREDENTIAL_IN_QUOTA_COOLDOWN
        return "ok"
    except Exception:
        return "ok"


def _check_moa_rung(
    entry: Dict[str, Any],
    issue_index: int,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[FallbackRungIssue]:
    """Diagnose a virtual ``moa`` rung.

    A moa rung is only reachable when the preset it names (or the default
    preset, when the model carries no preset name) resolves in config to a
    real aggregator provider+model, AND that aggregator provider has a
    reachable credential. ``load_config()`` deep-merges DEFAULT_CONFIG, so
    the default ``moa:`` section (whose preset pins the aggregator to
    openrouter) is indistinguishable from a user-written one — the raw user
    config is what tells us whether the dependency was ever declared. That
    invisible default-dependency is exactly the shape that hid this defect
    class, and it gets the dedicated ``virtual_moa_without_preset`` reason.
    """
    model = str(entry.get("model") or "").strip()
    try:
        from hermes_cli.config import load_config, read_user_config_raw
        from hermes_cli.moa_config import (
            DEFAULT_MOA_AGGREGATOR,
            DEFAULT_MOA_PRESET_NAME,
            resolve_moa_preset,
        )

        # Behavioral view: the moa: section the caller handed us (validate /
        # doctor pass the config they are checking); when the passed dict
        # does not declare one, fall back to the live profile config (which
        # is what `hermes fallback list` and the no-arg doctor path use).
        # This mirrors how get_fallback_chain(config) treats the chain keys.
        passed = config if isinstance(config, dict) else {}
        merged_cfg = passed.get("moa")
        if merged_cfg is None:
            try:
                from hermes_cli.config import load_config

                merged_cfg = load_config().get("moa")
            except Exception:
                merged_cfg = None
        # Raw view: did the USER actually declare MoA config (vs. riding the
        # built-in defaults merged in by load_config)? Operator surfaces pass
        # the *merged* config, so the passed section alone cannot answer this
        # — it always carries the built-in default preset. A passed section
        # counts as declared only when it holds something the defaults do not
        # (an extra preset, a top-level aggregator/reference_models, or a
        # non-default default_preset).
        try:
            raw_user_moa = (read_user_config_raw() or {}).get("moa")
        except Exception:
            raw_user_moa = None
        user_declared_moa = bool(
            isinstance(raw_user_moa, dict)
            and (raw_user_moa.get("presets") or raw_user_moa.get("aggregator")
                 or raw_user_moa.get("reference_models"))
        )
        if not user_declared_moa:
            passed_moa = passed.get("moa")
            if isinstance(passed_moa, dict):
                presets = passed_moa.get("presets")
                extra_presets = (
                    [k for k in presets if k != DEFAULT_MOA_PRESET_NAME]
                    if isinstance(presets, dict) else []
                )
                if (
                    extra_presets
                    or passed_moa.get("aggregator")
                    or passed_moa.get("reference_models")
                    or str(passed_moa.get("default_preset") or "").strip()
                    not in ("", DEFAULT_MOA_PRESET_NAME)
                ):
                    user_declared_moa = True

        preset_name = model or None
        preset = resolve_moa_preset(merged_cfg or {}, preset_name)
        agg = preset.get("aggregator") or {}
        agg_provider = str(agg.get("provider") or "").strip()
        agg_model = str(agg.get("model") or "").strip()
        if not agg_provider or not agg_model or agg_provider.lower() == "moa":
            return FallbackRungIssue(
                rung_index=issue_index,
                provider=str(entry.get("provider") or "moa"),
                model=model,
                reason=REASON_VIRTUAL_MOA_WITHOUT_PRESET,
                detail=(
                    "virtual moa rung: preset resolves to an empty aggregator "
                    "(no moa: preset configured that names a real provider+model)"
                ),
            )
        if not user_declared_moa:
            # Everything MoA-specific came from built-in defaults: the rung
            # silently depends on whichever provider the default preset pins.
            builtin_agg = DEFAULT_MOA_AGGREGATOR or {}
            dep = (
                f"{agg_provider} ({agg_model})"
                if not (
                    str(builtin_agg.get("provider") or "").strip().lower() == agg_provider.lower()
                    and str(builtin_agg.get("model") or "").strip() == agg_model
                )
                else f"{agg_provider} ({agg_model}) via the built-in default preset"
            )
            return FallbackRungIssue(
                rung_index=issue_index,
                provider=str(entry.get("provider") or "moa"),
                model=model,
                reason=REASON_VIRTUAL_MOA_WITHOUT_PRESET,
                detail=(
                    "virtual moa rung without any user-declared moa: preset — "
                    f"the rung silently depends on {dep}'s credentials"
                ),
            )
        # The user declared the preset — check the aggregator provider itself.
        agg_entry = {"provider": agg_provider, "model": agg_model}
        reason = _diagnose_standard_rung(agg_entry, issue_index)
        if reason is not None:
            return FallbackRungIssue(
                rung_index=issue_index,
                provider=str(entry.get("provider") or "moa"),
                model=model,
                reason=reason.reason,
                detail=(
                    f"virtual moa rung's aggregator {agg_provider} ({agg_model}) "
                    f"is unreachable: {reason.reason}"
                ),
            )
        return None
    except Exception as exc:
        # A preset that raises (renamed/deleted) is unreachable by definition.
        return FallbackRungIssue(
            rung_index=issue_index,
            provider=str(entry.get("provider") or "moa"),
            model=model,
            reason=REASON_VIRTUAL_MOA_WITHOUT_PRESET,
            detail=f"virtual moa rung: preset cannot be resolved ({exc})",
        )


def _env_credential_present(env_vars: List[str]) -> bool:
    """True when any of ``env_vars`` is set in this profile's scope.

    Mirrors the runtime's own api-key resolution order
    (``hermes_cli.auth._resolve_api_key_provider_secret``):
      1. ``agent.secret_scope.get_secret`` — authoritative inside a turn
         (multiplex-safe), falls back to ``os.environ`` when unscoped.
      2. ``get_env_value_prefer_dotenv`` — the runtime's dotenv-preferring
         read, so a key that exists only in ``<HERMES_HOME>/.env`` (the
         common case before any process export) still counts as present.
    """
    if not env_vars:
        return False
    try:
        from hermes_cli.auth import has_usable_secret
    except Exception:  # pragma: no cover - import always available in-repo
        has_usable_secret = lambda v: bool((v or "").strip())  # noqa: E731

    for name in env_vars:
        val = ""
        try:
            from agent.secret_scope import get_secret

            val = (get_secret(name) or "").strip()
        except Exception:
            val = ""
        if not has_usable_secret(val):
            try:
                from hermes_cli.config import get_env_value_prefer_dotenv

                val = (get_env_value_prefer_dotenv(name) or "").strip()
            except Exception:
                val = ""
        if has_usable_secret(val):
            return True
    return False


def _provider_env_vars(provider: str) -> List[str]:
    """Declared credential env vars: merged profile first, auth registry next."""
    env_vars: List[str] = []
    profile = _provider_profile(provider)
    env_vars.extend(getattr(profile, "api_key_env_vars", None) or [])
    if not env_vars:
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY

            rcfg = PROVIDER_REGISTRY.get(provider)
            if rcfg is not None and rcfg.api_key_env_vars:
                env_vars.extend(rcfg.api_key_env_vars)
        except Exception:
            pass
    # De-dup, preserve order.
    seen = set()
    return [v for v in env_vars if not (v in seen or seen.add(v))]


def _diagnose_standard_rung(
    entry: Dict[str, Any],
    issue_index: int,
) -> Optional[FallbackRungIssue]:
    """Diagnose a plain (non-moa) rung's credential reachability.

    Order mirrors the runtime's own credential precedence, so a verdict here
    matches what a turn would actually experience:
      1. Inline ``api_key`` / ``key_env`` on the rung entry → reachable.
      2. The provider's declared env vars (through the profile scope, then
         the dotenv-preferring read the runtime uses) → reachable.
      3. OAuth singlets (nous/xai/minimax/qwen/codex) → reachable.
      4. Credential pool: classify cooldown vs terminal auth state. Checked
         LAST because the runtime only reaches the pool after env keys, and
         an exhausted pool alongside a live env key still serves.
    """
    provider = str(entry.get("provider") or "").strip()
    model = str(entry.get("model") or "").strip()

    if _has_inline_or_env_key(entry):
        return None

    env_vars = _provider_env_vars(provider)
    if env_vars and _env_credential_present(env_vars):
        return None

    profile = _provider_profile(provider)
    auth_type = str(getattr(profile, "auth_type", "") or "").strip().lower()
    is_oauthish = auth_type in {"oauth_device_code", "oauth_external", "oauth_minimax"} or provider in {
        "openai-codex", "codex", "nous", "xai-oauth", "minimax-oauth", "qwen-oauth",
    }

    entries = _pool_entries_for(provider)
    if entries:
        pool_state = _classify_pool_state(entries)
        if pool_state == "ok":
            return None
        return FallbackRungIssue(
            rung_index=issue_index,
            provider=provider,
            model=model,
            reason=pool_state,
            detail=(
                f"credential pool for '{provider}' has {len(entries)} entr"
                f"{'y' if len(entries) == 1 else 'ies'} but none can serve: "
                f"{pool_state}{_pool_recovery_hint(entries)}"
            ),
        )

    if is_oauthish:
        if provider in {"openai-codex", "codex"}:
            if _codex_has_usable_auth():
                return None
        elif _oauth_provider_logged_in(provider):
            return None
    elif _entry_carries_base_url(entry) and not env_vars:
        # A pinned endpoint on a provider with no declared credential
        # surface: assume the endpoint authenticates itself (or is local).
        return None

    hint = f" Set {', '.join(env_vars)}." if env_vars else ""
    return FallbackRungIssue(
        rung_index=issue_index,
        provider=provider,
        model=model,
        reason=REASON_NO_CREDENTIAL_IN_SCOPE,
        detail=f"no reachable credential for provider '{provider}' in this "
               f"profile's scope.{hint}",
    )


def diagnose_fallback_chain(
    config: Optional[Dict[str, Any]] = None,
) -> List[FallbackRungIssue]:
    """Return one warning per unreachable rung in the effective chain.

    Warning-only: the returned issues never mutate config and never change
    ladder order. A healthy chain returns ``[]`` — callers must treat that
    as "no output", keeping startup clean.

    ``config`` may be a pre-loaded config dict (tests, batch validation);
    when omitted the profile's own config is loaded.
    """
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            return []

    chain = get_fallback_chain(config)
    issues: List[FallbackRungIssue] = []
    for index, entry in enumerate(chain):
        provider = str(entry.get("provider") or "").strip()
        try:
            from hermes_cli.providers import normalize_provider

            canonical = normalize_provider(provider)
        except Exception:
            canonical = provider.strip().lower()
        if canonical == "moa":
            issue = _check_moa_rung(entry, index, config)
        else:
            issue = _diagnose_standard_rung(entry, index)
        if issue is not None:
            issues.append(issue)
    return issues


def format_rung_issue(issue: FallbackRungIssue) -> str:
    """One-line human rendering used by fallback list / doctor / validation."""
    label = f"fallback[{issue.rung_index}] {issue.provider}"
    if issue.model:
        label += f" ({issue.model})"
    return f"{label}: unreachable — {issue.reason}: {issue.detail}"
