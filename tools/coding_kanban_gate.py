"""Safe, reusable intake for implementation work originating in chat.

The intake is deliberately independent of any one UI.  Tool dispatchers, the
gateway, ACP, the classic CLI, and the TUI can all pass the same request shape
and receive the same structured result.  Only the dispatcher-owned Kanban
worker may perform the eventual workspace mutation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional


CODING_TOOLS = frozenset({
    "write_file", "patch", "terminal", "execute_code", "delegate_task",
    "project_create", "project_switch", "codex_app_server",
})
SUPPORTED_CODING_AGENTS = frozenset({"codex", "cursor"})
ACTIVE_TASK_STATUSES = frozenset({"todo", "ready", "running", "review"})
CODING_LANE = "DEV"
_ENGINEERING_ASSIGNEES = ("dev", "forge", "quill", "chip")

_READ_ONLY_COMMANDS = frozenset({
    "awk", "cat", "cut", "date", "diff", "dirname", "du", "file", "find",
    "gh", "git", "grep", "head", "jq", "ls", "ps", "pwd", "readlink", "rg",
    "sed", "sort", "stat", "tail", "test", "tree", "uniq", "wc", "which", "whoami",
})
_GIT_READ_ONLY = frozenset({"branch", "diff", "log", "ls-files", "remote", "rev-parse", "show", "status", "tag"})
_CODING_AGENT_RE = re.compile(r"\b(codex|cursor|claude)(?:\s+(?:code|cli|agent))?\b", re.I)
_CODING_INTENT_RE = re.compile(
    r"\b(?:implement|modify|edit|write|add|remove|delete|fix|refactor|patch|"
    r"change|create|build|develop|code|commit|branch|worktree|checkout|launch)\b", re.I,
)
_IMPLEMENTATION_INTENT_RE = re.compile(
    r"\b(?:implement|modify|edit|fix|refactor|patch|change|build|develop|"
    r"code|commit|branch|worktree|checkout|launch)\b", re.I,
)
_RESEARCH_RE = re.compile(r"\b(?:explain|research|inspect|review|analy[sz]e|look\s+up)\b", re.I)
_WRITE_MARKERS = re.compile(
    r"(?:>>?|<<|\btee\b|(?:^|\s)(?:-delete|-exec)\b|\bsed\s+[^\n]*-i\b|"
    r"\bperl\s+[^\n]*-i\b|\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|"
    r"\bchmod\b|\bchown\b|\binstall\b|\btruncate\b|"
    r"\bsed\s+[^\n]*['\"]w(?:\s|['\"])|\bgit\s+[^\n]*--output(?:=|\s))", re.I,
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalise_path(value: Any, fallback: str) -> str:
    raw = _text(value) or fallback
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError:
        return raw


def _workspace(args: dict) -> str:
    return _normalise_path(
        args.get("workspace") or args.get("workdir") or os.environ.get("TERMINAL_CWD"),
        os.getcwd(),
    )


def _repository(workspace: str, args: dict) -> str:
    explicit = _text(args.get("repository") or args.get("repo"))
    if explicit:
        # owner/repo identifiers are not filesystem paths.
        if "/" in explicit and not Path(explicit).exists():
            return explicit
        return _normalise_path(explicit, workspace)
    try:
        result = subprocess.run(
            ["git", "-C", workspace, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _normalise_path(result.stdout.strip(), workspace)
    except (OSError, subprocess.SubprocessError):
        pass
    return workspace


def _acceptance(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    return [_text(item) for item in values if _text(item)]


def _json_value(value: Any) -> Any:
    """Keep provider metadata deterministic and JSON-safe."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _profile_name(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        from hermes_cli.profiles import normalize_profile_name
        return normalize_profile_name(text)
    except Exception:
        return text.casefold()


def _resolve_assignee(args: dict) -> str:
    explicit = args.get("assignee") or args.get("specialist") or args.get("coding_specialist")
    if explicit:
        return _profile_name(explicit)
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        kanban = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        configured = kanban.get("coding_assignee") or kanban.get("default_assignee")
        configured_name = _profile_name(configured) if configured else ""
        if configured_name in _ENGINEERING_ASSIGNEES:
            return configured_name
    except Exception:
        pass
    # Never route implementation work to the active chat profile: that may be
    # Orion/default or another non-engineering profile. Pick the first
    # qualified engineering profile that exists in the live roster.
    try:
        from hermes_cli.profiles import profile_exists
        for candidate in _ENGINEERING_ASSIGNEES:
            if profile_exists(candidate):
                return candidate
    except Exception:
        pass
    return _ENGINEERING_ASSIGNEES[0]


def _requested_agent(args: dict, user_message: Any = None) -> tuple[str, Optional[str]]:
    explicit = _text(args.get("coding_agent") or args.get("agent"))
    # Cursor is opt-in. Never infer it from prose, including a negative or
    # incidental mention such as "don't use Cursor".
    requested = (explicit or "codex").casefold()
    if requested == "claude":
        return requested, "Claude coding requires a qualifying subscription."
    if requested not in SUPPORTED_CODING_AGENTS:
        return requested, f"Unsupported coding agent {requested!r}; use Codex or Cursor."
    return requested, None


def _first_line(value: str) -> str:
    return (next((line.strip() for line in value.splitlines() if line.strip()), "Implement requested change")[:240])


def _persist_session_linkage(session_id: str, task_id: str, origin_session_id: str, origin_message_id: str) -> tuple[bool, Optional[str]]:
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            db.set_session_kanban_linkage(
                session_id, task_id,
                origin_session_id=origin_session_id,
                origin_message_id=origin_message_id,
            )
        finally:
            db.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def intake_coding_task(
    request: Optional[dict] = None,
    *,
    repository: Optional[str] = None,
    workspace: Optional[str] = None,
    scope: Optional[str] = None,
    acceptance_criteria: Any = None,
    coding_agent: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    provider_metadata: Optional[dict] = None,
    origin_session_id: Optional[str] = None,
    origin_message_id: Optional[str] = None,
    assignee: Optional[str] = None,
    specialist: Optional[str] = None,
    board: Optional[str] = None,
) -> dict[str, Any]:
    """Create or reuse one canonical implementation task.

    The idempotency identity intentionally includes every request dimension
    that can change what a worker should do, including acceptance criteria,
    provider metadata, and origin linkage.
    """
    request = request if isinstance(request, dict) else {}
    repository = repository if repository is not None else request.get("repository") or request.get("repo")
    workspace = workspace if workspace is not None else request.get("workspace") or request.get("workdir")
    scope = scope if scope is not None else request.get("scope")
    acceptance_criteria = acceptance_criteria if acceptance_criteria is not None else request.get("acceptance_criteria")
    coding_agent = coding_agent if coding_agent is not None else request.get("coding_agent")
    provider = provider if provider is not None else request.get("provider")
    model = model if model is not None else request.get("model")
    provider_metadata = provider_metadata if provider_metadata is not None else request.get("provider_metadata")
    origin_session_id = origin_session_id if origin_session_id is not None else request.get("origin_session_id") or request.get("session_id")
    origin_message_id = origin_message_id if origin_message_id is not None else request.get("origin_message_id") or request.get("message_id")
    assignee = assignee if assignee is not None else request.get("assignee")
    specialist = specialist if specialist is not None else request.get("specialist")
    board = board if board is not None else request.get("board")
    args = {
        "repository": repository, "workspace": workspace, "scope": scope,
        "acceptance_criteria": acceptance_criteria, "coding_agent": coding_agent,
        "provider": provider, "model": model, "provider_metadata": provider_metadata,
        "origin_session_id": origin_session_id, "origin_message_id": origin_message_id,
        "assignee": assignee, "specialist": specialist,
    }
    workspace_value = _normalise_path(workspace, os.getcwd())
    repository_value = _repository(workspace_value, args)
    scope_value = _text(scope) or "Implement the requested code change."
    acceptance_value = _acceptance(acceptance_criteria)
    if not acceptance_value:
        acceptance_value = [
            "Implement the requested change in the recorded repository and workspace.",
            "Verify the change before completing the coding task.",
        ]
    session_value = _text(origin_session_id)
    message_value = _text(origin_message_id)
    if not session_value:
        return {"ok": False, "status": "invalid_request", "error_type": "origin_required", "error": "origin_session_id is required"}
    if not message_value:
        message_value = hashlib.sha256(scope_value.encode("utf-8")).hexdigest()[:16]
    agent_value, agent_error = _requested_agent({"coding_agent": coding_agent}, scope_value)
    if agent_error:
        return {"ok": False, "status": "subscription_required" if agent_value == "claude" else "unsupported", "error_type": "subscription_required" if agent_value == "claude" else "unsupported_coding_agent", "coding_agent": agent_value, "error": agent_error}
    try:
        requested_assignee = _profile_name(assignee or specialist)
        if requested_assignee and requested_assignee not in _ENGINEERING_ASSIGNEES:
            return {"ok": False, "status": "invalid_assignee", "error_type": "invalid_assignee", "error": f"Coding assignee must be an engineering profile: {requested_assignee!r}"}
        assignee_value = requested_assignee or _resolve_assignee(args)
        from hermes_cli.profiles import profile_exists
        if not profile_exists(assignee_value):
            return {"ok": False, "status": "invalid_assignee", "error_type": "invalid_assignee", "error": f"Unknown coding assignee profile {assignee_value!r}"}
    except (ValueError, OSError) as exc:
        return {"ok": False, "status": "invalid_assignee", "error_type": "invalid_assignee", "error": str(exc)}
    provider_value = _text(provider) or None
    model_value = _text(model) or None
    provider_meta_value = _json_value(provider_metadata or {})
    origin = {"session_id": session_value, "message_id": message_value}
    identity = {
        "repository": repository_value, "workspace": workspace_value,
        "scope": scope_value, "acceptance_criteria": acceptance_value,
        "coding_agent": agent_value, "provider": provider_value,
        "model": model_value, "provider_metadata": provider_meta_value,
        "origin": origin, "assignee": assignee_value,
    }
    identity_key = "chat-coding:" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "canonical": True, "lane": CODING_LANE, "coding_agent": agent_value,
        "repository": repository_value, "workspace": workspace_value,
        "scope": scope_value, "acceptance_criteria": acceptance_value,
        "provider": provider_value, "model": model_value,
        "provider_metadata": provider_meta_value, "origin": origin,
        "request_identity": identity,
        "specialist": assignee_value,
    }
    try:
        from hermes_cli import kanban_db
        conn = kanban_db.connect(board=board)
        try:
            reused = conn.execute(
                "SELECT 1 FROM tasks WHERE idempotency_key = ? AND status != 'archived' LIMIT 1",
                (identity_key,),
            ).fetchone() is not None
            task_id = None
            if not reused:
                # Recover a pre-metadata canonical card created by an older
                # Hermes process when all identity fields that version knew
                # about agree. New cards always use the complete key above.
                for candidate in kanban_db.list_tasks(conn, session_id=session_value):
                    candidate_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
                    if (
                        candidate_meta.get("canonical") is True
                        and candidate_meta.get("lane") == CODING_LANE
                        and candidate_meta.get("coding_agent") == agent_value
                        and candidate_meta.get("repository") == repository_value
                        and candidate_meta.get("workspace") == workspace_value
                        and candidate_meta.get("scope") == scope_value
                        and candidate_meta.get("acceptance_criteria") == acceptance_value
                        and candidate_meta.get("provider") == provider_value
                        and candidate_meta.get("model") == model_value
                        and candidate_meta.get("provider_metadata") == provider_meta_value
                        and candidate_meta.get("origin") == origin
                        and candidate_meta.get("specialist") == assignee_value
                        and "request_identity" not in candidate_meta
                    ):
                        task_id = candidate.id
                        reused = True
                        break
            if task_id is None:
                task_id = kanban_db.create_task(
                    conn, title=_first_line(scope_value),
                    body="\n".join([scope_value, "", "Acceptance criteria:", *[f"- {item}" for item in acceptance_value]]),
                    assignee=assignee_value, created_by="chat", workspace_kind="dir",
                    workspace_path=workspace_value, idempotency_key=identity_key,
                    session_id=session_value, metadata=metadata,
                    model_override=model_value,
                    provider_override=provider_value if model_value else None,
                    initial_status="running",
                )
            task = kanban_db.get_task(conn, task_id)
        finally:
            conn.close()
    except Exception as exc:
        return {"ok": False, "status": "kanban_unavailable", "error_type": "kanban_unavailable", "error": f"Kanban intake failed: {exc}"}
    if task is None or not task.assignee or task.metadata is None:
        return {"ok": False, "status": "invalid_task", "error_type": "kanban_task_invalid", "error": "Kanban did not return a qualified coding task."}
    persisted, persistence_error = _persist_session_linkage(session_value, task.id, session_value, message_value)
    if not persisted:
        return {
            "ok": False, "status": "session_linkage_failed",
            "error_type": "session_linkage_failed", "task_id": task.id,
            "error": f"Kanban task created but session linkage could not be persisted: {persistence_error}",
        }
    return {
        "ok": True, "task_id": task.id, "status": task.status,
        "assignee": task.assignee, "coding_agent": agent_value,
        "provider": provider_value, "model": model_value,
        "reused": reused,
        "session_persisted": persisted,
        "session_persistence_error": persistence_error,
        "origin": origin,
    }


create_or_reuse_coding_task = intake_coding_task


def _command_parts(command: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s*(?:&&|\|\||[|;]|\n)\s*", command) if part.strip()]


def _gh_is_read_only(words: list[str]) -> bool:
    """Allow only known read-only gh operations; fail closed otherwise."""
    if len(words) < 2:
        return False
    args = words[1:]
    # gh accepts repository/host/cwd options before the command.
    while args and args[0].startswith("-"):
        option = args.pop(0)
        if option in {"--repo", "--hostname", "--cwd"}:
            if not args:
                return False
            args.pop(0)
    if args[0] == "api":
        # gh api defaults to GET. Any explicit method, form field, or input
        # changes the request and must remain behind the coding gate.
        mutation_flags = {"-X", "--method", "-f", "--raw-field", "-F", "--field", "--input"}
        return not any(
            item in mutation_flags
            or any(
                item.startswith(flag + "=")
                or (flag in {"-X", "-f", "-F"} and item.startswith(flag) and len(item) > len(flag))
                for flag in mutation_flags
            )
            for item in args[1:]
        )
    roots = {
        "auth": {"status"},
        "gist": {"list", "view"},
        "issue": {"list", "status", "view"},
        "label": {"list", "view"},
        "pr": {"list", "status", "view"},
        "project": {"list", "view"},
        "release": {"list", "view"},
        "repo": {"list", "status", "view"},
        "run": {"list", "view"},
        "variable": {"list"},
        "secret": {"list"},
        "workflow": {"list", "view"},
    }
    root = args[0]
    subcommand = next((item for item in args[1:] if not item.startswith("-")), None)
    return subcommand in roots.get(root, set())


def _simple_command_is_read_only(command: str) -> bool:
    if not command or _WRITE_MARKERS.search(command) or re.search(r"[`$()]", command):
        return False
    if re.search(r"\b(?:codex|cursor|claude|aider|copilot)\b", command, re.I):
        return False
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    if not words:
        return True
    while words and "=" in words[0] and not words[0].startswith("="):
        words.pop(0)
    base = Path(words[0]).name.lower() if words else ""
    if base not in _READ_ONLY_COMMANDS:
        return False
    if base == "git":
        subcommands = [word for word in words[1:] if not word.startswith("-")]
        if not subcommands or subcommands[0] not in _GIT_READ_ONLY:
            return False
        if subcommands[0] == "remote" and len(subcommands) > 1 and subcommands[1] not in {"show", "get-url"}:
            return False
        if subcommands[0] in {"branch", "tag"} and any(not word.startswith("-") for word in words[2:]):
            return False

    if base == "gh":
        return _gh_is_read_only(words)
    return True


def is_coding_intent(tool_name: str, args: Optional[dict] = None, user_message: Any = None) -> bool:
    args = args if isinstance(args, dict) else {}
    if tool_name == "write_file":
        message = _text(user_message)
        if re.search(r"\b(?:report|audit|research|findings|notes?)\b", message, re.I) and not _CODING_INTENT_RE.search(message):
            return False
        return True
    if tool_name in {"patch", "codex_app_server"}:
        if tool_name == "codex_app_server":
            message = _text(user_message)
            return bool(_IMPLEMENTATION_INTENT_RE.search(message) and not (_RESEARCH_RE.search(message) and not _IMPLEMENTATION_INTENT_RE.search(message)))
        return True
    if tool_name == "terminal":
        command = _text(args.get("command"))
        return not command or not all(_simple_command_is_read_only(part) for part in _command_parts(command))
    if tool_name == "execute_code":
        return bool(re.search(
            r"(?:open\s*\([^)]*['\"](?:w|a|x)|\.write(?:_text|_bytes)?\s*\(|"
            r"\.(?:touch|to_csv|to_excel|to_json|to_pickle|to_sql)\s*\(|"
            r"\b(?:unlink|remove|rename|replace|mkdir|rmdir|subprocess|os\.system|shutil\.)\b)",
            _text(args.get("code")), re.I,
        ))
    if tool_name == "delegate_task":
        text = "\n".join(filter(None, [_text(args.get("goal")), _text(args.get("context"))]))
        return bool(_CODING_INTENT_RE.search(text) and not (_RESEARCH_RE.search(text) and not re.search(r"\b(?:implement|edit|write|fix|change|patch)\b", text, re.I)))
    if tool_name in {"project_create", "project_switch"}:
        return bool(re.search(r"\b(?:branch|worktree|repository|repo|codebase|implementation)\b", _text(user_message), re.I))
    return False


def coding_tool_gate_refusal(tool_name: str, *, function_args: Optional[dict] = None, session_id: Optional[str] = None, task_id: Optional[str] = None, turn_id: Optional[str] = None, user_message: Any = None) -> Optional[str]:
    """Return a JSON handoff/refusal, or ``None`` when execution is safe."""
    args = function_args if isinstance(function_args, dict) else {}
    worker_id = _text(os.environ.get("HERMES_KANBAN_TASK"))
    if worker_id:
        if not is_coding_intent(tool_name, args, user_message):
            return None
        try:
            from hermes_cli import kanban_db
            with kanban_db.connect_closing() as conn:
                task = kanban_db.get_task(conn, worker_id)
            metadata = task.metadata if task and isinstance(task.metadata, dict) else {}
            run_id = _text(os.environ.get("HERMES_KANBAN_RUN_ID"))
            active_run = task and run_id and task.current_run_id is not None and str(task.current_run_id) == run_id
            canonical_metadata_invalid = bool(metadata) and (
                metadata.get("canonical") is not True
                or metadata.get("lane") != CODING_LANE
                or metadata.get("coding_agent") not in SUPPORTED_CODING_AGENTS
            )
            if not task or not task.assignee or task.status not in ACTIVE_TASK_STATUSES or canonical_metadata_invalid or not active_run:
                reason = "task is not an active qualified coding worker"
                return json.dumps({"ok": False, "error_type": "kanban_task_required", "tool": tool_name, "task_id": worker_id, "error": reason})
            return None
        except Exception as exc:
            return json.dumps({"ok": False, "error_type": "kanban_unavailable", "tool": tool_name, "task_id": worker_id, "error": str(exc)})
    # A bare low-level dispatch (for example, an ACP-approved write_file
    # call) has no intake identity to derive. Leave those established callers
    # alone; conversational/tool routes provide user_message or explicit
    # intake fields and are gated below.
    intake_fields = {
        "repository", "repo", "scope", "acceptance_criteria", "coding_agent",
        "provider", "model", "provider_metadata", "origin_session_id",
        "origin_message_id", "specialist",
    }
    has_intake_context = bool(_text(user_message)) or bool(intake_fields.intersection(args))
    if (
        not _text(session_id)
        or tool_name not in CODING_TOOLS
        or not is_coding_intent(tool_name, args, user_message)
        or not has_intake_context
    ):
        return None
    agent, agent_error = _requested_agent(args, user_message)
    if agent_error:
        result = {"ok": False, "error_type": "subscription_required" if agent == "claude" else "unsupported_coding_agent", "status": "subscription_required" if agent == "claude" else "unsupported", "tool": tool_name, "coding_agent": agent, "error": agent_error}
    else:
        result = intake_coding_task(
            repository=args.get("repository") or args.get("repo"), workspace=args.get("workspace") or args.get("workdir"),
            scope=args.get("scope") or args.get("goal") or args.get("context") or user_message,
            acceptance_criteria=args.get("acceptance_criteria"), coding_agent=agent,
            provider=args.get("provider"), model=args.get("model"), provider_metadata=args.get("provider_metadata"),
            origin_session_id=session_id, origin_message_id=args.get("origin_message_id") or turn_id or task_id,
            assignee=args.get("assignee") or args.get("specialist"),
        )
        if result.get("ok"):
            result["error_type"] = "kanban_task_required"
            result["error"] = "Coding work was handed off to the Kanban worker; this chat remains read-only."
            result["tool"] = tool_name
    return json.dumps(result, ensure_ascii=False)
