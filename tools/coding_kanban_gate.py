"""Intent-aware handoff gate for coding work requested from chat.

The gate deliberately lives below the model-facing dispatchers.  A model can
inspect and reason in chat, but a code-changing operation is handed to the
canonical DEV lane before its handler is allowed to run.
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
    "write_file",
    "patch",
    "terminal",
    "execute_code",
    "delegate_task",
    "project_create",
    "project_switch",
    "codex_app_server",
})
CANONICAL_ASSIGNEE = "dev"
IMPLEMENTATION_ASSIGNEES = frozenset({"dev", "forge", "quill", "chip"})
SUPPORTED_CODING_AGENTS = frozenset({"codex", "cursor"})
ACTIVE_TASK_STATUSES = frozenset({"todo", "ready", "running", "review"})

_READ_ONLY_COMMANDS = frozenset({
    "awk", "cat", "cut", "diff", "dirname", "du", "file", "find",
    "git", "grep", "head", "jq", "ls", "pwd", "readlink", "rg", "sed",
    "sort", "stat", "tail", "test", "tree", "uniq", "wc", "which",
    "whoami", "python", "python3", "node", "ruby", "go", "cargo", "npm",
    "date", "ps", "curl", "sqlite3", "gh",
    "echo", "printf", "true", "false", "env", "id", "hostname", "uname",
    "realpath", "clear", "tput", "basename",
    "hermes",
    "[", "[[",
})
_GIT_READ_ONLY = frozenset({
    "branch", "diff", "log", "ls-files", "remote", "rev-parse", "show",
    "status", "tag",
    "worktree", "fetch", "cherry", "merge-base", "rev-list", "cat-file",
    "for-each-ref", "show-ref", "count-objects", "ls-remote", "describe",
    "blame",
})
_WRITE_MARKERS = re.compile(
    r"(?:^|[^<])(?:>>?|\btee\b|(?:^|\s)(?:-delete|-exec)\b|\bsed\s+[^\n]*-i\b|\bperl\s+[^\n]*-i\b|"
    r"\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\bchmod\b|\bchown\b|"
    r"\binstall\b|\btruncate\b|\btruncate\b)",
    re.IGNORECASE,
)
# Redirections that only move bytes around scratch space or duplicate an fd
# do not write repository files.  Stripped before the write-marker check so
# `hermes kanban list --json > /tmp/out.json 2>/dev/null` stays read-only.
_SCRATCH_REDIRECT_RE = re.compile(
    r"(?:[0-9]?>>?|&>)\s*(?:/tmp|/var/tmp|/dev/null)\S*|"
    r"[0-9]?>>?\s*&\d+",
)
# Content inside quotes is data, not shell syntax: `echo 'a > b'` writes
# nothing even though it contains a `>` character.  Mask quoted regions
# before the write-marker scan; heredoc bodies (`<<EOF ... EOF`) are stdin
# input, not writes, and are handled separately by the marker's `>`/`tee`
# alternatives when combined with an actual file target.
_QUOTED_REGION_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
# Shell control keywords are structural, not actions: `for`, `do`, `done`,
# `if`, `then`, `fi`, etc.  A fragment that is only control structure (from
# quote-aware `_command_parts` splitting) must not fail read-only probes.
_SHELL_CONTROL_RE = re.compile(
    r"^(?:do|done|then|fi|else|elif|while|until|case|esac|in|if)\b\s*", re.I,
)
_SHELL_FOR_HEADER_RE = re.compile(r"^for\s+\S+\s+in\b", re.I)
# Interpreter (python/node/ruby) probes fail closed only on actual write or
# execute patterns, not on read-only words like `open`/`run`/`call`.
_INTERPRETER_WRITE_RE = re.compile(
    r"(?:open\s*\([^)]*(?:['\"](?:w|a|x|wb|ab)|mode\s*=\s*['\"](?:w|a|x))|"
    r"\.write(?:File|FileSync|_text|_bytes)?\s*\(|"
    r"\b(?:os|Path|shutil)\.(?:unlink|remove|rename|replace|mkdir|rmdir|"
    r"rmtree|copytree|copy|move|system|popen)\s*\(|"
    r"\b(?:unlink|mkdir|rmdir|rmtree)\s*\(|"
    r"\b(?:subprocess|check_call|check_output)\b)",
    re.I,
)
_CODING_AGENT_RE = re.compile(r"\b(codex|cursor|claude)(?:\s+(?:code|cli|agent))?\b", re.I)
_CODING_INTENT_RE = re.compile(
    r"\b(?:implement|modify|edit|write|add|remove|delete|fix|refactor|patch|"
    r"change|create|build|develop|code|commit|branch|worktree|checkout|launch|ship)\b",
    re.I,
)
_RESEARCH_RE = re.compile(r"\b(?:explain|research|inspect|査|review|analy[sz]e|look\s+up)\b", re.I)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _has_coding_intent(value: Any) -> bool:
    text = _text(value)
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return bool(_CODING_INTENT_RE.search(text) or _CODING_INTENT_RE.search(normalized))


def _first_line(value: str, fallback: str) -> str:
    line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return (line or fallback)[:240]


def _command_parts(command: str) -> list[str]:
    """Split a shell command enough to classify each simple command.

    Quote-aware: separators (&&, ||, |, ;) only split OUTSIDE single/double
    quotes, so a semicolon inside a quoted argument (e.g. a multi-statement
    sqlite3 query string) is not mistaken for a command separator. A bare
    fragment from such a split would fail read-only classification and spawn
    a junk DEV task.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        # Outside quotes: a separator ends the current simple command.
        # A lone `&` stays a separator (background command), EXCEPT when it is
        # part of an fd-dup redirect (`2>&1`, `>&2`, `<&3`): splitting there
        # would produce `2>` + `1` fragments that trip the write-marker check.
        if ch in "&|;":
            if ch == "&" and i + 1 < n and command[i + 1] == "&":
                i += 2
            elif ch == "|" and i + 1 < n and command[i + 1] == "|":
                i += 2
            elif ch == "&" and (
                (buf and buf[-1] in "><") or (i + 1 < n and command[i + 1] in "><")
            ):
                buf.append(ch)
                i += 1
                continue
            else:
                i += 1
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _simple_command_is_read_only(command: str) -> bool:
    command = command.strip()
    if not command:
        return False
    # Shell control fragments (`for id in ...`, `do`, `done`, `if`, `then`,
    # `fi`) are structural, not actions.  A loop header binds variables only;
    # the body (split out by _command_parts) is what gets classified.
    if _SHELL_FOR_HEADER_RE.match(command):
        return True
    stripped_control = _SHELL_CONTROL_RE.sub("", command)
    if not stripped_control:
        return True
    command = stripped_control
    # Scratch-space and fd-dup redirections are not repository writes.  Quote
    # contents are data (`echo 'a > b'` writes nothing) so mask them first;
    # heredoc bodies (`<<EOF`) are stdin, and only a real file target (`>` or
    # `tee`) after the marker check gates.
    write_scan = _QUOTED_REGION_RE.sub(" ", _SCRATCH_REDIRECT_RE.sub("", command))
    if not command or _WRITE_MARKERS.search(write_scan):
        return False
    if re.search(r"\b(?:codex|cursor|claude|aider|copilot|gh\s+pr\s+(?:checkout|create))\b", command, re.I):
        return False
    try:
        words = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    if not words:
        return True
    # Environment assignments (`VAR=...`), `export VAR=...`, and `env`
    # prefixes are wrappers; only the wrapped command matters. `env` is
    # stripped along with its option tokens (`-i`, `-u NAME`, `--unset=...`,
    # `VAR=...`) before classifying the wrapped command; a bare `env` with
    # no wrapped command is read-only (it just prints the environment).
    while words:
        w0 = words[0]
        if "=" in w0 and not w0.startswith("="):
            words.pop(0)
            continue
        if w0.lower() == "export" and len(words) > 1 and "=" in words[1] and not words[1].startswith("="):
            words.pop(0)
            continue
        if w0.lower() == "env":
            i = 1
            while i < len(words):
                w = words[i]
                if w in ("-i", "-0"):
                    i += 1
                elif w in ("-u", "--unset") and i + 1 < len(words):
                    i += 2
                elif w.startswith("--unset="):
                    i += 1
                elif "=" in w and not w.startswith("="):
                    i += 1
                else:
                    break
            rest = words[i:]
            if not rest:
                return True
            return _simple_command_is_read_only(" ".join(rest))
        break
    if not words:
        return True
    base = Path(words[0]).name.lower()
    if base not in _READ_ONLY_COMMANDS:
        return False
    if base == "curl":
        # curl is read-only only for inspection probes. Output flags that
        # write files (-o/--output/-O/--remote-name/--output-dir/--remote-header-name)
        # and request-body/mutation flags (-d/--data*, -F/--form, -T/--upload-file,
        # -X/--request) fail closed before task association.
        for word in words[1:]:
            if word.startswith("--"):
                name = word[2:].split("=", 1)[0].lower()
                if name in {
                    "output", "remote-name", "remote-name-all", "output-dir",
                    "remote-header-name", "data", "data-raw", "data-binary",
                    "data-urlencode", "data-ascii", "form", "upload-file",
                    "request",
                }:
                    return False
            elif word.startswith("-") and len(word) > 1:
                # Combined short flags (e.g. -sfo) fail closed if any
                # mutating letter is present; -f (--fail) and -I (--head)
                # remain safe.
                if any(c in word[1:] for c in "oOJdFTX"):
                    return False
    if base == "git":
        try:
            subcommand = next(word for word in words[1:] if not word.startswith("-"))
        except StopIteration:
            return False
        if subcommand not in _GIT_READ_ONLY:
            return False
        if subcommand == "remote":
            # `git remote` is read-only only for inspection.  `add`, `remove`,
            # and `set-*` mutate repository configuration before any worker
            # could be authorized.
            remote_args = [word for word in words[2:] if not word.startswith("-")]
            if remote_args and words[2] not in {"show", "get-url"}:
                return False
        if subcommand == "branch":
            # `git branch --show-current`, `-a`, and `--list` inspect; a
            # positional branch name creates a branch.
            if any(not word.startswith("-") for word in words[2:]):
                return False
        if subcommand == "tag" and any(not word.startswith("-") for word in words[2:]):
            return False
        if subcommand == "worktree":
            # Only `git worktree list` is read-only; add/remove/move/repair
            # mutate the repo and must fail closed.
            wt_args = [word for word in words[2:] if not word.startswith("-")]
            if wt_args != ["list"]:
                return False
    if base in {"python", "python3", "node", "ruby", "go", "cargo", "npm"}:
        # Package/build commands mutate caches or the workspace.  Inline
        # calculations and version/help probes remain available.
        lowered = command.lower()
        if base in {"npm", "cargo", "go"} and not re.search(r"\b(?:--version|version|help)\b", lowered):
            return False
        # Read probes (`json.load(open('/tmp/x.json'))`, sqlite3 SELECTs) are
        # read-only; fail closed only on actual write/execute calls.
        if _INTERPRETER_WRITE_RE.search(command):
            return False
    if base == "gh":
        # Keep status/view/list probes available, but never permit a mutating
        # GitHub subcommand to slip through the read-only classification.
        try:
            gh_subcommand = next(word for word in words[1:] if not word.startswith("-"))
        except StopIteration:
            return False
        if gh_subcommand not in {"auth", "pr", "run", "repo", "issue"}:
            return False
        if any(word in {"create", "close", "merge", "edit", "delete", "checkout", "comment", "rerun"} for word in words[1:]):
            return False
    if base == "hermes":
        # Read-only Hermes CLI probes must not create gate-junk DEV cards.
        # Allow read subcommands; fail closed for everything mutating.
        _HERMES_READ_ONLY = {
            "webhook": {"list"},
            "kanban": {"list", "ls", "show", "stats", "diagnostics", "diag",
                       "context", "runs", "assignees", "notify-list",
                       "attachments", "log", "tail", "boards"},
            "gateway": {"status", "doctor", "version", "health", "info"},
            "config": {"get", "list", "show"},
            "cron": {"list"},
            "session": {"list", "show", "search"},
            "version": set(),
            "doctor": set(),
            "help": set(),
            "skills": {"list"},
        }
        # `boards` is read-only ONLY for list/current inspection; create/rm/
        # switch/rename/set-workdir mutate board configuration and must fail
        # closed.  `repair` and `gc` are mutating and deliberately absent.
        _HERMES_KANBAN_BOARDS_READ_ONLY = frozenset({"list", "current"})
        try:
            hermes_cmd = next(word for word in words[1:] if not word.startswith("-"))
        except StopIteration:
            return False
        allowed = _HERMES_READ_ONLY.get(hermes_cmd)
        if allowed is None:
            return False
        if not allowed:
            return True
        try:
            hermes_sub = next(word for word in words[2:] if not word.startswith("-"))
        except StopIteration:
            return False
        if hermes_cmd == "kanban" and hermes_sub == "boards":
            try:
                boards_sub = next(word for word in words[3:] if not word.startswith("-"))
            except StopIteration:
                return False
            return boards_sub in _HERMES_KANBAN_BOARDS_READ_ONLY
        if hermes_sub not in allowed:
            return False
    return True


def _write_file_is_coding(args: dict, user_message: Any) -> bool:
    """Permit ordinary report/artifact authoring outside the repository.

    ``write_file`` is also the file tool used for reports and research notes.
    Those must not be forced through the coding lane merely because the tool
    can write.  Repository files, or a request explicitly describing code
    work, remain coding operations.
    """
    path = _text(args.get("path") or args.get("file_path"))
    message = _text(user_message)
    report_request = bool(re.search(r"\b(?:report|audit|research|findings|notes?)\b", message, re.I))
    if (_has_coding_intent(message) or Path(path).suffix.lower() in {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb", ".sh"}) and not report_request:
        return True
    if not path:
        return True
    try:
        candidate = Path(path).expanduser().resolve()
        repo = Path(_repository(_workspace(args), args)).resolve()
        if candidate == repo or repo in candidate.parents:
            return True
        if report_request:
            return False
    except OSError:
        return True
    return False


def _execute_code_is_read_only(code: str) -> bool:
    code = _text(code)
    if not code:
        return True
    try:
        import ast
        tree = ast.parse(code)
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for item in node.names:
                    if item.name in {"write_file", "patch", "terminal"}:
                        aliases.add(item.asname or item.name)
            elif isinstance(node, ast.Import):
                for item in node.names:
                    if item.name.rsplit(".", 1)[-1] in {"write_file", "patch", "terminal"}:
                        aliases.add(item.asname or item.name.rsplit(".", 1)[-1])
        if aliases and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in aliases
            for node in ast.walk(tree)
        ):
            return False
    except (SyntaxError, ValueError):
        return False
    if re.search(
        r"(?:open\s*\([^)]*(?:['\"](?:w|a|x|wb|ab)|mode\s*=\s*['\"](?:w|a|x))|"
        r"\.write(?:_text|_bytes)?\s*\(|"
        r"\b(?:os|Path|pathlib|shutil)\.(?:unlink|remove|rename|replace|mkdir|rmdir|rmtree|copytree|copy|move)\s*\(|"
        r"\b(?:unlink|mkdir|rmdir|rmtree)\s*\(|"
        r"\b(?:subprocess|os\.system|os\.popen)\b|"
        r"\b(?:write_file|patch|terminal)\s*\()",
        code,
        re.I,
    ):
        return False
    return True


def is_coding_intent(tool_name: str, args: Optional[dict] = None, user_message: Any = None) -> bool:
    """Return whether this call could start implementation work."""
    args = args if isinstance(args, dict) else {}
    if tool_name == "codex_app_server":
        return bool(_has_coding_intent(user_message) and not (
            _RESEARCH_RE.search(_text(user_message))
            and not _has_coding_intent(_text(user_message))
        ))
    if tool_name == "write_file":
        return _write_file_is_coding(args, user_message)
    if tool_name == "patch":
        return True
    if tool_name == "terminal":
        command = _text(args.get("command"))
        return not command or not all(_simple_command_is_read_only(part) for part in _command_parts(command))
    if tool_name == "execute_code":
        return not _execute_code_is_read_only(_text(args.get("code")))
    if tool_name == "delegate_task":
        goals: list[str] = []
        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for child in value.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)
            elif value:
                goals.append(_text(value))
        collect(args)
        text = "\n".join(item for item in goals if item)
        # A review/research request is read-only unless it also asks for a
        # change.  This keeps documentation/research delegation available.
        return bool(_has_coding_intent(text) and not (
            _RESEARCH_RE.search(text) and not _has_coding_intent(text)
        ))
    if tool_name in {"project_create", "project_switch"}:
        return bool(re.search(r"\b(?:branch|worktree|repository|repo|codebase|implementation)\b", _text(user_message), re.I))
    return False


def _requested_coding_agent(args: dict, user_message: Any) -> tuple[str, Optional[str]]:
    explicit = _text(args.get("coding_agent") or args.get("agent"))
    source = "\n".join(
        item for item in (
            explicit,
            _text(user_message),
            _text(args.get("command")),
            _text(args.get("goal")),
            _text(args.get("context")),
        ) if item
    )
    matches = _CODING_AGENT_RE.findall(source)
    if explicit:
        requested = explicit.casefold()
    elif matches:
        requested = matches[-1].casefold()
    else:
        requested = "codex"
    if requested == "claude":
        return requested, "Claude is not an allowed coding worker; use Codex or explicitly request Cursor."
    if requested not in SUPPORTED_CODING_AGENTS:
        return requested, f"Unsupported coding agent {requested!r}; use Codex or Cursor."
    return requested, None


def _workspace(args: dict) -> str:
    raw = _text(args.get("workspace") or args.get("workdir") or os.environ.get("TERMINAL_CWD"))
    if not raw:
        raw = os.getcwd()
    try:
        return str(Path(raw).expanduser().resolve())
    except OSError:
        return raw


def _repository(workspace: str, args: dict) -> str:
    explicit = _text(args.get("repository") or args.get("repo"))
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ["git", "-C", workspace, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return str(Path(result.stdout.strip()).resolve())
    except (OSError, subprocess.SubprocessError):
        pass
    return workspace


def _metadata(*, agent: str, session_id: str, message_id: str, scope: str, repository: str, workspace: str, acceptance: list[str]) -> dict:
    return {
        "canonical": True,
        "lane": "DEV",
        "coding_agent": agent,
        "origin": {"session_id": session_id, "message_id": message_id},
        "scope": scope,
        "repository": repository,
        "workspace": workspace,
        "acceptance_criteria": acceptance,
    }


def _refusal(*, error: str, tool_name: str, **fields: Any) -> str:
    payload = {"error": error, "error_type": "kanban_task_required", "tool": tool_name}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def _worker_refusal(tool_name: str, task_id: str, reason: str) -> str:
    return _refusal(
        error=f"Kanban worker {task_id!r} cannot use {tool_name}: {reason}",
        tool_name=tool_name,
        task_id=task_id,
    )


def _task_is_review_lane(task) -> bool:
    """Return whether this task was claimed into the SDLC review lane.

    Review-lane cards are claimed by reviewer profiles (e.g. ``orion``) via
    ``claim_review_task`` after the implementer submitted the card for review.
    The lane is detected from the claim history (a ``claimed`` event whose
    payload records ``source_status: "review"``) or from explicit review
    identity on the card (``review_identity`` metadata or a ``review_submitted``
    event).  A later plain ``claimed`` event (a changes-requested requeue back
    to the implementation lane) ends the review lane.
    """
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if _text(metadata.get("review_identity")):
        return True
    try:
        from hermes_cli import kanban_db

        conn = kanban_db.connect()
        try:
            events = kanban_db.list_events(conn, task.id)
        finally:
            conn.close()
    except Exception:
        return False
    # Newest-first: the most recent claim decides the lane.  A claim from
    # status=review marks a reviewer-owned run; a plain ready->running claim
    # after a review (changes requested) returns the card to the
    # implementation lane.
    for event in reversed(events):
        if event.kind == "claimed":
            payload = event.payload if isinstance(event.payload, dict) else {}
            return payload.get("source_status") == "review"
        if event.kind == "review_submitted":
            return True
    return False


def _canonical_worker_task(task_id: str, *, tool_name: Optional[str] = None):
    from hermes_cli import kanban_db

    conn = kanban_db.connect()
    try:
        task = kanban_db.get_task(conn, task_id)
    finally:
        conn.close()
    if task is None:
        return None, "task does not exist"
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    run_id = _text(os.environ.get("HERMES_KANBAN_RUN_ID"))
    if not run_id or task.current_run_id is None or str(task.current_run_id) != run_id:
        return task, "worker is not the active dispatcher run for this task"
    if tool_name is not None and _task_is_review_lane(task):
        # Review-lane workers are reviewers (e.g. ``orion``), not
        # implementation profiles.  They need the terminal mutations the SDLC
        # review requires (``gh pr merge``/``close``/``review``, ``git push``/
        # ``fetch``, deploy scripts, service checks) but must not write
        # implementation code.  The session path (``tool_name`` is None) never
        # inherits review-lane privileges.
        if tool_name in {"write_file", "patch", "execute_code"}:
            return task, "review-lane workers cannot write implementation code"
        if tool_name in {"delegate_task", "codex_app_server", "project_create", "project_switch"}:
            return task, "review-lane workers cannot start coding work"
        return task, None
    if task.assignee not in IMPLEMENTATION_ASSIGNEES:
        return task, "task is not assigned to a validated implementation profile"
    if task.status not in ACTIVE_TASK_STATUSES:
        return task, f"task status is {task.status!r}"
    expected_lane = "DEV" if task.assignee == "dev" else task.assignee.upper()
    if metadata.get("canonical") is not True or metadata.get("lane") not in {expected_lane, "ENGINEERING"}:
        return task, "task lacks canonical implementation metadata"
    if metadata.get("coding_agent") not in SUPPORTED_CODING_AGENTS:
        return task, "task has no supported coding-agent metadata"
    return task, None


def _canonical_worker_task_for_session(task_id: str, session_id: str):
    task, reason = _canonical_worker_task(task_id)
    if reason:
        # Session-associated chat calls do not have a dispatcher run, so the
        # worker-only run check is not applicable. Revalidate the task's
        # canonical ownership and active status without weakening those checks.
        try:
            from hermes_cli import kanban_db
            conn = kanban_db.connect()
            try:
                task = kanban_db.get_task(conn, task_id)
            finally:
                conn.close()
        except Exception:
            return None, "Kanban is unavailable"
        if task is None:
            return None, "task does not exist"
        metadata = task.metadata if isinstance(task.metadata, dict) else {}
        if (
            task.assignee not in IMPLEMENTATION_ASSIGNEES
            or task.status not in ACTIVE_TASK_STATUSES
            or metadata.get("canonical") is not True
            or metadata.get("coding_agent") not in SUPPORTED_CODING_AGENTS
            or metadata.get("origin", {}).get("session_id") != session_id
        ):
            return task, "session is not associated with a canonical active implementation task"
        return task, None
    return task, None


def _session_task_id(session_id: str) -> Optional[str]:
    if not session_id:
        return None
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            row = db.get_session(session_id)
        finally:
            db.close()
        config = row.get("model_config") if row else None
        if isinstance(config, str):
            config = json.loads(config)
        return _text(config.get("kanban_task_id")) or None if isinstance(config, dict) else None
    except Exception:
        return None


def _call_is_scoped_to_task(tool_name: str, args: dict, task) -> tuple[bool, str]:
    """Return ``(ok, reason)`` whether this call's explicit targets stay
    inside the associated task's recorded repository/workspace.

    The session-association allow path must not unlock coding calls that
    target unrelated repositories, workspaces, or paths.  Every explicit
    target the call names is compared against the task metadata recorded at
    intake; a mismatch keeps the call gated instead of silently allowing it.
    """
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    task_repo = _text(metadata.get("repository"))
    task_ws = _text(metadata.get("workspace")) or _text(getattr(task, "workspace_path", None))
    bases = [p for p in (task_repo, task_ws) if p]
    if not bases:
        return True, "task has no recorded scope"

    def inside(path: str) -> bool:
        try:
            candidate = Path(path).expanduser().resolve()
        except OSError:
            return False
        for base in bases:
            try:
                base_path = Path(base).expanduser().resolve()
            except OSError:
                continue
            if candidate == base_path or base_path in candidate.parents:
                return True
        return False

    if tool_name in {"write_file", "patch"}:
        path = _text(args.get("path") or args.get("file_path"))
        if path and not inside(path):
            return False, f"target path {path!r} is outside the task repository/workspace"
        return True, ""

    if tool_name == "terminal":
        workdir = _text(args.get("workdir") or args.get("cwd") or os.environ.get("TERMINAL_CWD"))
        if workdir and not inside(workdir):
            return False, f"workdir {workdir!r} is outside the task repository/workspace"
        # A command that explicitly cd's or targets an absolute path outside
        # the task workspace must fail closed even without a workdir arg.
        command = _text(args.get("command"))
        if command:
            for part in _command_parts(command):
                for match in re.finditer(r"\b(?:cd|pushd|git\s+-C)\s+['\"]?([^ '\"]+)", part):
                    target = match.group(1)
                    if target.startswith("/") and not inside(target):
                        return False, f"command target {target!r} is outside the task repository/workspace"
        return True, ""

    if tool_name == "execute_code":
        workdir = _text(args.get("workdir") or args.get("cwd"))
        if workdir and not inside(workdir):
            return False, f"workdir {workdir!r} is outside the task repository/workspace"
        return True, ""

    if tool_name == "delegate_task":
        ws = _text(args.get("workspace") or args.get("workdir"))
        repo = _text(args.get("repository") or args.get("repo"))
        if ws and not inside(ws):
            return False, f"workspace {ws!r} is outside the task repository/workspace"
        if repo and not inside(repo):
            return False, f"repository {repo!r} is outside the task repository/workspace"
        return True, ""

    if tool_name in {"project_create", "project_switch", "codex_app_server"}:
        return True, ""
    return True, ""


def _persist_session_task_id(session_id: str, task_id: str) -> None:
    if not session_id or not task_id:
        return
    try:
        from hermes_state import SessionDB
        db = SessionDB()
        try:
            row = db.get_session(session_id) or {}
            config = row.get("model_config")
            if isinstance(config, str):
                config = json.loads(config)
            if not isinstance(config, dict):
                config = {}
            config["kanban_task_id"] = task_id
            db.update_session_meta(session_id, json.dumps(config, ensure_ascii=False))
        finally:
            db.close()
    except Exception:
        return


def coding_tool_gate_refusal(
    tool_name: str,
    *,
    function_args: Optional[dict] = None,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    turn_id: Optional[str] = None,
    user_message: Any = None,
) -> Optional[str]:
    """Return a refusal/handoff payload, or ``None`` when the call may run."""
    if tool_name not in CODING_TOOLS:
        return None
    args = function_args if isinstance(function_args, dict) else {}
    worker_id = _text(os.environ.get("HERMES_KANBAN_TASK"))
    if worker_id:
        try:
            task, reason = _canonical_worker_task(worker_id, tool_name=tool_name)
        except Exception:
            return _worker_refusal(tool_name, worker_id, "Kanban is unavailable")
        if not is_coding_intent(tool_name, args, user_message):
            return None
        if reason:
            return _worker_refusal(tool_name, worker_id, reason)
        worker_agent = _text(os.environ.get("HERMES_CODING_AGENT"))
        if worker_agent and worker_agent.casefold() not in SUPPORTED_CODING_AGENTS:
            return _worker_refusal(tool_name, worker_id, "the worker provider is not supported")
        return None

    normalized_session = _text(session_id)
    if not normalized_session or not is_coding_intent(tool_name, args, user_message):
        return None

    associated_id = _session_task_id(normalized_session)
    if associated_id:
        associated_task, association_error = _canonical_worker_task_for_session(
            associated_id, normalized_session
        )
        if associated_task is not None and association_error is None:
            in_scope, scope_reason = _call_is_scoped_to_task(tool_name, args, associated_task)
            if in_scope:
                return None
            return _refusal(
                error=f"Coding call is outside the associated task's scope: {scope_reason}",
                tool_name=tool_name,
                error_type="kanban_task_scope_mismatch",
                task_id=associated_id,
            )

    agent, agent_error = _requested_coding_agent(args, user_message)
    if agent_error:
        return _refusal(error=agent_error, tool_name=tool_name, error_type="unsupported_coding_agent")

    workspace = _workspace(args)
    repository = _repository(workspace, args)
    scope = _text(args.get("scope") or args.get("goal") or args.get("context") or user_message)
    if not scope:
        # Tool-only call with no user message: title from the actual tool
        # invocation so gate-created cards describe the real request.
        tool_summary = _text(args.get("command") or args.get("path") or args.get("url"))
        if tool_summary:
            scope = "Run: " + tool_summary[:200]
        else:
            scope = "Implement the requested code change."
    acceptance_value = args.get("acceptance_criteria")
    if isinstance(acceptance_value, str):
        acceptance = [acceptance_value.strip()] if acceptance_value.strip() else []
    elif isinstance(acceptance_value, list):
        acceptance = [_text(item) for item in acceptance_value if _text(item)]
    else:
        acceptance = []
    if not acceptance:
        acceptance = ["Implement the requested change in the recorded repository and workspace.", "Verify the change before completing the DEV task."]
    message_id = _text(args.get("origin_message_id") or turn_id or task_id)
    if not message_id:
        message_id = hashlib.sha256(_text(user_message).encode("utf-8")).hexdigest()[:16]
    metadata = _metadata(
        agent=agent,
        session_id=normalized_session,
        message_id=message_id,
        scope=scope,
        repository=repository,
        workspace=workspace,
        acceptance=acceptance,
    )
    idempotency_payload = {
        "session_id": normalized_session,
        "message_id": message_id,
        "scope": scope,
        "repository": repository,
        "workspace": workspace,
        "coding_agent": agent,
    }
    idempotency_key = "chat-coding:" + hashlib.sha256(
        json.dumps(idempotency_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    try:
        from hermes_cli import kanban_db

        conn = kanban_db.connect()
        try:
            existing_id = None
            for candidate in kanban_db.list_tasks(conn, session_id=normalized_session):
                candidate_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
                if (
                    candidate.assignee == CANONICAL_ASSIGNEE
                    and candidate_meta.get("canonical") is True
                    and candidate_meta.get("coding_agent") == agent
                    and candidate_meta.get("origin") == metadata["origin"]
                    and candidate_meta.get("repository") == repository
                    and candidate_meta.get("workspace") == workspace
                    and candidate.status in ACTIVE_TASK_STATUSES
                ):
                    existing_id = candidate.id
                    break
            created_id = existing_id or kanban_db.create_task(
                conn,
                title=f"{_first_line(scope, 'Implement requested change')}",
                body="\n".join([scope, "", "Acceptance criteria:", *[f"- {item}" for item in acceptance]]),
                assignee=CANONICAL_ASSIGNEE,
                created_by="chat",
                workspace_kind="dir",
                workspace_path=workspace,
                idempotency_key=idempotency_key,
                session_id=normalized_session,
                metadata=metadata,
                initial_status="running",
            )
            task = kanban_db.get_task(conn, created_id)
        finally:
            conn.close()
    except Exception:
        return _refusal(
            error="Kanban is unavailable; coding work is blocked until the DEV board can be reached.",
            tool_name=tool_name,
            error_type="kanban_unavailable",
        )
    if task is None or task.assignee not in IMPLEMENTATION_ASSIGNEES or not isinstance(task.metadata, dict):
        return _refusal(
            error="Kanban did not return a canonical DEV task; coding work is blocked.",
            tool_name=tool_name,
            error_type="kanban_task_invalid",
        )
    _persist_session_task_id(normalized_session, task.id)
    return _refusal(
        error="Coding work was handed off to the DEV Kanban worker. The originating chat remains read-only for this change.",
        tool_name=tool_name,
        task_id=task.id,
        status=task.status,
        assignee=task.assignee,
        coding_agent=task.metadata.get("coding_agent"),
        reused=existing_id is not None,
    )
