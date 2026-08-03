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
})
CANONICAL_ASSIGNEE = "dev"
SUPPORTED_CODING_AGENTS = frozenset({"codex", "cursor"})
ACTIVE_TASK_STATUSES = frozenset({"todo", "ready", "running", "review"})

_READ_ONLY_COMMANDS = frozenset({
    "awk", "cat", "cut", "diff", "dirname", "du", "file", "find",
    "git", "grep", "head", "jq", "ls", "pwd", "readlink", "rg", "sed",
    "sort", "stat", "tail", "test", "tree", "uniq", "wc", "which",
    "whoami", "python", "python3", "node", "ruby", "go", "cargo", "npm",
})
_GIT_READ_ONLY = frozenset({
    "branch", "diff", "log", "ls-files", "remote", "rev-parse", "show",
    "status", "tag",
})
_GIT_GLOBAL_OPTIONS_WITH_VALUES = frozenset({
    "-C", "-c", "--config-env", "--exec-path", "--git-dir",
    "--namespace", "--super-prefix", "--work-tree",
})
_GIT_GLOBAL_OPTIONS = frozenset({
    "--bare", "--no-pager", "--paginate", "--no-replace-objects",
    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
    "--icase-pathspecs", "--no-optional-locks",
})
_GIT_BRANCH_READ_ONLY_OPTIONS = frozenset({
    "-a", "--all", "-l", "--list", "-r", "--remotes", "-v", "--verbose",
    "-vv", "--contains", "--no-contains", "--merged", "--no-merged",
    "--points-at", "--format", "--sort", "--column", "--no-column",
    "--color", "--no-color", "--abbrev", "--omit-empty", "--show-current",
})
_GIT_TAG_READ_ONLY_OPTIONS = frozenset({
    "-l", "--list", "-n", "--contains", "--no-contains", "--merged",
    "--no-merged", "--points-at", "--column", "--no-column", "--sort",
    "--format", "--color", "--no-color", "--omit-empty", "-v", "--verify",
})
_GIT_TAG_MUTATING_OPTIONS = frozenset({
    "-a", "--annotate", "-s", "--sign", "-u", "--local-user", "-f",
    "--force", "-d", "--delete", "--create-reflog", "--edit",
})
_GIT_REMOTE_READ_ONLY_SUBCOMMANDS = frozenset({"show", "get-url"})
_GIT_REMOTE_MUTATING_SUBCOMMANDS = frozenset({
    "add", "rename", "remove", "rm", "set-head", "set-branches", "set-url",
    "update", "prune",
})
_GIT_DIFF_READ_ONLY_OPTIONS = frozenset({
    "--cached", "--staged", "--merge-base", "--no-index", "--no-ext-diff",
    "--text", "--ignore-space-at-eol", "--ignore-space-change",
    "--ignore-all-space", "-w", "-b", "--ignore-blank-lines", "--patience",
    "--histogram", "--minimal", "--raw", "--patch", "-p", "--no-patch",
    "-u", "--unified", "--stat", "--numstat", "--shortstat", "--dirstat",
    "--summary", "--name-only", "--name-status", "--submodule", "--no-renames",
    "--find-renames", "-M", "--find-copies", "-C", "--full-index", "--binary",
    "--no-full-index", "--textconv", "--check", "--quiet", "--exit-code",
    "--relative", "--no-relative", "--color", "--no-color", "--abbrev",
})
_WRITE_MARKERS = re.compile(
    r"(?:^|[^<])(?:>>?|<<|\btee\b|\bsed\s+[^\n]*-i\b|\bperl\s+[^\n]*-i\b|"
    r"\brm\b|\bmv\b|\bcp\b|\bmkdir\b|\btouch\b|\bchmod\b|\bchown\b|"
    r"\binstall\b|\btruncate\b|\btruncate\b)",
    re.IGNORECASE,
)
_INLINE_MUTATION_APIS = re.compile(
    r"(?:\b(?:writeFile|writeFileSync|appendFile|appendFileSync|unlink|unlinkSync|"
    r"rm|rmSync|rename|renameSync|mkdir|mkdirSync|rmdir|rmdirSync|copyFile|"
    r"copyFileSync|cp|cpSync|writeSync|ftruncate|ftruncateSync|chmod|chmodSync|"
    r"chown|chownSync|symlink|symlinkSync|link|linkSync|truncate|truncateSync|"
    r"mkdtemp|mkdtempSync|createWriteStream)\s*\(|"
    r"\b(?:child_process\s*\.\s*)?(?:exec|execSync|execFile|execFileSync|"
    r"spawn|spawnSync|fork|Popen)\s*\(|"
    r"\b(?:File|FileUtils)\s*\.\s*(?:write|binwrite|delete|unlink|rename|"
    r"chmod|chown|truncate|symlink|link|open|mkpath|mkdir|mkdir_p|rmdir|cp|mv|"
    r"rm|touch|remove_entry|remove_file)\s*\(|"
    r"\b(?:system|popen|IO\.popen|Process\.spawn|Process\.exec)\s*\(|"
    r"\.(?:write_text|write_bytes|touch|unlink|rename|replace|mkdir|rmdir|"
    r"chmod|lchmod|chown|lchown|symlink_to|hardlink_to)\s*\(|"
    r"\bprocess\.env(?:\.|\[)|\b(?:os\.)?open\s*\([^)]*\b(?:O_WRONLY|O_RDWR|O_CREAT|O_APPEND)\b)",
    re.IGNORECASE,
)
_CODING_AGENT_RE = re.compile(r"\b(codex|cursor|claude)(?:\s+(?:code|cli|agent))?\b", re.I)
_CODING_INTENT_RE = re.compile(
    r"\b(?:implement|modify|edit|write|add|remove|delete|fix|refactor|patch|"
    r"change|create|build|develop|code|commit|branch|worktree|checkout|launch)\b",
    re.I,
)
_RESEARCH_RE = re.compile(r"\b(?:explain|research|inspect|査|review|analy[sz]e|look\s+up)\b", re.I)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first_line(value: str, fallback: str) -> str:
    line = next((line.strip() for line in value.splitlines() if line.strip()), "")
    return (line or fallback)[:240]


def _command_parts(command: str) -> list[str]:
    """Split a shell command enough to classify each simple command."""
    parts = re.split(r"\s*(?:&&|\|\||[|;])\s*", command)
    return [part.strip() for part in parts if part.strip()]


def _git_subcommand(words: list[str]) -> tuple[Optional[str], int]:
    """Return the Git subcommand and its index, rejecting ambiguous globals."""
    index = 1
    while index < len(words):
        word = words[index]
        if word == "--":
            return None, index
        if not word.startswith("-"):
            return word, index
        option = word.split("=", 1)[0]
        if option in _GIT_GLOBAL_OPTIONS_WITH_VALUES:
            if "=" not in word:
                index += 1
                if index >= len(words):
                    return None, index
        elif option not in _GIT_GLOBAL_OPTIONS:
            return None, index
        index += 1
    return None, index


def _git_options_are_read_only(
    args: list[str], allowed: frozenset[str], *, value_options: frozenset[str] = frozenset(),
) -> bool:
    """Accept only known inspection options for the operation-specific Git paths."""
    index = 0
    while index < len(args):
        word = args[index]
        if word == "--":
            return True
        if not word.startswith("-"):
            index += 1
            continue
        option = word.split("=", 1)[0]
        if option not in allowed and option not in value_options:
            return False
        if option in value_options and "=" not in word:
            index += 1
            if index >= len(args):
                return False
        index += 1
    return True


def _git_is_read_only(words: list[str]) -> bool:
    subcommand, subcommand_index = _git_subcommand(words)
    if subcommand not in _GIT_READ_ONLY:
        return False
    args = words[subcommand_index + 1:]

    if subcommand == "branch":
        # Branch names make the default form create a branch; all branch
        # mutation flags are rejected even when a later positional is absent.
        if any(
            word.split("=", 1)[0] in {"-d", "-D", "-m", "-M", "-c", "-C", "--delete", "--move", "--copy"}
            or (word.startswith(("-d", "-D", "-m", "-M", "-c", "-C")) and word != "--")
            for word in args
        ):
            return False
        if any(not word.startswith("-") for word in args):
            return False
        return _git_options_are_read_only(
            args, _GIT_BRANCH_READ_ONLY_OPTIONS,
            value_options=frozenset({"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--format", "--sort", "--column", "--color", "--abbrev"}),
        )

    if subcommand == "tag":
        if any(
            word.split("=", 1)[0] in _GIT_TAG_MUTATING_OPTIONS
            or word.startswith(("-a", "-s", "-u", "-f", "-d")) and word not in {"--"}
            for word in args
        ):
            return False
        # `git tag` lists tags.  Verification is also read-only; any other
        # positional tag name is ambiguous because it can create a tag.
        verify = any(word in {"-v", "--verify"} for word in args)
        if any(not word.startswith("-") for word in args) and not verify and not any(
            word in {"-l", "--list"} for word in args
        ):
            return False
        return _git_options_are_read_only(
            args, _GIT_TAG_READ_ONLY_OPTIONS,
            value_options=frozenset({"--contains", "--no-contains", "--merged", "--no-merged", "--points-at", "--column", "--sort", "--format", "--color"}),
        )

    if subcommand == "remote":
        if not args:
            return True
        operation = next((word for word in args if not word.startswith("-")), None)
        if operation is None:
            return _git_options_are_read_only(args, frozenset({"-v", "--verbose"}))
        if operation in _GIT_REMOTE_MUTATING_SUBCOMMANDS or operation not in _GIT_REMOTE_READ_ONLY_SUBCOMMANDS:
            return False
        return _git_options_are_read_only(args, frozenset({"-v", "--verbose"}))

    if subcommand == "diff":
        if any(word == "--output" or word.startswith("--output=") or word == "-o" or word.startswith("-o") for word in args):
            return False
        return _git_options_are_read_only(
            args, _GIT_DIFF_READ_ONLY_OPTIONS,
            value_options=frozenset({"--unified", "-U", "--diff-algorithm", "--anchored", "--word-diff", "--word-diff-regex", "--ignore-matching-lines", "-I", "--src-prefix", "--dst-prefix", "--line-prefix", "--inter-hunk-context", "--output-indicator-new", "--output-indicator-old", "--output-indicator-context", "--dirstat-by-file", "--submodule"}),
        )

    return True


def _contains_inline_mutation(code: str) -> bool:
    return bool(_INLINE_MUTATION_APIS.search(code))


def _simple_command_is_read_only(command: str) -> bool:
    command = command.strip()
    if not command or _WRITE_MARKERS.search(command):
        return False
    if re.search(r"\b(?:codex|cursor|claude|aider|copilot|gh\s+pr\s+(?:checkout|create))\b", command, re.I):
        return False
    try:
        words = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    if not words:
        return True
    # Environment assignments and command wrappers are harmless only when
    # the wrapped command itself is an explicitly read-only command.
    while words and ("=" in words[0] and not words[0].startswith("=")):
        words.pop(0)
    if not words:
        return True
    base = Path(words[0]).name.lower()
    if base not in _READ_ONLY_COMMANDS:
        return False
    if base == "find" and any(word in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for word in words[1:]):
        return False
    if base == "git":
        return _git_is_read_only(words)
    if base in {"python", "python3", "node", "ruby", "go", "cargo", "npm"}:
        # Package/build commands mutate caches or the workspace.  Inline
        # calculations and version/help probes remain available.
        lowered = command.lower()
        if base in {"npm", "cargo", "go"} and not re.search(r"\b(?:--version|version|help)\b", lowered):
            return False
        if _contains_inline_mutation(command) or re.search(
            r"\b(?:open|remove|replace|run|call|check_call|check_output|system|popen|exec|spawn)\s*\(",
            command, re.I,
        ):
            return False
    return True


def _execute_code_is_read_only(code: str) -> bool:
    code = _text(code)
    if not code:
        return True
    if re.search(
        r"(?:open\s*\([^)]*(?:['\"](?:w|a|x|r\+|w\+|a\+|x\+|wb|ab|wb\+|ab\+)|mode\s*=\s*['\"](?:w|a|x|r\+|w\+|a\+|x\+|wb|ab|wb\+|ab\+))|"
        r"\.write(?:_text|_bytes)?\s*\(|\b(?:unlink|remove|rename|replace|mkdir|makedirs|rmdir|chmod|chown|symlink|link|truncate)\s*\(|"
        r"\b(?:subprocess|os\.system|os\.popen|shutil\.(?:copy|copy2|copyfile|copytree|move|rmtree))\b|"
        r"\b(?:write_file|patch|terminal)\s*\()",
        code,
        re.I,
    ):
        return False
    return not _contains_inline_mutation(code)


def is_coding_intent(tool_name: str, args: Optional[dict] = None, user_message: Any = None) -> bool:
    """Return whether this call could start implementation work."""
    args = args if isinstance(args, dict) else {}
    if tool_name in {"write_file", "patch"}:
        return True
    if tool_name == "terminal":
        command = _text(args.get("command"))
        return not command or not all(_simple_command_is_read_only(part) for part in _command_parts(command))
    if tool_name == "execute_code":
        return not _execute_code_is_read_only(_text(args.get("code")))
    if tool_name == "delegate_task":
        goals = [_text(args.get("goal")), _text(args.get("context"))]
        for item in args.get("tasks") or []:
            if isinstance(item, dict):
                goals.extend((_text(item.get("goal")), _text(item.get("context"))))
        text = "\n".join(item for item in goals if item)
        # A review/research request is read-only unless it also asks for a
        # change.  This keeps documentation/research delegation available.
        return bool(_CODING_INTENT_RE.search(text) and not (
            _RESEARCH_RE.search(text) and not re.search(r"\b(?:implement|edit|write|fix|change|patch)\b", text, re.I)
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


def _canonical_worker_task(task_id: str):
    from hermes_cli import kanban_db

    conn = kanban_db.connect()
    try:
        task = kanban_db.get_task(conn, task_id)
    finally:
        conn.close()
    if task is None:
        return None, "task does not exist"
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    if task.assignee != CANONICAL_ASSIGNEE:
        return task, "task is not assigned to DEV"
    if task.status not in ACTIVE_TASK_STATUSES:
        return task, f"task status is {task.status!r}"
    if metadata.get("canonical") is not True or metadata.get("lane") != "DEV":
        return task, "task lacks canonical DEV metadata"
    if metadata.get("coding_agent") not in SUPPORTED_CODING_AGENTS:
        return task, "task has no supported coding-agent metadata"
    origin = metadata.get("origin")
    if not isinstance(origin, dict) or not _text(origin.get("session_id")):
        return task, "task lacks origin linkage"
    return task, None


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
            task, reason = _canonical_worker_task(worker_id)
        except Exception:
            return _worker_refusal(tool_name, worker_id, "Kanban is unavailable")
        if reason:
            return _worker_refusal(tool_name, worker_id, reason)
        worker_agent = _text(os.environ.get("HERMES_CODING_AGENT"))
        if worker_agent and worker_agent.casefold() not in SUPPORTED_CODING_AGENTS:
            return _worker_refusal(tool_name, worker_id, "the worker provider is not supported")
        return None

    normalized_session = _text(session_id)
    if not normalized_session or not is_coding_intent(tool_name, args, user_message):
        return None

    agent, agent_error = _requested_coding_agent(args, user_message)
    if agent_error:
        return _refusal(error=agent_error, tool_name=tool_name, error_type="unsupported_coding_agent")

    workspace = _workspace(args)
    repository = _repository(workspace, args)
    scope = _text(args.get("scope") or args.get("goal") or args.get("context") or user_message)
    if not scope:
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
    if task is None or task.assignee != CANONICAL_ASSIGNEE or not isinstance(task.metadata, dict):
        return _refusal(
            error="Kanban did not return a canonical DEV task; coding work is blocked.",
            tool_name=tool_name,
            error_type="kanban_task_invalid",
        )
    return _refusal(
        error="Coding work was handed off to the DEV Kanban worker. The originating chat remains read-only for this change.",
        tool_name=tool_name,
        task_id=task.id,
        status=task.status,
        assignee=task.assignee,
        coding_agent=task.metadata.get("coding_agent"),
        reused=existing_id is not None,
    )
