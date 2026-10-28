"""Attended owner attestation of executed local tests; exact PR contract stays intact.

No completion, promotion, CI publication or release is performed here. The owner
selects the required focused command plan; stdout is hashed, never persisted as
untrusted prose or treated as hosted CI. This is the same OS-owner trust boundary
as reconcile-pr-contract, not a worker-supplied green receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_reconcile as rec
from hermes_cli.kanban_native_review import native_review_snapshot
from hermes_cli.kanban_recovery import contract_version, require_idle

POLICY = "private-merged-exact-contract-local-tests-v1"


def _git(cwd, *args):
    return subprocess.check_output(['git', '-C', str(cwd), *args],
        stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, timeout=30).strip()


def _checkout(plan: dict, repo: str, head_sha: str, merge_sha: str):
    if not isinstance(plan, dict) or set(plan) != {'cwd', 'sha', 'commands'}:
        raise ValueError("test plan requires exactly cwd, sha and commands")
    cwd = Path(plan['cwd']).resolve(strict=True)
    if not cwd.is_dir() or plan['sha'] not in {head_sha, merge_sha}:
        raise ValueError("tests must run at the exact reviewed head or merge")
    remote = _git(cwd, 'config', '--get', 'remote.origin.url')
    match = re.fullmatch(r'(?:https://github\.com/|git@github\.com:)([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?', remote)
    if not match or match[1].casefold() != repo.casefold():
        raise ValueError("test checkout origin must match the exact contract repository")
    if _git(cwd, 'rev-parse', 'HEAD') != plan['sha'] or _git(cwd, 'status', '--porcelain'):
        raise ValueError("test checkout is dirty or at a stale SHA")
    commands = plan['commands']
    if (not isinstance(commands, list) or not 1 <= len(commands) <= 20
            or any(not isinstance(argv, list) or not argv or len(argv) > 100
                   or any(not isinstance(a, str) or not a or '\x00' in a for a in argv)
                   for argv in commands)):
        raise ValueError("owner must supply 1..20 required test argv arrays; shell strings are forbidden")
    return cwd


def execute_test_plan(plan: dict, repo: str, head_sha: str, merge_sha: str) -> dict:
    cwd = _checkout(plan, repo, head_sha, merge_sha)
    results = []
    for command in plan['commands']:
        started = int(time.time())
        # A real process exit is authoritative; caller-provided 'passed' is not.
        # Spool bounded-by-time output to disk instead of retaining it in memory.
        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=subprocess.STDOUT, timeout=600, check=False)
            except (OSError, subprocess.TimeoutExpired):
                raise ValueError("required local test could not execute or timed out") from None
            output.seek(0)
            digest = hashlib.file_digest(output, 'sha256').hexdigest()
        results.append(dict(command=command, exit_code=result.returncode,
                            started_at=started, ended_at=int(time.time()), output_sha256=digest))
        if result.returncode != 0:
            raise ValueError("required local test failed; no acceptance recorded")
    _checkout(plan, repo, head_sha, merge_sha)
    return dict(tested_sha=plan['sha'], plan_sha256=hashlib.sha256(
        json.dumps(plan, sort_keys=True).encode()).hexdigest(), results=results)


def _native(conn, task_id, review_task, review_run, pr_url, head_sha, merge_sha):
    if bool(review_task) != (review_run is not None):
        raise ValueError("native review task/run must be supplied together")
    return (native_review_snapshot(conn, task_id, review_task, review_run, pr_url, head_sha, merge_sha)
            if review_task is not None else None)


def reconcile_local_acceptance(conn, task_id: str, *, operator: str, confirm: str,
        expected_contract_version: str, pr_url: str, head_sha: str, merge_sha: str,
        review_url: str, reason: str, test_plan: dict,
        native_review_task: str | None = None, native_review_run: int | None = None) -> dict:
    actor = rec.authorize_local_owner(conn, operator, task_id, confirm)
    reason = rec._required(reason, 'reason')
    before = dict(require_idle(conn, task_id))
    if before['completion_contract'] != pr_url or not rec._PR.fullmatch(pr_url):
        raise ValueError("reconciliation requires the persisted exact PR contract")
    version = contract_version(conn, task_id)
    if expected_contract_version != version:
        raise ValueError("stale task/contract version")
    native = _native(conn, task_id, native_review_task, native_review_run, pr_url, head_sha, merge_sha)
    evidence = rec.collect_merged_pr_evidence(pr_url, head_sha, merge_sha, review_url, native)
    match = rec._PR.fullmatch(pr_url)
    assert match is not None
    repo = match[1]
    try:
        tests = execute_test_plan(test_plan, repo, head_sha, merge_sha)
    except ValueError:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, 'pr_local_acceptance_failed', {
                'actor': actor, 'contract_version': version, 'pr_url': pr_url,
                'head_sha': head_sha, 'merge_sha': merge_sha,
                'detail': 'required test execution/checkout failed; no acceptance'})
        raise
    # Revalidate mutable GitHub evidence AFTER the actual subprocesses, not only
    # before them. Native provenance and the full card snapshot recheck under lock.
    current_evidence = rec.collect_merged_pr_evidence(pr_url, head_sha, merge_sha, review_url, native)
    if current_evidence != evidence:
        raise ValueError("PR/review evidence changed while tests ran")
    payload = dict(policy=POLICY, task_id=task_id, actor=actor, reason=reason,
        contract_version=version, original_contract=pr_url, replacement_contract=pr_url,
        hosted_ci_accepted=False, release_authorized=False, evidence=evidence, tests=tests,
        native_review_task=native_review_task, native_review_run=native_review_run)
    with kb.write_txn(conn):
        if dict(require_idle(conn, task_id)) != before or contract_version(conn, task_id) != version:
            raise ValueError("task changed during acceptance collection")
        if _native(conn, task_id, native_review_task, native_review_run, pr_url, head_sha, merge_sha) != native:
            raise ValueError("independent review provenance changed")
        kb._append_event(conn, task_id, 'pr_local_acceptance_reconciled', payload)
    return payload


def local_acceptance(conn, task_id: str, published_pr: str | None):
    """Revalidate only explicit owner receipts; never fall back on a generic 403."""
    row = conn.execute("SELECT kind,payload FROM task_events WHERE task_id=? AND kind IN "
        "('pr_local_acceptance_reconciled','pr_local_acceptance_failed') ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if row is None:
        return None
    denied = dict(ok=False, classification='stale', detail='Owner local acceptance is stale or revoked.',
                  recovery='Ask the attended owner to revalidate the exact contract and required tests.')
    if row['kind'] != 'pr_local_acceptance_reconciled':
        return denied
    try:
        payload = json.loads(row['payload'])
        evidence = payload['evidence']
        if (payload['policy'] != POLICY or payload['task_id'] != task_id
                or payload['contract_version'] != contract_version(conn, task_id)
                or (published_pr is not None and published_pr != evidence['pr_url'])):
            return denied
        native = _native(conn, task_id, payload['native_review_task'], payload['native_review_run'],
                         evidence['pr_url'], evidence['head_sha'], evidence['merge_sha'])
        if rec.collect_merged_pr_evidence(evidence['pr_url'], evidence['head_sha'], evidence['merge_sha'],
                                         evidence['review_url'], native) != evidence:
            return denied
        return dict(ok=True, classification='owner_local_tests', pr_url=evidence['pr_url'],
                    head_sha=evidence['head_sha'], merge_sha=evidence['merge_sha'],
                    checks=[], hosted_ci_accepted=False, release_authorized=False,
                    contract_version=payload['contract_version'], local_tests=payload['tests'])
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError):
        return denied


def receipt_still_bound(conn, task_id: str, receipt: dict) -> bool:
    """Inside the terminal transaction: no network under the write lock."""
    if receipt.get('classification') != 'owner_local_tests':
        return True
    row = conn.execute("SELECT kind,payload FROM task_events WHERE task_id=? AND kind IN "
        "('pr_local_acceptance_reconciled','pr_local_acceptance_failed') ORDER BY id DESC LIMIT 1", (task_id,)).fetchone()
    if row is None or row['kind'] != 'pr_local_acceptance_reconciled':
        return False
    payload = json.loads(row['payload'])
    evidence = payload['evidence']
    if (receipt['contract_version'] != contract_version(conn, task_id)
            or payload['tests'] != receipt['local_tests']
            or evidence['pr_url'] != receipt['pr_url']):
        return False
    if payload['native_review_task'] is not None:
        try:
            native = _native(conn, task_id, payload['native_review_task'], payload['native_review_run'],
                             evidence['pr_url'], evidence['head_sha'], evidence['merge_sha'])
        except ValueError:
            return False
        recorded = dict(evidence['review'])
        recorded.pop('corroboration', None)
        return native == recorded
    return True


def cmd_reconcile_local_acceptance(args) -> int:
    from hermes_cli import kanban_db_connect as kbc
    path = Path(args.test_plan)
    if path.stat().st_size > 64000:
        raise ValueError('test plan exceeds 64000 bytes')
    plan = json.loads(path.read_text(encoding='utf-8'))
    with kbc.connect_closing() as conn:
        payload = reconcile_local_acceptance(conn, args.task_id, operator=args.operator,
            confirm=args.confirm, expected_contract_version=args.contract_version,
            pr_url=args.pr_url, head_sha=args.head_sha, merge_sha=args.merge_sha,
            review_url=args.review_url, reason=args.reason, test_plan=plan,
            native_review_task=args.native_review_task, native_review_run=args.native_review_run)
    print(json.dumps(payload))
    return 0
