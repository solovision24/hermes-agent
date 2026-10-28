"""Disposable git/board + real test processes, GitHub transport fixtures only."""
import argparse
import copy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from hermes_cli import kanban as cli, kanban_db as kb
from hermes_cli import kanban_local_acceptance as local, kanban_reconcile as rec
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_recovery import contract_version
from hermes_cli.kanban_pr_acceptance_store import prepare_acceptance, record_acceptance

PR = 'https://github.com/acme/repo/pull/65'
REVIEW = PR + '#pullrequestreview-12'


@pytest.fixture
def board(tmp_path, monkeypatch):
    import pwd
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_RUN_ID', raising=False)
    monkeypatch.setattr(rec.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    cwd = tmp_path / 'checkout'
    cwd.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(cwd), *args], text=True, stderr=subprocess.DEVNULL).strip()
    git('init')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Fixture')
    git('remote', 'add', 'origin', 'https://github.com/acme/repo.git')
    (cwd / 'contract.txt').write_text('immutable data')
    git('add', 'contract.txt')
    git('commit', '-m', 'disposable regression fixture')
    head = git('rev-parse', 'HEAD')
    git('commit', '--allow-empty', '-m', 'merge fixture')
    merge = git('rev-parse', 'HEAD')
    plan = dict(cwd=str(cwd), sha=merge, commands=[[sys.executable, '-c',
        "from pathlib import Path; assert Path('contract.txt').read_text() == 'immutable data'; print('fixture assertion passed')"]])
    pr = dict(html_url=PR, head=dict(sha=head), merge_commit_sha=merge, merged=True,
              merged_at='2026-10-27T20:00:00Z', state='closed',
              user=dict(id=1, login='author', type='User'),
              base=dict(ref='main', repo=dict(full_name='acme/repo')))
    review = dict(id=12, html_url=REVIEW, pull_request_url='https://api.github.com/repos/acme/repo/pulls/65',
                  commit_id=head, state='APPROVED', submitted_at='2026-10-27T19:00:00Z', body='approved',
                  user=dict(id=2, login='reviewer', type='User'))
    remote = dict(checks=0, statuses=[], private=True, admin=True)
    def api(endpoint, **kwargs):
        if endpoint == 'repos/acme/repo/pulls/65': return copy.deepcopy(pr)
        if endpoint == 'repos/acme/repo/pulls/65/reviews/12': return copy.deepcopy(review)
        if endpoint == 'repos/acme/repo/pulls/65/reviews?per_page=100&page=1': return [copy.deepcopy(review)]
        if endpoint == 'repos/acme/repo': return dict(full_name='acme/repo', private=remote['private'], permissions=dict(admin=remote['admin']))
        if endpoint.endswith('check-runs?per_page=100'): return dict(total_count=remote['checks'])
        if endpoint.endswith('statuses?per_page=100'): return remote['statuses']
        raise AssertionError(endpoint)
    monkeypatch.setattr(rec, '_api', api)
    monkeypatch.setattr(rec, '_plan_limitation', lambda endpoint: dict(endpoint=endpoint, status=403, message=rec._PLAN_MESSAGE))
    kb.init_db()
    with connect() as conn:
        task = kb.create_task(conn, title='Merged private artifact', body='Keep full required tests and release gates', assignee='chip', completion_contract=PR)
        child = kb.create_task(conn, title='Independent release', parents=[task], assignee='orion')
        opts = dict(operator=pwd.getpwuid(os.geteuid()).pw_name, confirm=task,
                    expected_contract_version=contract_version(conn, task), pr_url=PR,
                    head_sha=head, merge_sha=merge, review_url=REVIEW,
                    reason='Owner accepts executed exact-merge local plan, never hosted CI', test_plan=plan)
        yield conn, task, child, opts, pr, review, remote, git


@pytest.mark.platforms('posix')
def test_cli_executes_real_plan_preserves_exact_contract_and_does_not_release(board, tmp_path, capsys):
    conn, task, child, opts, _, _, _, _ = board
    before = dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone())
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(opts['test_plan']))
    parser = build_parser(argparse.ArgumentParser().add_subparsers(dest='top'))
    argv = ['reconcile-local-acceptance', task, '--test-plan', str(plan_path)]
    for key, value in opts.items():
        if key == 'test_plan': continue
        flag = 'contract-version' if key == 'expected_contract_version' else key.replace('_', '-')
        argv += ['--' + flag, str(value)]
    assert cli.kanban_command(parser.parse_args(argv)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['hosted_ci_accepted'] is False
    assert payload['release_authorized'] is False
    assert payload['tests']['results'][0]['exit_code'] == 0
    assert len(payload['tests']['results'][0]['output_sha256']) == 64
    assert dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone()) == before
    assert kb.get_task(conn, child).status == 'todo'
    assert kb.complete_task(conn, task, summary='Reviewed source only; downstream release remains independent', metadata={'published_pr': PR})
    assert kb.get_task(conn, task).completion_contract == PR
    assert kb.get_task(conn, child).status == 'ready'  # normal completion only, not reconciliation
    receipt = [e.payload for e in kb.list_events(conn, task) if e.kind == 'pr_acceptance'][-1]
    assert receipt['classification'] == 'owner_local_tests' and receipt['checks'] == []


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('fault', ['version', 'failed-test', 'stale-checkout', 'dirty-checkout', 'wrong-origin', 'empty-plan', 'shell-plan', 'head', 'merge', 'review-head', 'self-review', 'negative-review', 'hosted-check', 'hosted-status', 'public', 'not-admin', 'worker', 'live-claim'])
def test_invalid_identity_review_tests_and_authority_never_record_acceptance(board, fault, monkeypatch):
    conn, task, child, opts, pr, review, remote, git = board
    if fault == 'version': opts['expected_contract_version'] = '0' * 64
    if fault == 'failed-test': opts['test_plan']['commands'].append([sys.executable, '-c', 'raise AssertionError("required negative result")'])
    if fault == 'stale-checkout': opts['test_plan']['sha'] = opts['head_sha']
    if fault == 'dirty-checkout': (local.Path(opts['test_plan']['cwd']) / 'contract.txt').write_text('changed')
    if fault == 'wrong-origin': git('remote', 'set-url', 'origin', 'https://github.com/foreign/repo.git')
    if fault == 'empty-plan': opts['test_plan']['commands'] = []
    if fault == 'shell-plan': opts['test_plan']['commands'] = ['echo passed']
    if fault == 'head': pr['head']['sha'] = 'c' * 40
    if fault == 'merge': pr['merge_commit_sha'] = 'c' * 40
    if fault == 'review-head': review['commit_id'] = 'c' * 40
    if fault == 'self-review': review['user']['id'] = 1
    if fault == 'negative-review': review['state'] = 'CHANGES_REQUESTED'
    if fault == 'hosted-check': remote['checks'] = 1
    if fault == 'hosted-status': remote['statuses'] = [dict(state='failure')]
    if fault == 'public': remote['private'] = False
    if fault == 'not-admin': remote['admin'] = False
    if fault == 'worker': monkeypatch.setenv('HERMES_KANBAN_TASK', task)
    if fault == 'live-claim': conn.execute('UPDATE tasks SET worker_pid=123 WHERE id=?', (task,))
    with pytest.raises((ValueError, PermissionError)):
        local.reconcile_local_acceptance(conn, task, **opts)
    assert not [e for e in kb.list_events(conn, task) if e.kind == 'pr_local_acceptance_reconciled']
    assert kb.get_task(conn, task).completion_contract == PR
    assert kb.get_task(conn, child).status == 'todo'


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('fault', ['spec', 'edit-away-back', 'owner', 'contract', 'changed-head', 'revoked-review', 'failed-retry', 'concurrent-record', 'later-run'])
def test_old_success_cannot_outlive_version_review_or_required_failure(board, fault):
    conn, task, child, opts, pr, review, _, _ = board
    local.reconcile_local_acceptance(conn, task, **opts)
    if fault == 'spec': kb.edit_task(conn, task, body='new acceptance')
    if fault == 'edit-away-back':
        original = kb.get_task(conn, task).body
        kb.edit_task(conn, task, body='temporary change')
        kb.edit_task(conn, task, body=original)
    if fault == 'owner': kb.assign_task(conn, task, 'forge')
    if fault == 'contract': conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (PR.replace('65', '66'), task))
    if fault == 'changed-head': pr['head']['sha'] = 'd' * 40
    if fault == 'revoked-review': review['state'] = 'DISMISSED'
    if fault == 'later-run': kb._synthesize_ended_run(conn, task, outcome='blocked')
    if fault == 'failed-retry':
        opts['test_plan']['commands'] = [[sys.executable, '-c', 'assert False']]
        with pytest.raises(ValueError): local.reconcile_local_acceptance(conn, task, **opts)
    if fault == 'concurrent-record':
        prepared = prepare_acceptance(conn, task, None, {'published_pr': PR})
        kb.edit_task(conn, task, body='changed between prepare and terminal write')
        with kb.write_txn(conn): assert not record_acceptance(conn, task, prepared)
    else:
        assert not kb.complete_task(conn, task, summary='Must reject stale acceptance', metadata={'published_pr': PR})
    assert kb.get_task(conn, task).status == 'ready'
    assert kb.get_task(conn, child).status == 'todo'


@pytest.mark.platforms('posix')
def test_audit_failure_rolls_back_and_race_does_not_mutate_card(board, monkeypatch):
    conn, task, child, opts, _, _, _, _ = board
    monkeypatch.setattr(kb, '_append_event', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('disk')))
    with pytest.raises(RuntimeError): local.reconcile_local_acceptance(conn, task, **opts)
    assert kb.get_task(conn, task).completion_contract == PR
    assert kb.get_task(conn, task).status == 'ready'
    assert kb.get_task(conn, child).status == 'todo'
