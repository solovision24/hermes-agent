"""Real disposable-board guard path; only GitHub and attended terminal are fixtures."""
import copy
import argparse
import json
import os
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb, kanban_recovery as recovery
from hermes_cli import kanban_reconcile as rec
from hermes_cli import kanban as cli
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_db_connect import connect
from hermes_cli.kanban_db_dispatch import check_respawn_guard

PR = 'https://github.com/acme/repo/pull/194'
HEAD = 'a' * 40


@pytest.fixture
def board(tmp_path, monkeypatch):
    import pwd
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_TASK', raising=False)
    monkeypatch.delenv('HERMES_KANBAN_RUN_ID', raising=False)
    monkeypatch.setattr(rec.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    kb.init_db()
    pr = dict(html_url=PR, state='open', merged=False,
              head=dict(sha=HEAD, ref='fix/existing', repo=dict(full_name='acme/repo')))
    monkeypatch.setattr(recovery, '_api', lambda *a, **k: copy.deepcopy(pr))
    with connect() as conn:
        task = kb.create_task(conn, title='Remediate existing PR', body='Keep original full spec',
                              assignee='forge', completion_contract=PR)
        kb._synthesize_ended_run(conn, task, outcome='blocked', metadata={
            'published_pr': PR, 'head_sha': HEAD, 'branch': 'fix/existing'})
        kb.add_comment(conn, task, 'forge', 'Existing work: ' + PR)
        opts = dict(operator=pwd.getpwuid(os.geteuid()).pw_name, confirm=task,
                    pr_url=PR, head_sha=HEAD, branch='fix/existing',
                    head_repository='acme/repo', assignee='forge', reason='Remediate same existing PR')
        yield conn, task, opts, pr


@pytest.mark.platforms('posix')
def test_same_pr_needs_explicit_bound_route_and_fresh_comment_does_not_retrap(board):
    conn, task, opts, _ = board
    before = dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone())
    assert check_respawn_guard(conn, task) == 'active_pr'
    recovery.authorize_pr_remediation(conn, task, **opts)
    kb.add_comment(conn, task, 'forge', 'More exact same PR evidence: ' + PR)
    assert check_respawn_guard(conn, task) is None
    assert dict(conn.execute('SELECT * FROM tasks WHERE id=?', (task,)).fetchone()) == before


@pytest.mark.platforms('posix')
@pytest.mark.parametrize('fault', ['owner', 'branch', 'head', 'pr', 'claim', 'open-run', 'new-run', 'spec', 'foreign-comment', 'remote-head', 'remote-owner', 'merged'])
def test_identity_or_worker_conflicts_stay_guarded(board, fault):
    conn, task, opts, pr = board
    if fault in {'owner', 'branch', 'head', 'pr', 'claim', 'open-run'}:
        if fault == 'owner': opts['assignee'] = 'chip'
        if fault == 'branch': opts['branch'] = 'fix/other'
        if fault == 'head': opts['head_sha'] = 'b' * 40
        if fault == 'pr': opts['pr_url'] = PR.replace('194', '195')
        if fault == 'claim': conn.execute("UPDATE tasks SET worker_pid=123 WHERE id=?", (task,))
        if fault == 'open-run': conn.execute('UPDATE task_runs SET ended_at=NULL WHERE task_id=?', (task,))
        with pytest.raises(ValueError):
            recovery.authorize_pr_remediation(conn, task, **opts)
    else:
        recovery.authorize_pr_remediation(conn, task, **opts)
        if fault == 'new-run': kb._synthesize_ended_run(conn, task, outcome='crashed')
        if fault == 'spec': kb.edit_task(conn, task, body='changed')
        if fault == 'foreign-comment': kb.add_comment(conn, task, 'forge', PR.replace('194', '195'))
        if fault == 'remote-head': pr['head']['sha'] = 'b' * 40
        if fault == 'remote-owner': pr['head']['repo']['full_name'] = 'foreign/repo'
        if fault == 'merged': pr.update(state='closed', merged=True)
    assert check_respawn_guard(conn, task) == 'active_pr'


@pytest.mark.platforms('posix')
def test_worker_cannot_authorize_and_failed_audit_rolls_back(board, monkeypatch):
    conn, task, opts, _ = board
    monkeypatch.setenv('HERMES_KANBAN_TASK', task)
    with pytest.raises(PermissionError):
        recovery.authorize_pr_remediation(conn, task, **opts)
    monkeypatch.delenv('HERMES_KANBAN_TASK')
    monkeypatch.setattr(kb, '_append_event', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('disk')))
    with pytest.raises(RuntimeError):
        recovery.authorize_pr_remediation(conn, task, **opts)
    assert check_respawn_guard(conn, task) == 'active_pr'


@pytest.mark.platforms('posix')
def test_supported_cli_does_not_authorize_a_sibling_build(board, capsys):
    conn, task, opts, _ = board
    parser = build_parser(argparse.ArgumentParser().add_subparsers(dest='top'))
    argv = ['authorize-pr-remediation', task]
    for key, value in opts.items():
        argv += ['--' + key.replace('_', '-'), value]
    assert cli.kanban_command(parser.parse_args(argv)) == 0
    assert json.loads(capsys.readouterr().out)['task_id'] == task
    assert check_respawn_guard(conn, task) is None
    sibling = kb.create_task(conn, title='Duplicate build', assignee='forge', completion_contract=PR)
    kb.add_comment(conn, sibling, 'forge', PR)
    assert check_respawn_guard(conn, sibling) == 'active_pr'
