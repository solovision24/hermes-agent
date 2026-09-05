# Kanban Telegram combined-candidate preservation manifest

Task: t_5c0cf317
Repository: solovision24/hermes-agent
Implementation PR: #60 (same branch; no second PR)

## Candidate lineage

This isolated candidate is built from the PR implementation head. The protected runtime references are comparison baselines, not claimed ancestors of this tree.

- Protected Codex-pool candidate: `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime baseline: `14208874aebbcbd97f900e967c226e45b154370d`
- Candidate tree SHA: `22b21c351468b60267e46981791b686522cf91c7`
- Isolated worktree: `/home/solo/.hermes/kanban/workspaces/t_5c0cf317/repo`

The candidate-vs-installed comparison was executed in this worktree with:

```text
git merge-base 22b21c351468b60267e46981791b686522cf91c7 14208874aebbcbd97f900e967c226e45b154370d
=> 9076adaca58b3c3b1839c42228f97410ddb43722

git hash-object gateway/run.py gateway/kanban_watchers.py tools/operational_sender.py cron/scheduler.py
=> 70a5cf09a2508a2660c5f8cb8ad33763cc4a3d15
=> a7b7dbf9a25a6792423872436b699771a3dace48
=> 1f9d9ed80a45acc9371d1df78c9b2dcc45582652
=> 7dc81ae2501d63854ed0d63cdff7e8439366a3d3
```

The merge-base and blob checks are evidence for release reconciliation, not ancestry proof for the protected candidate. Candidate symbols retain gateway runner wiring, `_kanban_dispatcher_watcher`, `_kanban_notifier_watcher`, `_kanban_advance`, `_kanban_rewind`, and the native Review wake paths; `cron/scheduler.py` remains outside this PR's diff. This is a local integration artifact only; the shared installed checkout and release owner were not changed.

## Executed preservation checks

From this isolated tree:

- `python -m pytest tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py tests/gateway/test_kanban_watchers_mixin.py tests/hermes_cli/test_kanban_notify.py::test_notifier_artifact_delivery_skips_missing_files tests/hermes_cli/test_kanban_db.py::test_notifier_behavior_on_blocked_recovery_exhausted -q`
- `python -m pytest tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py -q` — 25 passed, including real-loop response-thread rejection and multipart partial-batch rewind/retry/dedup.
- `python -m py_compile gateway/kanban_watchers.py tools/operational_sender.py`
- `git diff --check`
- `python -m pytest tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_assignee_resolution.py tests/hermes_cli/test_kanban_project_link.py -q` (native Kanban/Review and persistence coverage)

The full command output is retained in the Kanban run handoff; this manifest records the exact candidate lineage and boundary so release coordination can re-run it without treating comparison refs as a released upstream claim.

## Release boundary

Do not install or restart from this local candidate until Orion accepts PR #60 and the paired release owner `t_7d89c01a` performs the supported external release. Post-merge acceptance still requires gateway health plus one bounded real watcher delivery proving bot ID `8611668567`, username `solo_hermes_bot`, message ID, chat `8148316720`, and no `message_thread_id`.
