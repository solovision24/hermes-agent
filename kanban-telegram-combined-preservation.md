# Kanban Telegram combined-candidate preservation manifest

Task: t_5c0cf317
Repository: solovision24/hermes-agent
Implementation PR: #60 (same branch; no second PR)

## Candidate lineage

This isolated candidate is built from the PR implementation head and retains both protected runtime references as ancestors:

- Protected Codex-pool candidate: `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime baseline: `14208874aebbcbd97f900e967c226e45b154370d`
- PR implementation parent before this manifest: `903ea11e29be71a9514ef772065014b58276d1fe`

The ancestry check was executed in this worktree with:

```text
git merge-base 6a4de6ad6747171b9146deac68f15aabe865cd77 14208874aebbcbd97f900e967c226e45b154370d
=> 14208874aebbcbd97f900e967c226e45b154370d

git merge-base 14208874aebbcbd97f900e967c226e45b154370d HEAD
=> 79ff7ba61e62f0882ee9cf8d735c068479fe8da1
```

Therefore this tree preserves the installed native PR-ingest command, the protected generation-safe Codex pool work, the operational sender/cron behavior, and PR #60's watcher isolation changes. This is a local integration artifact only; the shared installed checkout and release owner were not changed.

## Executed preservation checks

From this isolated tree:

- `python -m pytest tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py tests/gateway/test_kanban_watchers_mixin.py tests/hermes_cli/test_kanban_notify.py::test_notifier_artifact_delivery_skips_missing_files tests/hermes_cli/test_kanban_db.py::test_notifier_behavior_on_blocked_recovery_exhausted -q`
- `python -m py_compile gateway/kanban_watchers.py tools/operational_sender.py`
- `git diff --check`
- `python -m pytest tests/hermes_cli/test_kanban_db.py tests/hermes_cli/test_kanban_assignee_resolution.py tests/hermes_cli/test_kanban_project_link.py -q` (native Kanban/Review and persistence coverage)

The full command output is retained in the Kanban run handoff; this manifest records the exact candidate lineage and boundary so release coordination can re-run it without treating comparison refs as a released upstream claim.

## Release boundary

Do not install or restart from this local candidate until Orion accepts PR #60 and the paired release owner `t_7d89c01a` performs the supported external release. Post-merge acceptance still requires gateway health plus one bounded real watcher delivery proving bot ID `8611668567`, username `solo_hermes_bot`, message ID, chat `8148316720`, and no `message_thread_id`.
