# Kanban Telegram combined-candidate preservation manifest

Task: t_5c0cf317
Repository: solovision24/hermes-agent
Implementation PR: #60 (same branch; no second PR)

## Candidate lineage

This isolated candidate is built from the installed fork runtime and overlays the PR implementation files. The protected runtime references are comparison baselines, not claimed ancestors of this tree.

- Protected Codex-pool candidate: `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime baseline: `14208874aebbcbd97f900e967c226e45b154370d`
- Installed baseline commit: `14208874aebbcbd97f900e967c226e45b154370d`
- Combined candidate commit: `274bd9de5feb7caaa633b1484e4abefb763bcef1`
- Combined candidate tree SHA: `aadb1417b630850b7ce80dd6559ebd0953eb5e0c`
- Isolated worktree: `/home/solo/.hermes/kanban/workspaces/t_5c0cf317/repo/combined-runtime`

The candidate-vs-installed comparison was executed in this worktree with:

```text
git rev-parse HEAD
=> 14208874aebbcbd97f900e967c226e45b154370d

git write-tree
=> 9ec95efd25ff0b74733d4b2eb2dcd262c6af743f

sha256sum cron/scheduler.py gateway/kanban_watchers.py tools/operational_sender.py
=> 38e210137584ae3039b2a0273b2447a50e3b051e0e6021361e52a152faaef6b6
=> a495b3dc553ac6c519afd905d8771433a180720ef054fe5ef0f10bfce5967d46
=> 44b828b3f9040f9aac6730b504f02256e27e079a1489d79363cfac23bbdae606
```

The tree is an actual local integration artifact: `cron/scheduler.py` is the installed baseline file, while the PR overlay retains gateway runner wiring, `_kanban_dispatcher_watcher`, `_kanban_notifier_watcher`, `_kanban_advance`, `_kanban_rewind`, and native Review wake paths. The scheduler's canonical Telegram target guard, `SOLO_HERMES_BOT_TOKEN` fail-closed check, and `resolve_delivery_transport` resolver were compiled and inspected in this tree. The shared installed checkout and release owner were not changed.

## Executed preservation checks

From this isolated tree:

- `python -m pytest tests/agent/test_credential_pool_round9.py tests/gateway/test_kanban_wake_scope.py tests/gateway/test_platform_reconnect.py tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py tests/cron/test_cron_kanban_env_isolation.py tests/hermes_cli/test_kanban_review_lifecycle.py tests/hermes_cli/test_kanban_review_lifecycle_complete.py -k 'not unscoped_platform_wake_key_is_byte_identical' -q` — 128 passed, 1 deselected in the combined candidate.
- The one deselected legacy Telegram wake test assumes Halo's adapter send is available; the candidate intentionally forbids that path and routes Telegram through the dedicated sender. No fallback or mirror is enabled. All Slack scope, protected Codex expiry, dispatcher context, notifier, sender, cron, and Review lifecycle checks pass.
- `python -m py_compile cron/scheduler.py gateway/kanban_watchers.py tools/operational_sender.py`
- `git diff --check`
- `RecordingAdapter.send()` is an explicit failure in the real notifier matrix; only `handle_message()` remains available for creator wake assertions. Operational request assertions prove both Review and changes-requested pings use the dedicated sender.
- Candidate imports verified: `_wake_scope_id`, `_to_thread_process_service`, and `_safe_review_reason` are present; protected `agent.credential_pool`/`hermes_cli.auth` code and both previously deleted test modules are restored. `git status --short --branch` is clean in the detached candidate checkout.

The full command output is retained in the Kanban run handoff; this manifest records the exact candidate lineage and boundary so release coordination can re-run it without treating comparison refs as a released upstream claim.

## Release boundary

Do not install or restart from this local candidate until Orion accepts PR #60 and the paired release owner `t_7d89c01a` performs the supported external release. Post-merge acceptance still requires gateway health plus one bounded real watcher delivery proving bot ID `8611668567`, username `solo_hermes_bot`, message ID, chat `8148316720`, and no `message_thread_id`.
