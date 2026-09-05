# Kanban Telegram combined-runtime preservation manifest

Task: t_5c0cf317
Repository: solovision24/hermes-agent
Implementation PR: #60 (same branch; no second PR)

## Candidate lineage

This is an isolated local integration candidate. It starts from installed fork baseline `14208874aebbcbd97f900e967c226e45b154370d` and narrowly layers the accepted operational sender/watcher commits, while retaining installed watcher behavior (delivery modes, subscription retention/GC, wake routing, review handoff rendering, and fresh-context offloads).

- Protected Codex candidate reference: `6a4de6ad6747171b9146deac68f15aabe865cd77` (comparison reference only; not claimed ancestry)
- Installed baseline: `14208874aebbcbd97f900e967c226e45b154370d`
- Isolated candidate commit: `b5dcf2261642056a31daf3b65fa46c79cbb5a4ca`
- Isolated candidate tree: `5ec4d08e1b31498a0422e05a972412856d814bd7`
- Isolated worktree: `/tmp/kanban-integrate`
- Candidate checkout is clean; shared installed runtime was not modified, installed, or restarted.

The candidate retains the installed `gateway/kanban_watchers.py` control flow and integrates the dedicated Telegram sender at the text/artifact transport boundary. Telegram operational delivery remains identity-verified and canonical; Halo adapter sends are not used for operational pings. Creator wake/review dispatch remains adapter-based and preserves tenant scope, review handoff detail, and no-topic operational delivery.

## Exact verification

From the isolated candidate:

- `python -m pytest tests/agent/test_credential_pool_round9.py tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py tests/cron/test_cron_kanban_env_isolation.py tests/hermes_cli/test_kanban_review_lifecycle.py tests/hermes_cli/test_kanban_review_lifecycle_complete.py tests/gateway/test_kanban_watchers_mixin.py tests/gateway/test_platform_reconnect.py tests/gateway/test_kanban_wake_scope.py -q` — 134 passed.
- `python -m py_compile agent/credential_pool.py hermes_cli/auth.py hermes_cli/runtime_provider.py gateway/kanban_watchers.py tools/operational_sender.py` — passed.
- `git diff --check` — passed.
- Protected credential-pool omission and concurrent-login tests pass against the candidate: 3 passed.
- Real notifier-loop tests use a disposable board, mock only the operational transport boundary, forbid Halo operational sends, verify review/changes-requested pings and creator wakes, canonical chat/no topic, failure rewind/retry, partial artifact behavior, and preserved delivery-mode semantics. Restored lifecycle coverage verifies done/reopen/archive retention, both wake-worthy review/triage handoffs, and notify-only no-wake behavior. The unscoped Telegram wake-key regression uses the same transport-boundary fixture and preserves the byte-identical session-key assertion.

## Release boundary

Do not install or restart this local candidate until Orion accepts PR #60 and paired release owner `t_7d89c01a` performs the supported external release. Post-merge acceptance still requires gateway health plus one bounded real watcher delivery proving bot ID `8611668567`, username `solo_hermes_bot`, message ID, chat `8148316720`, and absent `message_thread_id`.
