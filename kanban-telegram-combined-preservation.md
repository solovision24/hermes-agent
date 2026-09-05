# Isolated Kanban Telegram Combined Candidate

## Summary

This artifact records the isolated candidate prepared for PR #60. It is not an installation authorization and does not alter the shared Hermes runtime.

## Candidate identity

- Repository: `solovision24/hermes-agent`
- Candidate commit: `89a1275460e07f48b5e5a38d28cad3cd494551a2`
- Branch: `agent/forge/kanban-operational-watcher`
- PR: https://github.com/solovision24/hermes-agent/pull/60
- Protected paired-release reference: `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime reference: `14208874aebbcbd97f900e967c226e45b154370d`

## Preservation contract

The candidate retains the PR #60 operational sender and watcher changes, while the release owner must reconcile the protected paired-release and installed-runtime changes before installation. The candidate has not been installed, selected as the live checkout, or restarted.

Required release-owner checks before activation:

1. Compare `gateway/`, `cron/`, and auth/runtime resolver changes against both protected references.
2. Preserve installed sender, cron, and native Review behavior while applying this candidate's watcher isolation.
3. Run the focused notifier/sender suite and the supported health checks from the release procedure.
4. Perform the external restart only after the combined checkout is reviewed and verified.

## Verification performed in isolation

- `python -m pytest tests/gateway/test_kanban_notifier.py tests/test_kanban_operational_sender.py tests/gateway/test_kanban_watchers_mixin.py tests/hermes_cli/test_kanban_notify.py::test_notifier_artifact_delivery_skips_missing_files tests/hermes_cli/test_kanban_db.py::test_notifier_behavior_on_blocked_recovery_exhausted -q` — 21 passed.
- `python -m py_compile gateway/kanban_watchers.py tools/operational_sender.py tests/test_kanban_operational_sender.py` — passed.
- `git diff --check` — passed.
- `git ls-remote origin refs/heads/agent/forge/kanban-operational-watcher` — returned candidate SHA above.

## Delivery semantics covered

The test fixtures use a fake Telegram transport at `_api_call`, not Halo adapter sends. They provide verified bot identity and delivery proof, preserve no-thread requests, and record operational text/artifact deliveries. Missing token or transport failure remains a notifier failure; the notifier rewinds before retry and drops only after its bounded 12-failure policy. Partial artifact batches may replay already-uploaded files after rewind by design.

## Ownership boundary

Orion/release owner `t_7d89c01a` owns merge, combined-runtime reconciliation, installation, supported restart, health, and bounded real watcher proof. This file is evidence for that handoff only; no shared runtime mutation was performed.
