# Kanban Telegram isolation — preservation evidence

## Summary

This candidate is based on the canonical SoLo fork operational-watcher PR branch, not NousResearch upstream. The implementation preserves the protected paired-release candidate and the installed runtime as comparison references; no shared runtime checkout, service, or gateway process was changed.

## Immutable references

- Public PR focused head before this publication repair: `53c74704ceee540a35f0e9a1b0526d240556cca7`
- PR branch base at verification: `6e7f38c3f3e70066d16fa35cd8c716b49c043c6d`
- Fork `origin/main` at verification: `6e7f38c3f3e70066d16fa35cd8c716b49c043c6d`
- Protected paired-release candidate (not replaced): `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime reference (not replaced): `14208874aebbcbd97f900e967c226e45b154370d`
- Working candidate SHA: `4f7e919bfcab9da222a3f8dc76585a1fb2f436f3` (isolated combined candidate; preserved separately)
- Working candidate tree: `49670c58fd87a484d9e15f8fe01ad6b7d9dd5ecd`
- Working candidate path: `/tmp/kanban-integrate`

## Change scope

- `gateway/kanban_watchers.py`: Telegram completion-artifact failures now propagate after the batch, causing the existing notifier failure path to rewind the claimed cursor. Partial uploads are deliberately retried rather than falsely marked delivered.
- `tests/test_kanban_operational_sender.py`: symlink-to-system-path denial and partial artifact failure regression coverage.
- `tests/gateway/test_kanban_notifier.py`: the real watcher matrix now records every Halo adapter send attempt before raising; missing-token, wrong-identity, transport retry, response-thread rejection, review delivery, and multipart retry cases assert zero Halo attempts after each tick. Done→reopen status routing and archive cleanup assertions remain covered.
- Existing PR changes retain exact operational bot identity/destination/no-topic proof, real notifier-loop sender isolation, local-path validation, and Review lifecycle wake behavior.

## Verification

- Combined candidate verification: runtime import `RUNTIME_IMPORT_OK`; provider-boundary suite 9 passed; prescribed preservation suite 143 passed in 87.64s (152 total, in-process imports from the committed isolated candidate).
- Original protected auth/resolver control: `tests/agent/test_credential_pool_round9.py tests/agent/test_credential_pool_provider_boundary.py` — 12 passed in 1.10s against protected candidate `6a4de6ad6747171b9146deac68f15aabe865cd77`.
- Multipart retry/dedup retains its adapter and asserts zero Halo outbound attempts on the dedup tick; the discard-adapter mutation now fails.
- The installed legacy notifier suite remains incompatible with the dedicated Telegram sender unless its old adapter-send fixtures are migrated to the transport boundary; its run was recorded as 50 passed, 8 failed, 1 skipped (failures are fixture expectations, not accepted as product proof).
- `py_compile` for changed Python modules: passed.
- `git diff --check`: passed.
- Combined candidate checkout: clean at `4f7e919bfcab9da222a3f8dc76585a1fb2f436f3`; its files were not pushed as PR history.
- Focused public checkout: clean at `53c74704ceee540a35f0e9a1b0526d240556cca7` before this manifest-only commit.
- This manifest commit is separate from the tested candidate commit; PR #60 remains the public implementation artifact.
- Shared runtime: intentionally not switched, installed, or restarted; release authority remains the paired-release owner task.

## Release boundary

Orion must review/merge this same PR, then the release owner must perform preservation-safe installation and supported external restart. Final production proof must include health, a real watcher lifecycle notification, exact sender bot, canonical chat, message ID, and absence of `message_thread_id`.
