# Kanban Telegram isolation — preservation evidence

## Summary

This candidate is based on the canonical SoLo fork operational-watcher PR branch, not NousResearch upstream. The implementation preserves the protected paired-release candidate and the installed runtime as comparison references; no shared runtime checkout, service, or gateway process was changed.

## Immutable references

- PR branch base before this remediation: `cefcb01cd98741d9859437dcdf86cbf48357b123`
- Fork `origin/main` at verification: `6e7f38c3f3e70066d16fa35cd8c716b49c043c6d`
- Protected paired-release candidate (not replaced): `6a4de6ad6747171b9146deac68f15aabe865cd77`
- Installed runtime reference (not replaced): `14208874aebbcbd97f900e967c226e45b154370d`
- Working candidate SHA: `a17bc0cda8738da4337a339ba8537dc1a6559e6f` (isolated combined candidate)
- Working candidate tree: `aa8172d38984418b670ef97e678bbcaf173f6a50`
- Working candidate path: `/tmp/kanban-integrate`

## Change scope

- `gateway/kanban_watchers.py`: Telegram completion-artifact failures now propagate after the batch, causing the existing notifier failure path to rewind the claimed cursor. Partial uploads are deliberately retried rather than falsely marked delivered.
- `tests/test_kanban_operational_sender.py`: symlink-to-system-path denial and partial artifact failure regression coverage.
- `tests/gateway/test_kanban_notifier.py`: the real watcher matrix now records every Halo adapter send attempt before raising; missing-token, wrong-identity, transport retry, response-thread rejection, review delivery, and multipart retry cases assert zero Halo attempts after each tick. Done→reopen status routing and archive cleanup assertions remain covered.
- Existing PR changes retain exact operational bot identity/destination/no-topic proof, real notifier-loop sender isolation, local-path validation, and Review lifecycle wake behavior.

## Verification

- Focused preservation suite: 143 passed in 87.06s (in-process imports from the committed isolated candidate).
- Multipart retry/dedup retains its adapter and asserts zero Halo outbound attempts on the dedup tick; the discard-adapter mutation now fails.
- The installed legacy notifier suite remains incompatible with the dedicated Telegram sender unless its old adapter-send fixtures are migrated to the transport boundary; its run was recorded as 50 passed, 8 failed, 1 skipped (failures are fixture expectations, not accepted as product proof).
- `py_compile` for changed Python modules: passed.
- `git diff --check`: passed.
- Candidate checkout: clean at `a17bc0cda8738da4337a339ba8537dc1a6559e6f`.
- This manifest commit is separate from the tested candidate commit; PR #60 remains the public implementation artifact.
- Shared runtime: intentionally not switched, installed, or restarted; release authority remains the paired-release owner task.

## Release boundary

Orion must review/merge this same PR, then the release owner must perform preservation-safe installation and supported external restart. Final production proof must include health, a real watcher lifecycle notification, exact sender bot, canonical chat, message ID, and absence of `message_thread_id`.
