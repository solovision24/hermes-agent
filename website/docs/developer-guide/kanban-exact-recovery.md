# Exact-identity Kanban recovery (source compatibility extension)

These CLI operations are opt-in local OS-owner operations, not automatic retry,
CI, review, deployment or release authority. They are pending source extensions
until independently reviewed and separately adopted. Do not run them from a
worker, impersonate an owner, clear worker markers, or edit the board with SQL.
The existing `authorize_local_owner` boundary requires a real interactive POSIX
terminal, matching OS account, owned board and exact task confirmation. It does
not isolate processes running under the same OS uid.

## Same-card, same-PR remediation

`active_pr` deliberately prevents duplicate builds. A later comment about the
same PR can also obscure an earlier review handback. Use
`authorize-pr-remediation` only for the original implementer continuing an
existing open artifact. The latest ended implementation handoff must already
record `published_pr`, full lowercase `head_sha` and `branch`; missing structured
identity is not repaired by guessing from prose.

```sh
hermes kanban authorize-pr-remediation TASK \
  --operator OS_ACCOUNT --confirm TASK --assignee ORIGINAL_IMPLEMENTER \
  --pr-url https://github.com/OWNER/REPO/pull/NUMBER \
  --head-sha FULL_HEAD_SHA --branch EXISTING_BRANCH \
  --head-repository HEAD_OWNER/HEAD_REPO --reason 'Exact same-PR remediation'
```

The command records an authorization and worker-facing scope comment, without
editing title/body/owner/workspace, requeueing, reclaiming or spawning. Use the
normal supported lifecycle operation afterward. Dispatch checks the exact card
version, owner, latest run and every recent PR URL, then rereads the live
PR/head/branch/head repository. Another PR, changed specification/owner/run,
moved or merged PR, inaccessible API or an outstanding worker/claim/open run
keeps the guard. Auth/rate-limit/recent-success guards remain intact. A crash
consumes the authorization by changing run identity; it is not permission for an
unbounded restart loop. Review continues through the existing native lane.

## Exact-contract acceptance without hosted CI

`reconcile-local-acceptance` is a separate, stronger alternative to the legacy
`reconcile-pr-contract` conversion. It leaves the exact persisted PR URL intact.
It never sets a blanket `local-only` policy, posts check statuses, marks a card
done, promotes a child or authorizes release.

1. Read `hermes kanban recovery-version TASK`. Record that content/event/run
   version, not only a body hash; edit-away-and-back invalidates earlier proof.
2. Independently verify exact-head review. Supply either an affirmative,
   independent premerge GitHub review URL, or a premerge comment corroborated by
   a completed dispatched native Orion source review with exact intake,
   claim/spawn/completion, PR/head/merge and reviewer session/check provenance.
   Later acceptance bookkeeping by Orion does not retroactively make Orion an
   author of a previously independently reviewed immutable head. Earlier actual
   implementation by that reviewer still fails independence.
3. Prepare a clean checkout at exactly the reviewed PR head or merge, with an
   `origin` matching that contract's GitHub repository. Select every applicable
   focused required test in a JSON plan. The attended owner is responsible for
   coverage and safe commands; a zero exit alone cannot prove semantic coverage.
4. Invoke the operation. It executes every argv array without an implicit shell,
   checks exit codes, hashes output, checks checkout identity/cleanliness before
   and after, and rereads mutable PR/review proof after execution.
5. Inspect the audit event. Only then use normal completion with truthful scope
   and the exact `published_pr`; full-goal, parent, review-pin and genuinely
   required implementation/safety gates remain authoritative. Existing review
   and release descendants keep their owners. Never claim live acceptance from
   this source receipt.

Plan shape (no caller-supplied `passed` field is accepted):

```json
{"cwd":"/isolated/repository","sha":"FULL_HEAD_OR_MERGE_SHA","commands":[["npm","run","focused-test"]]}
```

```sh
hermes kanban reconcile-local-acceptance TASK \
  --operator OS_ACCOUNT --confirm TASK --contract-version CURRENT_VERSION \
  --pr-url https://github.com/OWNER/REPO/pull/NUMBER \
  --head-sha FULL_REVIEWED_HEAD --merge-sha FULL_VERIFIED_MERGE \
  --review-url EXACT_AFFIRMATIVE_REVIEW_URL \
  --reason 'Owner accepts real exact-artifact local tests, not hosted CI' \
  --test-plan /isolated/required-test-plan.json
# For a corroborated native review, also supply:
# --native-review-task EXISTING_REVIEW_TASK --native-review-run DISPATCHED_RUN_ID
```

The repository must be private, the accepting GitHub identity must have admin
permission, and both classic protection and rules endpoints must return the
specific GitHub plan-limitation 403 response. A generic permission 403 is not
accepted. Any hosted check/status evidence requires normal acceptance instead.
The reviewed PR must already be merged at exactly the supplied head/merge.
Missing independent review is a real blocker, not waived by an external merge.

Normal completion revalidates GitHub/review proof outside the write transaction,
then checks the exact receipt/card/native-review binding inside the terminal
transaction. A moved contract/specification/owner/run, revoked review, changed
PR, failed later required-test attempt or concurrent write invalidates proof.
Failures do not erase historical receipts. No network runs under the terminal
SQLite write lock. Test output is not stored as raw prose or sent to GitHub.

## Existing-card limits and sequencing

- MC PR182 / `t_54fe371c`: the captured independent Orion source review remains
  available. Current reviewer-owned acceptance retries are bookkeeping, not
  premerge authorship. A fresh owner-selected exact-artifact test plan and genuine
  attended invocation are still required. Full worker-routing/native/live
  acceptance remains with its existing integration/release owner.
- SoLoVoice PR65 / `t_749cab1e`: merged head and older local checks alone do not
  supply independent approval. An old COMMENTED review on an earlier head cannot
  pass. Preserve existing review/release descendants; obtain a supported genuine
  exact-head independent review disposition rather than fabricating a receipt
  or creating a dependency loop.
- C1 / `t_7da54954`: this is not a new C1 implementation or triage operation.
  Preserve the original full specification and attachment1114. PR85 already owns
  the separately reviewed preserve-spec `resume-triage` source operation. Its
  installed gap and protected attended adoption remain with `t_e9c1ef60`.

This change is stacked above immutable PR85 source head
`0d95f38e95947cbcf0d7ecd0a777f200de3253be`; it does not mutate PR85 or authorize
adoption. Fork main no longer contains the installed compatibility acceptance
module, so porting this onto that unrelated surface is a separately sequenced
integration, not a reason to replace live code or broaden this PR.
