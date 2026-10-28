---
title: "Staged engineering handoff template"
description: "Cross-project build, review, merge and release evidence without review-before-review"
---

# Staged engineering handoff template

This repo-agnostic template separates evidence by stage. The SoLo deployment's
universal policy authority is `/home/solo/.hermes/AGENTS.md`; decision provenance
is SoLo's October 27, 2026 approval on `t_c3179622`. Runtime changes in this branch
are a local compatibility candidate based on reviewed `be3efe9fff09cddbbc0f4448db7b650f1526c843`,
not an upstream release or an installed-enforcement claim.

## BUILD / REVIEW submission

- Canonical card and owner: `<task ID / qualified implementer>`.
- Scope and boundaries: `<in-scope implementation / preserved safety constraints>`.
- Branch / PR / immutable full head: `<branch / exact PR URL / 40-character SHA>`.
- Required implementation verification: `<commands / results / log or artifact paths>`.
- Safety blockers: `<none known after inspection, or exact unresolved findings>`.
- Independent review: **requested**, not already required to exist.
- Deployment implications: `<restart, schema, data, rollback, no-deploy classification>`.
- Remaining integration/release acceptance: `<specific criterion / existing dependent card / assigned owner / decision provenance>`.
- Program acceptance: **preserved**, not claimed complete or live by this source handoff.

Use the same implementation card and `reviewer=orion` in the SoLo deployment.
The PR body includes `## Kanban Task`, the canonical ID, and
`https://hmc.sololink.cloud/#/tasks?task=<task_id>`.

Suggested structured handoff metadata (replace all placeholders):

```json
{
  "published_pr": "https://github.com/OWNER/REPO/pull/123",
  "head_sha": "<full lowercase 40-character Git SHA>",
  "original_implementer": "<profile>",
  "required_tests": [{"command": "<actual command>", "status": "passed"}],
  "safety_blockers": [],
  "deployment_implications": "<what release owner must verify>",
  "dependent_acceptance": [{"task_id": "<existing ID>", "owner": "<assigned profile>", "criterion": "<specific bounded criterion>", "decision": "<canonical decision reference>"}]
}
```

`dependent_acceptance` is evidence for review, not deferral authority. The reviewed
owner-only reconciliation path remains the authority for operative completion-scope
changes, with exact source/child text, hashes, link, owner, actor/reason and atomic
race checks. No free-form metadata waives a safety issue or a failed required test.

## Independent REVIEW / MERGE verdict

Before the first submission, self-review the entire scoped diff and independently
exercise the happy path and relevant failures. Independent review is not the
implementer's first debugger. Label stubs, fixtures, missing checks and live gaps
honestly. On every remediation, include a closure matrix for ALL prior findings:

| Canonical finding/event | Classification | Fix or remaining blocker | Exercised verification / evidence |
|---|---|---|---|
| `<event ID + finding>` | `<new / regression / unresolved>` | `<exact change or justified open issue>` | `<actual command and result, or explicitly unverified>` |

Orion inspects the full scoped diff and affected neighboring paths, batches every
known blocker into one verdict, and distinguishes new findings, regressions and
unresolved prior findings. Never knowingly drip-feed findings. New discoveries and
immediate stop-ship safety findings remain allowed. After two unsuccessful
remediation rounds, Halo triages cause, scope/architecture or qualified DEV/Forge
ownership before another automatic patch-only cycle; preserve the SAME canonical
card and current Ops redesign owner. There is no artificial approval or review-round
cap and no waiver of exact-head independent approval. The shared admission judge
receives canonical prior `changes_requested` findings, not only the latest summary.

- Reviewed immutable head: `<full SHA, re-read from current PR>`.
- Independent reviewer and native claim: `<profile / task / run>`.
- Checks actually executed: `<commands / results / evidence>`.
- Findings: `<correctable defects or no blocking findings>`.
- Source decision: `<request changes / approve source / approved and merged>`.
- Required checks: `<applicable repository-required evidence; no fabricated CI>`.
- Deferred program acceptance: `<specific criteria / existing assigned dependent owners>`.
- Deployment state: **not live** unless separately verified.

Structured approval metadata uses `reviewed_head`, `review_outcome` (`approved`,
`approved_source_only`, or `approved_merged_source_only`), and a nonempty
`reviewer_checks` list. A structured handoff head requires a real independent
native review claim before completion. If a published PR is declared, its current
head is re-read with the review assignee's GitHub scope; stale/missing evidence
refuses completion, including for `local-only` source tasks. A changed head requires
a new review. A declared `published_pr` (or native intake `url`) requires a full
lowercase 40-character `head_sha` at admission. Invalid/conflicting PR URLs and
missing/malformed pins fail closed at admission and historical handoff completion;
terminal approval cannot guess or backfill an omitted immutable pin from GitHub.
Legacy **non-PR** handoffs without structured head pins retain the existing
lifecycle; published PRs never use that compatibility exemption. This is not a
universal retrospective approval claim.

## RELEASE receipt

- Canonical release card / authorized owner: `<existing card / owner>`.
- Exact approved source and deployable artifact: `<SHA / version / digest>`.
- Maintenance and data safety: `<backup consistency / restore rehearsal / drain proof>`.
- Change route and rollback: `<reviewed operator / bounded scope / rollback receipt>`.
- Installed/served proof: `<actual process/binary/assets hashes / readiness / live behavior>`.
- Acceptance remaining: `<specific unverified items, never implicit green>`.

Review source before requiring these live proofs. An attended-only operation still
requires a real attended owner window; never impersonate an owner terminal, invent
probes/ACKs, force review, edit raw DB state or interrupt healthy workers to escape
a stage boundary.

## Before / after and validation map

| Boundary | Before | Candidate behavior | Regression evidence |
|---|---|---|---|
| Review judge | Same full-goal completion semantics as `complete` | Fixed SYSTEM stage policy judges readiness; original body is preserved | `test_kanban_stages.py`: MC1 and unrelated project, domain/CLI/tool |
| Dashboard/domain admission | Did not share the CLI/tool judge | `request_review` domain invokes one judge before write txn; scope/run/ownership rechecked under lock | Cross-surface and race cases |
| Explicit required-test/safety failures | Prose/model assessment only | Declared `required_tests` failures or nonempty `safety_blockers` deterministically refuse admission | Negative metadata cases, including non-goal tasks |
| Deferral provenance | Checked during owner reconciliation | Revalidated at review/completion; changed parent/child/link/owner/onward deferral refuses | Scope-drift cases plus existing reconciliation suite |
| Exact-head terminal approval | General PR required-check gate and recovery-specific native provenance | Structured native pins additionally require independent review and current remote head | Wrong/missing head, self-review, moved remote, reviewer auth cases |
| `active_pr` | Ready-lane duplicate-build guard; review lane skips it | Unchanged; never remove duplicate-build protection to request review | Existing lifecycle/dispatcher tests |
| Completion/release | Full-goal judge and PR acceptance | Unchanged full-goal semantics; review admission never marks done/releases children | Full-goal control plus source/release dependency test |

Run `scripts/run_tests.sh tests/hermes_cli/test_kanban_stages.py -j 1 -p tests.hermes_cli.scope_preload`
plus the existing scope/reconciliation/native-review/goals/PR acceptance suites.
Fixture model transports prove prompt/parser/transition wiring, not model reasoning,
independent PR approval, installed code or live MC1 acceptance. Record actual JUnit
counts, failures/errors/skips; do not call skipped tests passed.

## Deployment boundary

Do not apply an old hardcoded adoption manifest to a new candidate. Derive and
review the complete exact delta against the actual live checkout; preserve tracked
and untracked drift. Use the existing Ops adoption/recovery owners, safe backup,
drain, rollback and installed-artifact proof. Run a real same-card native PR183
handoff only after reviewed installed enforcement exists; bridge and release
acceptance remain on their existing canonical owners.
