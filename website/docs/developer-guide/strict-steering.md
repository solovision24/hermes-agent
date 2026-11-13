# Strict run-bound steering

The TUI/desktop JSON-RPC service offers an opt-in contract for clients that must
never demote a steering request into a normal next user turn. Existing
`session.steer`, `/steer`, redirect, and their interactive fallback behavior are
unchanged. Mission Control's explicit **Steer now** is the concrete consumer.

## Protocol

1. Attach to the live runtime session using the existing session lifecycle.
   Call `session.steer.state` with `{session_id}`. Its result is
   `{run_id, active, receipts}`. A new opaque `run_id` is created when the prompt
   is admitted, before the deferred agent build completes. Idle sessions that
   have never admitted a prompt return `run_id: null, active: false`.
2. Persist the text and one client-generated `intent_id` bound to that exact
   session and `run_id` before sending `session.steer.submit` with
   `{session_id, run_id, intent_id, text}`. Preserve the identifiers literally.
   Do not substitute a new run token after a timeout or reconnect.
3. A receipt is `{run_id, intent_id, status}`. `accepted` means held for the
   active run, NOT delivered. The text is appended to a new steer-typed user
   row at the first just-completed tool batch after admission. Only then does
   the receipt become `delivered`. Previously persisted tool rows remain
   byte-identical. Concurrent legacy steering shares that single new row.
4. Subscribe to `steer.update` events, carrying the same receipt shape, or poll
   `session.steer.state`. A terminal/error/cancelled run without a checkpoint
   changes pending receipts to `missed`. Strict text never enters
   `_pending_steer`, `pending_steer`, or the normal prompt queue.
5. On a missing/lost RPC acknowledgement, retry ONLY the exact original tuple
   or read the state. The same intent/text returns its existing receipt even
   after completion. A changed text, terminal new intent, stale token, or full
   receipt table errors with `4009`. Missing/foreign/unattached sessions error
   with `4001`. Malformed parameters fail contract validation with `4000`
   (blank values are rejected with `4002`). No error authorizes normal send.

## Delivery and recovery boundaries

`delivered` proves insertion into the running agent's model-facing message list
at a real tool checkpoint. It does not prove that a provider accepted a later
request, that the model obeyed the instruction, or that the turn succeeded.
No-tool turns and startup failures are explicitly `missed`, not fallback sends.
The checkpoint is the end of a tool-call batch, not an interrupt in the middle
of a tool call. An old tool row in conversation history is not a checkpoint.

Receipts remain queryable through transport detach/reattach to the same live
runtime session and through completion until another turn starts. They are not
persistent across backend restart, session eviction, runtime replacement, or a
new turn. A new run has a new token, so an old ambiguous request cannot execute
again. If the original receipt is no longer available, outcome is **unknown**,
not "not delivered": retain recoverable client content and require an explicit
user decision before a new submission. Never auto-retry with a fresh ID/token.
A browser reload must restore the original persisted tuple, not invent one.

Each run retains up to 128 receipts without eviction; overflow fails closed.
Intent/run identifiers are limited to 128 characters and text to 65,536
characters. State and events contain identifiers/status only, never text.

Strict steering currently supports in-process agent turns. Compute-host
isolated turns fail closed with `4010` rather than admitting a request to the
wrong process. Clients must preserve the unsent content and show unsupported.
These RPCs use the existing attached-transport/session-record authority check;
a session ID alone grants neither receipt access nor steering control.

## Verification

- `tests/tui_gateway/test_strict_steer.py`: real dispatcher and contract validation,
  startup admission, exact tool insertion, duplicate/conflicting requests,
  stale/wrong-session/transport rejection, reattachment, terminal/no-tool,
  concurrent duplicate and checkpoint/finish races, lost event, receipt capacity.
- `tests/tui_gateway/test_strict_steer_service.py`: separate stdio service processes
  using the real dispatcher, `prompt.submit`, turn adapter, AIAgent loop and tool
  executor. Startup and already-running cases each cover tool delivery and
  no-tool missed outcomes, retries before/after completion, and one terminal
  response. Only the model and harmless probe tool are deterministic fixtures;
  this is not a live-provider or installed-production acceptance test.

Generated wire artifacts are updated with `scripts/gen_gateway_contracts.py`.
Installing this source requires the normal reviewed update/drain/restart lane.
Source tests do not establish that an already-running gateway supports it.
