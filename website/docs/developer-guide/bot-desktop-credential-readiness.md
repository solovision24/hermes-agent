---
title: Bot Desktop credential readiness
sidebar_label: Credential readiness
---

# Bot Desktop credential readiness

Linux Bot Desktop uses a private session bus and profile-specific XDG data. A
persistent browser profile does not imply that its encrypted credential collection
is unlocked after a new desktop session. Unlike an authenticated display-manager
login, the desktop launcher has no login password to supply to PAM/keyring unlock.
Do not join the host login bus or share another profile's collection to hide this.

## Runtime contract

`display.status` (and the other display status responses) adds optional
`credential_storage: {state, message}`. The CLI also prints the guidance in
`hermes computer-use screen status`. Existing desktop/browser launch behavior is
unchanged. This is readiness/diagnostic groundwork, not automatic unlock, browser
launch gating, an encrypted-storage guarantee, or a complete companion UI fix.

| State | Meaning | Suggested UI behavior |
| --- | --- | --- |
| `locked` | Existing default collection is locked. | Show inline guidance beside Take over; user authenticates in the system dialog. |
| `unlocked` | Default collection reports unlocked. | Clear unlock guidance; do not claim browser backend/encryption is verified. |
| `setup_required` | Provider is running, but no default alias exists. | Explain encrypted, non-empty-password setup in the system UI. |
| `inactive` | Provider is activatable, but has no owner. | Explain first browser access can start setup/unlock; status does not start it. |
| `unavailable` | No bus was published or no provider is activatable. | Guide encrypted-provider setup for this desktop/profile. Never suggest basic/plaintext storage. |
| `unknown` | Missing probe utility, denied/timed-out connection, malformed reply or unsupported bus address. | Show an honest unverified state, not success. |
| `null`/absent | No running local Linux desktop was probed, a remote/sandbox target, or an older runtime. | Do not report the gateway host's keyring as the target's status. |

The probe has one shared 800 ms deadline. It calls `busctl` with an explicit
published local Unix address, `--auto-start=no`, no inherited credential/session
environment, and stderr suppressed. It reads NameHasOwner, activatable service
names when needed, ReadAlias(default), and the collection's Locked flag. It never
reads labels, attributes, secrets, browser cookies, or keyring files; invokes
Unlock/Prompt; starts a service; or changes the screen/lease. A failed read is
`unknown`; there is no fallback to the gateway's ambient bus. `busctl` absence is
not permission to downgrade browser storage. This revision does not probe remote
or sandbox storage until a target-local execution path is integrated.

## Product ownership and remaining acceptance

The private-bus/browser lifecycle owner is upstream Hermes
`tools/bot_desktop/{launcher.sh,runtime.py,browser.py}`. A companion integrating
`display.status` owns rendering readiness, setup/unlock guidance, and bounded
user-requested retries. For SoLoAgent/Mission Control, the relevant bridge is
`src-tauri/src/api_service_display.rs::status_view` (currently preserves extra
status fields); its `src-web/src/{lib/agent-screen.ts,components/AgentScreen.tsx}`
consumer still needs explicit readiness presentation. Do not claim this runtime
foundation alone fixes the companion's repeated unexplained prompts.

Scope demonstrated on one Linux host: disposable HOME/XDG directories, private
D-Bus, real GNOME Keyring, Xvfb, and system Chrome using gnome-libsecret. This is
not a fresh OS installation or a second Linux-user acceptance run, and does not
establish effects on macOS, Windows, KWallet, all distros, or all users. Private
buses/directories under one Unix UID avoid accidental routing, not hostile
same-UID access; test a genuinely different UID for OS authorization claims.

A legitimate user authentication can be required again after a fresh desktop
session. A one-time setup flow is not a promise of permanent passwordless unlock.
Do not persist login passwords in scripts/environment, reset/delete keyrings,
blank passwords, enable `--password-store=basic`, or suppress/kill a prompt as a
substitute for proving its original trigger no longer prompts after authentication.

## Rollout and rollback

This additive diagnostic does not migrate credential state. Review source before
release; preserve the existing launcher, browser profile and encrypted keyrings.
Verify the installed revision and per-profile status, then coordinate attended
unlock and the exact browser relaunch/fresh-desktop checks before claiming host
recovery. Human control or an expired lease is never implied restart consent.
Roll back the source/runtime artifact to the recorded previous revision if the
metadata check misbehaves; no keyring restore/reset is part of that rollback.

Related references: [Secret Service API](https://specifications.freedesktop.org/secret-service-spec/latest/),
[GNOME Keyring PAM overview (archived)](https://wiki.gnome.org/Projects/GnomeKeyring/Pam).
