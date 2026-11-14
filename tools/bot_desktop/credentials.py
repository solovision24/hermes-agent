"""Read-only credential readiness for a bot desktop's private Secret Service.

Never use the gateway's ambient/session bus as a fallback. This probe reads only
service ownership, the default alias, and its Locked flag: no secrets, labels,
Unlock, prompting, provider activation, or disk/keyring modifications. An unlocked
collection is not proof of its at-rest encryption or of browser backend selection.
"""
from __future__ import annotations

import json
import subprocess
import time
from typing import Mapping

_MESSAGES = {
    "locked": ("Browser credential storage is locked. Take control of this screen and unlock the existing "
               "keyring in the system authentication dialog. Enter its password there, never in chat. "
               "A fresh desktop session may require authentication again; do not reset the keyring."),
    "unlocked": "The default credential collection is unlocked. Browser backend and at-rest encryption are not verified by this check.",
    "setup_required": ("No default credential collection is configured. Take control and complete encrypted "
                       "keyring setup in the system dialog with a non-empty password before saving browser logins."),
    "inactive": ("The credential provider is available but not running in this desktop session. The browser "
                 "may start it and request keyring setup or unlock. Authenticate only in the system dialog."),
    "unavailable": ("No Secret Service provider is available in this desktop session. Configure encrypted "
                    "credential storage for this profile; do not use plaintext storage or another profile's bus."),
    "unknown": "Credential readiness could not be checked. Do not assume browser logins have encrypted storage.",
}


def probe(env: Mapping[str, str], *, timeout: float = 0.8) -> dict[str, str]:
    """Bounded metadata-only probe, explicitly routed to the caller's published bus.

    ``env`` must belong to the already-running, owning desktop. Sandboxed/remote
    desktops must probe at their execution target, never on the gateway host.
    """
    address = env.get("DBUS_SESSION_BUS_ADDRESS", "")
    if not address:
        # No published route means unprobed, not evidence of a missing provider.
        return _result("unknown")
    # Bot Desktop publishes a private local Unix bus. Refuse network/ambient bus
    # routing, multiple-address fallbacks, and values with control characters.
    if (not address.startswith("unix:path=/") or ";" in address or
            any(ord(c) < 32 for c in address)):
        return _result("unknown")
    deadline = time.monotonic() + timeout

    def call(*args: str):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        proc = subprocess.run(
            ["busctl", "--address=" + address, "--json=short", "--auto-start=no",
             "--timeout=" + str(remaining), *args],
            # Nothing inherits provider credentials or the gateway's bus/session.
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=remaining, check=True,
        )
        return json.loads(proc.stdout)["data"]

    try:
        owned = call("call", "org.freedesktop.DBus", "/org/freedesktop/DBus",
                     "org.freedesktop.DBus", "NameHasOwner", "s", "org.freedesktop.secrets")
        if not isinstance(owned, list) or len(owned) != 1 or type(owned[0]) is not bool:
            return _result("unknown")
        if owned == [False]:
            available = call("call", "org.freedesktop.DBus", "/org/freedesktop/DBus",
                             "org.freedesktop.DBus", "ListActivatableNames")
            if (not isinstance(available, list) or len(available) != 1 or
                    not isinstance(available[0], list)):
                return _result("unknown")
            return _result("inactive" if "org.freedesktop.secrets" in available[0] else "unavailable")

        alias = call("call", "org.freedesktop.secrets", "/org/freedesktop/secrets",
                     "org.freedesktop.Secret.Service", "ReadAlias", "s", "default")
        if alias == ["/"]:
            return _result("setup_required")
        if (not isinstance(alias, list) or len(alias) != 1 or not isinstance(alias[0], str) or
                not alias[0].startswith("/org/freedesktop/secrets/collection/")):
            return _result("unknown")
        locked = call("get-property", "org.freedesktop.secrets", alias[0],
                      "org.freedesktop.Secret.Collection", "Locked")
        if type(locked) is not bool:
            return _result("unknown")
        return _result("locked" if locked else "unlocked")
    except (OSError, subprocess.SubprocessError, TimeoutError, ValueError, KeyError, TypeError):
        # No exception text: it can contain local paths, bus addresses or provider output.
        return _result("unknown")


def _result(state: str) -> dict[str, str]:
    return {"state": state, "message": _MESSAGES[state]}
