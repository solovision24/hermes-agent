"""Tests for GHSA-ppp5-vxwm-4cf7 — Host-header validation.

DNS rebinding defence: a victim browser that has the dashboard open
could be tricked into fetching from an attacker-controlled hostname
that TTL-flips to 127.0.0.1. Same-origin / CORS checks won't help —
the browser now treats the attacker origin as same-origin. Validating
the Host header at the application layer rejects the attack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo = str(Path(__file__).resolve().parents[1])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


class TestHostHeaderValidator:
    """Unit test the _is_accepted_host helper directly — cheaper and
    more thorough than spinning up the full FastAPI app."""



    def test_zero_zero_bind_accepts_anything(self):
        """0.0.0.0 means operator explicitly opted into all-interfaces
        (requires --insecure). No Host-layer defence is possible — rely
        on operator network controls."""
        from hermes_cli.web_server import _is_accepted_host

        for host in ("10.0.0.5", "evil.example", "my-server.corp.net"):
            assert _is_accepted_host(host, "0.0.0.0")
            assert _is_accepted_host(host + ":9119", "0.0.0.0")

    def test_explicit_non_loopback_bind_requires_exact_match(self):
        """If the operator bound to a specific non-loopback hostname,
        the Host header must match exactly."""
        from hermes_cli.web_server import _is_accepted_host

        assert _is_accepted_host("my-server.corp.net", "my-server.corp.net")
        assert _is_accepted_host("my-server.corp.net:9119", "my-server.corp.net")
        # Different host — reject
        assert not _is_accepted_host("evil.example", "my-server.corp.net")
        # Loopback — reject (we bound to a specific non-loopback name)
        assert not _is_accepted_host("localhost", "my-server.corp.net")



class TestHostHeaderMiddleware:
    """End-to-end test via the FastAPI app — verify the middleware
    rejects bad Host headers with 400."""

    def test_rebinding_request_rejected(self):
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Simulate start_server having set the bound_host
        app.state.bound_host = "127.0.0.1"
        try:
            client = TestClient(app)
            # The TestClient sends Host: testserver by default — which is
            # NOT a loopback alias, so the middleware must reject it.
            resp = client.get(
                "/api/status",
                headers={"Host": "evil.example"},
            )
            assert resp.status_code == 400
            assert "Invalid Host header" in resp.json()["detail"]
        finally:
            # Clean up so other tests don't inherit the bound_host
            if hasattr(app.state, "bound_host"):
                del app.state.bound_host


    def test_no_bound_host_skips_validation(self):
        """If app.state.bound_host isn't set (e.g. running under test
        infra without calling start_server), middleware must pass through
        rather than crash."""
        from fastapi.testclient import TestClient
        from hermes_cli.web_server import app

        # Make sure bound_host isn't set
        if hasattr(app.state, "bound_host"):
            del app.state.bound_host

        client = TestClient(app)
        resp = client.get("/api/status")
        # Should get through to the status endpoint, not a 400
        assert resp.status_code != 400


class TestWebSocketHostOriginGuard:
    """WebSocket upgrades must enforce the same dashboard boundary as HTTP."""

    def _ws_reason(self, origin: str, host: str = "127.0.0.1:9119", monkeypatch=None):
        """Drive ``_ws_host_origin_reason`` with a fake WS carrying the
        given Host/Origin headers against a loopback bind."""
        from hermes_cli.web_server import _ws_host_origin_reason, app

        if monkeypatch is not None:
            monkeypatch.setattr(app.state, "bound_host", "127.0.0.1", raising=False)

        class _FakeHeaders(dict):
            def get(self, key, default=None):
                return dict.get(self, key.lower(), default)

        class _FakeWS:
            headers = _FakeHeaders({"host": host, "origin": origin})

        return _ws_host_origin_reason(_FakeWS())

    def test_tunneled_public_origin_accepted_when_configured(self, monkeypatch):
        """A Cloudflare-tunnel deployment rewrites the Host header to the
        loopback bind but leaves the browser's Origin pointing at the public
        hostname. When the operator declares HERMES_DASHBOARD_PUBLIC_URL,
        the WS origin guard must accept that host on a loopback bind."""
        import hermes_cli.web_server as ws

        monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://hermes.solobot.cloud")
        reason = self._ws_reason(
            "https://hermes.solobot.cloud",
            host="127.0.0.1:9119",
            monkeypatch=monkeypatch,
        )
        assert reason is None
        # Ensure the helper really read the env (frozenset is not empty).
        assert ws._declared_public_origin_hosts() == frozenset({"hermes.solobot.cloud"})

    def test_tunneled_public_origin_rejected_without_config(self, monkeypatch):
        """Without a declared public URL the strict loopback-only origin
        default must reject the tunneled origin exactly as before."""
        monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
        reason = self._ws_reason(
            "https://hermes.solobot.cloud",
            host="127.0.0.1:9119",
            monkeypatch=monkeypatch,
        )
        assert reason == "origin_mismatch origin=https://hermes.solobot.cloud bound=127.0.0.1"

    def test_public_origin_must_match_declared_host_exactly(self, monkeypatch):
        """Accepting the declared public origin must not widen the gate to
        sibling/subdomains or unrelated hosts (DNS-rebinding defence)."""
        monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://hermes.solobot.cloud")
        for evil in (
            "https://evil.example",
            "https://solobot.cloud",
            "https://hermes.solobot.cloud.evil.example",
            "https://sub.hermes.solobot.cloud",
        ):
            reason = self._ws_reason(
                evil,
                host="127.0.0.1:9119",
                monkeypatch=monkeypatch,
            )
            assert reason == f"origin_mismatch origin={evil} bound=127.0.0.1"

    def test_loopback_origin_still_accepted_with_config(self, monkeypatch):
        """Loopback origins keep working even with a public URL declared —
        both the local SPA and the tunneled origin must be usable."""
        monkeypatch.setenv("HERMES_DASHBOARD_PUBLIC_URL", "https://hermes.solobot.cloud")
        assert (
            self._ws_reason(
                "http://localhost:9119",
                host="127.0.0.1:9119",
                monkeypatch=monkeypatch,
            )
            is None
        )

    def test_rebinding_websocket_host_is_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(
                url,
                headers={
                    "Host": "evil.example",
                    "Origin": "http://evil.example",
                },
            ):
                pass

        assert exc.value.code == 4403


    def test_loopback_websocket_host_and_origin_are_accepted(self, monkeypatch):
        from fastapi.testclient import TestClient

        import hermes_cli.web_server as ws

        monkeypatch.setattr(ws.app.state, "bound_host", "127.0.0.1", raising=False)
        monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)

        client = TestClient(ws.app)
        url = f"/api/events?token={ws._SESSION_TOKEN}&channel=security-test"
        with client.websocket_connect(
            url,
            headers={
                "Host": "localhost:9119",
                "Origin": "http://localhost:9119",
            },
        ):
            pass
