"""Dashboard HTTP contract for hosted MCP OAuth."""

import asyncio

from unittest.mock import patch

import pytest


_TEST_LOOP = asyncio.new_event_loop()


@pytest.fixture(scope="module", autouse=True)
def _close_test_loop():
    yield
    _TEST_LOOP.run_until_complete(_TEST_LOOP.shutdown_default_executor())
    _TEST_LOOP.close()


class _ASGIClient:
    """Small synchronous facade without TestClient's cross-thread portal."""

    def __init__(self, app, headers=None):
        self.app = app
        self.headers = dict(headers or {})

    def request(self, method, path, **kwargs):
        import httpx

        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://127.0.0.1",
                headers=self.headers,
            ) as client:
                return await client.request(method, path, **kwargs)

        return _TEST_LOOP.run_until_complete(send())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


def _client():
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    return _ASGIClient(app, {_SESSION_HEADER_NAME: _SESSION_TOKEN})


@pytest.fixture(autouse=True)
def _clear_flows(monkeypatch):
    from hermes_cli import web_server
    from hermes_cli.web_routers import mcp as mcp_router

    async def inline_to_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(mcp_router.asyncio, "to_thread", inline_to_thread)

    web_server._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False
    yield
    web_server._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False


def test_hosted_auth_start_returns_public_authorization_url(tmp_path, monkeypatch):
    from hermes_cli import web_server

    profile_home = tmp_path / "profiles" / "oauth-test"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)
    client = _client()

    def fake_worker(flow, cfg):
        try:
            asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))
        finally:
            flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={
        "reports": {"url": "https://mcp.example/mcp", "auth": "oauth"}
    }), patch(
        "hermes_cli.dashboard_auth.prefix.resolve_public_url", return_value="https://agent.example"
    ):
        response = client.post("/api/mcp/servers/reports/auth?profile=oauth-test")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "authorization_required"
    assert body["authorization_url"] == "https://idp.example/authorize?state=s1"
    flow = web_server._mcp_oauth_flows[body["flow_id"]]
    assert flow._worker_done.wait(1)
    assert flow.redirect_uri == "https://agent.example/api/mcp/oauth/callback/reports"


def test_mission_control_oauth_capability_is_profile_and_server_scoped(tmp_path, monkeypatch):
    from hermes_cli import web_server

    profile_home = tmp_path / "profiles" / "phaseboauth"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)
    client = _client()
    with patch(
        "hermes_cli.mcp_config._get_mcp_servers",
        return_value={"linear": {"url": "https://mcp.linear.app/mcp", "auth": "oauth"}},
    ):
        response = client.get(
            "/api/capabilities?profile=phaseboauth&server=linear"
        )

    assert response.status_code == 200
    assert response.json() == {
        "capabilities": {
            "mcp_oauth_flow_v1": {"callback_mode": "relay"},
        }
    }


def test_mission_control_starts_relay_flow_for_exact_target(tmp_path, monkeypatch):
    import asyncio

    from hermes_cli import web_server

    profile_home = tmp_path / "profiles" / "phaseboauth"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)

    def fake_worker(flow, cfg):
        try:
            asyncio.run(
                flow.publish_authorization_url(
                    "https://linear.app/oauth/authorize?state=expected-state"
                )
            )
        finally:
            flow.mark_worker_done()

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    with patch(
        "hermes_cli.mcp_config._get_mcp_servers",
        return_value={"linear": {"url": "https://mcp.linear.app/mcp", "auth": "oauth"}},
    ):
        response = _client().post(
            "/api/mcp/oauth/start",
            json={
                "profile": "phaseboauth",
                "server": "linear",
                "callback_url": "https://mc.example/api/mcp/oauth/callback",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "flow_id": body["flow_id"],
        "authorization_url": "https://linear.app/oauth/authorize?state=expected-state",
    }
    flow = web_server._mcp_oauth_flows[body["flow_id"]]
    assert flow._worker_done.wait(1)
    assert flow.profile == "phaseboauth"
    assert flow.server_name == "linear"
    assert flow.redirect_uri == "https://mc.example/api/mcp/oauth/callback"


def test_mission_control_status_is_bound_to_exact_profile_and_server():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-scoped",
        server_name="linear",
        profile="phaseboauth",
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://mc.example/api/mcp/oauth/callback",
    )
    flow.status = "approved"
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    client = _client()
    accepted = client.get(
        "/api/mcp/oauth/status?profile=phaseboauth&server=linear&flow_id=flow-scoped"
    )
    rejected = client.get(
        "/api/mcp/oauth/status?profile=other&server=linear&flow_id=flow-scoped"
    )

    assert accepted.status_code == 200
    assert accepted.json() == {"status": "succeeded"}
    assert rejected.status_code == 404


def test_mission_control_relay_consumes_callback_once():
    import asyncio

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-relay",
        server_name="linear",
        profile="phaseboauth",
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://mc.example/api/mcp/oauth/callback",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://linear.app/oauth/authorize?state=expected-state"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    payload = {
        "profile": "phaseboauth",
        "server": "linear",
        "flow_id": "flow-relay",
        "code": "provider-code",
        "state": "expected-state",
    }

    first = _client().post("/api/mcp/oauth/relay", json=payload)
    replay = _client().post("/api/mcp/oauth/relay", json=payload)

    assert first.status_code == 200
    assert first.json() == {"status": "pending"}
    assert flow._callback == ("provider-code", "expected-state")
    assert replay.status_code == 409
    assert "provider-code" not in first.text
    assert "expected-state" not in first.text


def test_mission_control_cancel_is_bound_to_exact_target():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-cancel",
        server_name="linear",
        profile="phaseboauth",
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://mc.example/api/mcp/oauth/callback",
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    client = _client()

    rejected = client.post(
        "/api/mcp/oauth/cancel",
        json={"profile": "other", "server": "linear", "flow_id": "flow-cancel"},
    )
    accepted = client.post(
        "/api/mcp/oauth/cancel",
        json={
            "profile": "phaseboauth",
            "server": "linear",
            "flow_id": "flow-cancel",
        },
    )

    assert rejected.status_code == 404
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True}
    assert flow.status == "error"


def test_mission_control_rejects_unsafe_oauth_callback_url(tmp_path, monkeypatch):
    from hermes_cli import web_server

    profile_home = tmp_path / "profiles" / "phaseboauth"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)
    with patch(
        "hermes_cli.mcp_config._get_mcp_servers",
        return_value={"linear": {"url": "https://mcp.linear.app/mcp", "auth": "oauth"}},
    ):
        response = _client().post(
            "/api/mcp/oauth/start",
            json={
                "profile": "phaseboauth",
                "server": "linear",
                "callback_url": "javascript:alert(1)",
            },
        )

    assert response.status_code == 422
    assert not web_server._mcp_oauth_flows


def test_mission_control_rejects_oversized_relay_material():
    import asyncio

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-bounded",
        server_name="linear",
        profile="phaseboauth",
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://mc.example/api/mcp/oauth/callback",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://linear.app/oauth/authorize?state=expected-state"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().post(
        "/api/mcp/oauth/relay",
        json={
            "profile": "phaseboauth",
            "server": "linear",
            "flow_id": "flow-bounded",
            "code": "x" * 4097,
            "state": "expected-state",
        },
    )

    assert response.status_code == 422
    assert flow._callback is None


def test_hosted_callback_bypasses_gated_cookie_auth(monkeypatch):
    import asyncio

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-gated",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/reports",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    response = _ASGIClient(web_server.app).get(
        "/api/mcp/oauth/callback/reports?code=abc&state=expected"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected")


def test_hosted_auth_allows_same_server_name_in_different_profiles(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)

    existing = DashboardOAuthFlow(
        flow_id="existing-default",
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path / "default"),
        redirect_uri="https://agent.example/callback/existing",
    )
    web_server._mcp_oauth_flows[existing.flow_id] = existing

    def fake_worker(flow, cfg):
        try:
            asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=work"))
        finally:
            flow.mark_worker_done()

    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={"reports": {"url": "https://mcp.example"}}), \
         patch.object(web_server, "_run_dashboard_mcp_oauth", fake_worker):
        response = _client().post("/api/mcp/servers/reports/auth?profile=work")

    assert response.status_code != 409




def test_flow_status_does_not_expose_authorization_code():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-status",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-status",
    )
    flow.authorization_url = "https://idp.example/authorize"
    flow.status = "approved"
    flow._callback = ("secret-code", "secret-state")
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().get("/api/mcp/oauth/flows/flow-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert "secret-code" not in response.text
    assert "secret-state" not in response.text
