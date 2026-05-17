from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_SRC = REPO_ROOT / "web" / "src"


def _read(rel: str) -> str:
    return (WEB_SRC / rel).read_text(encoding="utf-8")


def test_websocket_url_helper_honors_dashboard_base_path():
    """Cloudflare/prefix deployments need WS URLs under the same base path as fetch()."""
    api_ts = _read("lib/api.ts")

    assert "export function buildWebSocketUrl" in api_ts
    assert "${BASE}${normalizedPath}" in api_ts
    assert "window.location.protocol === \"https:\" ? \"wss:\" : \"ws:\"" in api_ts


def test_chat_websockets_use_base_path_aware_helper():
    """Regression for Chat tab 'session ended' behind Cloudflare path proxies.

    The REST client already prepends ``window.__HERMES_BASE_PATH__``. The Chat
    PTY, JSON-RPC, and events WebSockets must do the same instead of hardcoding
    root-relative /api paths, otherwise a dashboard served at /hermes will load
    but its browser WebSockets connect to the wrong origin path and close.
    """
    chat_page = _read("pages/ChatPage.tsx")
    sidebar = _read("components/ChatSidebar.tsx")
    gateway = _read("lib/gatewayClient.ts")

    assert 'buildWebSocketUrl("/api/pty", { token, channel, resume })' in chat_page
    assert 'buildWebSocketUrl("/api/events", { token, channel })' in sidebar
    assert 'buildWebSocketUrl("/api/ws", { token: resolved })' in gateway

    forbidden_fragments = [
        "window.location.host}/api/pty?",
        "window.location.host}/api/events?",
        "location.host}/api/ws?",
    ]
    combined = "\n".join([chat_page, sidebar, gateway])
    for fragment in forbidden_fragments:
        assert fragment not in combined
