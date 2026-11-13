"""Isolated stdio Hermes service; deterministic model only, real turn/tool/RPC paths.

Launched by test_strict_steer_service.py with an empty temporary HERMES_HOME.
No production gateway, external provider or credentials are used.
"""
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from tui_gateway import server
from tests.agent.test_steer import TestSteerSurvivesRedirectRebuild, _STEER_SURVIVAL_TOOL
from tests.agent.test_run_agent import _mock_response

home = Path(os.environ["HERMES_HOME"])
agent = TestSteerSurvivesRedirectRebuild()._loop_agent()
agent.session_id = "strict-service"
ready = threading.Event()


def model_call(kwargs):
    if not (home / "model-started").exists():
        (home / "model-started").touch()
        deadline = time.monotonic() + 20
        while not (home / "release-model").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("test model gate")
            time.sleep(0.01)
        if os.environ.get("STRICT_TEST_NO_TOOL") == "1":
            return _mock_response(content="No tool answer", finish_reason="stop")
        tc = SimpleNamespace(id="probe", type="function", function=SimpleNamespace(
            name=_STEER_SURVIVAL_TOOL, arguments="{}"))
        return _mock_response(content=None, finish_reason="tool_calls", tool_calls=[tc])
    assert sum("strict-service-guidance" in str(m.get("content", "")) for m in kwargs["messages"]) == 1
    return _mock_response(content="Guidance present once after tool", finish_reason="stop")


agent._interruptible_api_call = model_call
owner = {
    "agent": agent, "agent_ready": ready, "session_key": agent.session_id,
    "history": [], "history_lock": threading.Lock(), "history_version": 0,
    "running": False, "attached_images": [], "image_counter": 0, "cols": 80,
    "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
    "transport": server._stdio_transport, "profile_home": str(home),
}
server._sessions["strict-ui"] = owner
server._stdio_is_rpc_channel = True


def release_startup():
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (home / "release-startup").exists():
            ready.set()
            return
        time.sleep(0.01)


threading.Thread(target=release_startup, daemon=True).start()
server.write_json({"fixture_ready": True})
for line in sys.stdin:
    response = server.dispatch(json.loads(line), transport=server._stdio_transport)
    if response is not None:
        server.write_json(response)
