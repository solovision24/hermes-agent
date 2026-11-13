"""Process-boundary proof using the real Hermes dispatcher, turn adapter and agent.
The LLM is deterministic; this is not a live-provider acceptance test.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import pytest


@pytest.mark.parametrize("no_tool", [False, True])
@pytest.mark.parametrize("startup", [False, True])
def test_stdio_service_startup_checkpoint_terminal_and_lost_ack(tmp_path, no_tool, startup):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_RUNTIME_DIR": str(home / "runtime"),
           "HERMES_TEST_ISOLATION": "1", "STRICT_TEST_NO_TOOL": "1" if no_tool else "0"}
    root = Path(__file__).resolve().parents[2]
    fixture = root / "tests/tui_gateway/fixtures/strict_steer_service.py"
    stderr = (tmp_path / "service.stderr").open("w+")
    proc = subprocess.Popen([sys.executable, "-u", "-c",
        f"import runpy; runpy.run_path({str(fixture)!r}, run_name='__main__')"],
        cwd=root, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True)
    inbound = queue.Queue()
    def read():
        for line in proc.stdout:
            try:
                inbound.put(json.loads(line))
            except json.JSONDecodeError:
                continue
    threading.Thread(target=read, daemon=True).start()
    seen = []
    def until(predicate):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                frame = inbound.get(timeout=0.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            seen.append(frame)
            if predicate(frame):
                return frame
        stderr.flush()
        stderr.seek(0)
        pytest.fail(f"service checkpoint timed out; frames={seen!r}; stderr={stderr.read()[-8000:]}")
    def send(rid, method, **params):
        proc.stdin.write(json.dumps({"id": rid, "method": method,
            "params": {"session_id": "strict-ui", **params}}) + "\n")
        proc.stdin.flush()
    def reply(rid):
        return until(lambda f: f.get("id") == rid)
    try:
        until(lambda f: f.get("fixture_ready"))
        if not startup:
            (home / "release-startup").touch()
        send("prompt", "prompt.submit", text="Run the test probe")
        # Read startup state without waiting for the deferred agent-ready gate.
        run_id = None
        for i in range(30):
            send(f"state-{i}", "session.steer.state")
            state = reply(f"state-{i}")["result"]
            if state["run_id"]:
                run_id = state["run_id"]
                break
            time.sleep(0.02)
        assert run_id and state["active"]
        if not startup:
            deadline = time.monotonic() + 20
            while not (home / "model-started").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (home / "model-started").exists()
        args = {"run_id": run_id, "intent_id": "stable-intent", "text": "strict-service-guidance"}
        send("admit", "session.steer.submit", **args)
        assert reply("admit")["result"]["status"] == "accepted"
        send("retry", "session.steer.submit", **args)
        assert reply("retry")["result"]["status"] == "accepted"
        (home / "release-startup").touch()
        # Receipt admission occurred while startup was blocked, before any model/tool call.
        deadline = time.monotonic() + 20
        while not (home / "model-started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (home / "model-started").exists()
        (home / "release-model").touch()
        expected = "missed" if no_tool else "delivered"
        until(lambda f: f.get("params", {}).get("type") == "steer.update"
              and f["params"]["payload"]["status"] == expected)
        complete = until(lambda f: f.get("params", {}).get("type") == "message.complete")
        assert complete["params"]["payload"]["status"] == "complete"
        # Discarded/lost first acknowledgement cannot duplicate even after completion.
        send("late-retry", "session.steer.submit", **args)
        assert reply("late-retry")["result"]["status"] == expected
        send("receipt", "session.steer.state")
        state = reply("receipt")["result"]
        assert not state["active"] and len(state["receipts"]) == 1
        assert state["receipts"][0]["status"] == expected
        assert sum(f.get("params", {}).get("type") == "message.complete" for f in seen) == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        stderr.close()
