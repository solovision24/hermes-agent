"""Human control expires from actual input, independently for each desktop."""
import json
from tools.bot_desktop import lease


def test_legacy_human_lease_expires_after_thirty_minutes(tmp_path, monkeypatch):
    home = str(tmp_path)
    monkeypatch.setattr(lease.time, 'time', lambda: 1000.0)
    held = lease.acquire('abandoned', profile_key=home)
    # A lease written by the previous version has only since, no input timestamp.
    path = lease._path(home)
    data = json.loads(path.read_text())
    data.pop('last_input_at', None)
    path.write_text(json.dumps(data))
    monkeypatch.setattr(lease.time, 'time', lambda: 2799.0)
    assert lease.get(profile_key=home).holder == lease.HUMAN
    monkeypatch.setattr(lease.time, 'time', lambda: 2800.0)
    expired = lease.get(profile_key=home)
    assert expired.holder == lease.AGENT
    assert expired.viewer_id is None
    assert expired.epoch == held.epoch + 1
    assert lease.get(profile_key=home).epoch == expired.epoch


def test_only_holder_input_extends_control_and_profiles_expire_independently(tmp_path, monkeypatch):
    a, b = str(tmp_path/'a'), str(tmp_path/'b')
    now = [1000.0]
    monkeypatch.setattr(lease.time, 'time', lambda: now[0])
    first = lease.acquire('a', profile_key=a)
    lease.acquire('b', profile_key=b)
    now[0] += 1799
    assert not lease.record_input('watcher', profile_key=b)
    assert lease.record_input('a', profile_key=a)
    assert lease.get(profile_key=a).epoch == first.epoch
    now[0] += 1
    assert lease.get(profile_key=b).holder == lease.AGENT
    assert lease.get(profile_key=a).holder == lease.HUMAN
    # Merely polling status, observing, or checking input permission doesn't renew it.
    now[0] += 1799
    assert not lease.viewer_may_send_input('a', profile_key=a)
    assert not lease.record_input('a', profile_key=a), 'expired input cannot reacquire control'
    assert lease.get(profile_key=a).holder == lease.AGENT


def test_real_bridge_records_input_but_not_framebuffer_requests(tmp_path, monkeypatch):
    import asyncio
    from hermes_cli.web_routers import display
    home = str(tmp_path)
    now = [1000.0]
    monkeypatch.setattr(lease.time, 'time', lambda: now[0])
    lease.acquire('human', profile_key=home)

    class Ws:
        step = 0
        async def receive(self):
            self.step += 1
            if self.step == 1:
                now[0] = 2798.0
                return {'bytes': b'RFB 003.008\n' + b'\x01\x01' + b'\x03\x00' + bytes(8)}
            if self.step == 2:
                assert lease.get(profile_key=home).last_input_at == 1000.0
                now[0] = 2799.0
                return {'bytes': b'\x04\x01\x00\x00\x00\x00\x00\x61'}
            return {'type': 'websocket.disconnect', 'code': 1006}
        async def send_bytes(self, data): pass
        async def close(self, **kwargs): pass

    async def run():
        sock = tmp_path/'bot-desktop'/'rfb.sock'
        async def silent(reader, writer):
            await reader.read()
            writer.close()
        server = await asyncio.start_unix_server(silent, path=str(sock))
        try:
            await display._bridge(Ws(), {'hermes_home': home, 'viewer_id': 'human'})
        finally:
            server.close()
            await server.wait_closed()
    asyncio.run(run())
    assert lease.get(profile_key=home).last_input_at == 2799.0


def test_background_watcher_expires_unchanged_lease_without_status_or_viewer(tmp_path, monkeypatch):
    import tui_gateway.server as server
    now = [1000.0]
    monkeypatch.setattr(lease.time, 'time', lambda: now[0])
    monkeypatch.setattr(server, '_hermes_home', str(tmp_path))
    monkeypatch.setattr(server, '_served_profile_homes', [])
    events = []
    monkeypatch.setattr(server, '_broadcast_global_event', lambda *args: events.append(args))
    lease.acquire('abandoned', profile_key=str(tmp_path))
    server._poll_lease_files()  # seed cached file mtime
    now[0] += 1800
    server._poll_lease_files()  # no file changes and no UI status requests
    assert lease._read(lease._path(str(tmp_path))).holder == lease.AGENT
    assert any(ev == 'display.lease' and payload['lease']['holder'] == lease.AGENT for ev, payload in events)


def test_explicit_takeover_after_expiry_starts_a_fresh_human_turn(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(lease.time, 'time', lambda: now[0])
    before = lease.acquire('human', profile_key=str(tmp_path))
    now[0] += 1800
    after = lease.acquire('human', profile_key=str(tmp_path))
    assert after.epoch > before.epoch
    assert lease.get(profile_key=str(tmp_path)).holder == lease.HUMAN
    assert after.last_input_at == now[0]
