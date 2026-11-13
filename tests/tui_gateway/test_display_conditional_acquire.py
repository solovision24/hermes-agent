"""The declared RPC forwards optional agent-held compare-and-set acquisition."""
from tui_gateway.contracts.display import DisplayLeaseAcquireParams


def test_conditional_acquire_contract_accepts_an_agent_epoch():
    params = DisplayLeaseAcquireParams(viewer_id='human', if_agent_epoch=7)
    assert params.if_agent_epoch == 7


def test_contract_rejects_negative_or_noninteger_epochs():
    import pytest
    from pydantic import ValidationError
    for invalid in [-1, True, '7', 1.5]:
        with pytest.raises(ValidationError):
            DisplayLeaseAcquireParams(viewer_id='human', if_agent_epoch=invalid)


def test_rpc_forwards_conditional_epoch_and_keeps_other_viewers_lease(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from tools.bot_desktop import lease, runtime
    home = tmp_path/'profile'
    home.mkdir()
    monkeypatch.setattr(server, '_profile_home', lambda name: str(home) if name == 'probe' else None)
    monkeypatch.setattr(runtime, 'rfb_socket_path', lambda: home/'bot-desktop'/'rfb.sock')

    def rpc(method, **params):
        result = server.handle_request({'jsonrpc': '2.0', 'id': 7, 'method': method, 'params': {'profile': 'probe', **params}})
        assert 'error' not in result, result
        return result['result']

    viewer = rpc('display.observe')['viewer_id']
    first = lease.get(profile_key=str(home))
    other = lease.acquire('other-human', profile_key=str(home))
    refused = rpc('display.lease.acquire', viewer_id=viewer, if_agent_epoch=first.epoch)
    assert refused['lease']['holder'] == lease.HUMAN
    assert lease.get(profile_key=str(home)) == other
    # The same declared RPC remains an explicit reclaim when no condition is supplied.
    rpc('display.lease.acquire', viewer_id=viewer)
    assert lease.get(profile_key=str(home)).viewer_id == viewer


def test_runtime_snapshot_announces_conditional_acquire_but_legacy_contract_defaults_off(tmp_path, monkeypatch):
    import tui_gateway.server as server
    from tui_gateway.contracts.display import DisplayStatus
    home = tmp_path/'profile'
    home.mkdir()
    monkeypatch.setattr(server, '_profile_home', lambda name: str(home) if name == 'capability' else None)
    result = server.handle_request({'jsonrpc': '2.0', 'id': 8, 'method': 'display.status', 'params': {'profile': 'capability'}})
    assert 'error' not in result, result
    status = result['result']
    assert status.get('conditional_acquire') is True
    legacy = {key: value for key, value in status.items() if key != 'conditional_acquire'}
    assert DisplayStatus.model_validate(legacy).conditional_acquire is False
