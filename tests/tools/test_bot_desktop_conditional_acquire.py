"""An automatic handoff is conditional on the exact agent-held lease observed."""
from tools.bot_desktop import lease


def test_automatic_acquire_never_evicts_a_human_who_won_after_the_read(tmp_path):
    home = str(tmp_path)
    before = lease.get(profile_key=home)
    other = lease.acquire('other-human', profile_key=home)
    after = lease.acquire('automatic-click', profile_key=home, if_agent_epoch=before.epoch)
    assert after == other
    assert lease.get(profile_key=home).viewer_id == 'other-human'


def test_matching_agent_epoch_acquires_but_an_intervening_handoff_invalidates_it(tmp_path):
    home = str(tmp_path)
    epoch = lease.get(profile_key=home).epoch
    acquired = lease.acquire('automatic', profile_key=home, if_agent_epoch=epoch)
    assert acquired.holder == lease.HUMAN and acquired.viewer_id == 'automatic'
    assert acquired.epoch == epoch + 1
    after_release = lease.release('automatic', profile_key=home)
    stale = lease.acquire('stale-click', profile_key=home, if_agent_epoch=epoch)
    assert stale == after_release


def test_conditional_epoch_is_profile_local_and_manual_reclaim_remains_explicit(tmp_path):
    a, b = str(tmp_path/'a'), str(tmp_path/'b')
    epoch_a = lease.get(profile_key=a).epoch
    epoch_b = lease.get(profile_key=b).epoch
    lease.acquire('human-a', profile_key=a)
    in_b = lease.acquire('automatic-b', profile_key=b, if_agent_epoch=epoch_b)
    assert in_b.holder == lease.HUMAN and in_b.viewer_id == 'automatic-b'
    assert lease.acquire('automatic-a', profile_key=a, if_agent_epoch=epoch_a).viewer_id == 'human-a'
    manual = lease.acquire('explicit-reclaim', profile_key=a)
    assert manual.viewer_id == 'explicit-reclaim'
    assert lease.get(profile_key=b) == in_b


def test_two_processes_cannot_both_win_the_same_observed_agent_epoch(tmp_path):
    import os
    import subprocess
    import sys
    probe = "from tools.bot_desktop import lease; import sys; print(lease.acquire(sys.argv[1], profile_key=sys.argv[2], if_agent_epoch=0).viewer_id)"
    jobs = [subprocess.Popen([sys.executable, '-c', probe, viewer, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, 'HERMES_HOME': str(tmp_path)}) for viewer in ['click-a', 'click-b']]
    results = []
    for job in jobs:
        out, error = job.communicate(timeout=30)
        assert job.returncode == 0, error
        results.append(out.strip())
    assert results[0] == results[1]
    final = lease.get(profile_key=str(tmp_path))
    assert final.viewer_id == results[0]
    assert final.epoch == 1
