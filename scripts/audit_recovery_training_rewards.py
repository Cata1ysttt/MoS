"""Check recovery-only reward bounds, failure accounting and active PPO updates."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(run):
    config=json.loads((run/'config.json').read_text())
    assert config['training_task']=='shoot_recover'
    contract=json.loads((run/'physical_contract.json').read_text())
    dt=contract['sim_dt']*contract['decimation']
    batch=config['num_envs']*config['rollout_steps']
    text=(run/'metrics.jsonl').read_text()
    prefix=text[:text.rfind('\n')+1]  # ignore an in-progress append
    rows=[json.loads(line) for line in prefix.splitlines()]
    assert rows,'No complete training rows'
    active_total=entries=failures=actor_updates=0
    max_sum_error=max_failure_integer_error=0.
    inactive_rollouts=0
    for expected_iteration,row in enumerate(rows,1):
        assert row['iteration']==expected_iteration
        active=row['active_policy_steps'];terms=row['reward_terms']
        assert isinstance(active,int) and 0<=active<=batch
        active_total+=active
        assert row['total_active_policy_steps']==active_total
        assert row['recovery_entries']>=entries
        entries=row['recovery_entries']
        assert 0<=row['actor_gradient_updates']<=row['gradient_updates']<=config['epochs']*config['minibatches']
        actor_updates+=row['actor_gradient_updates']
        fraction=active/batch
        assert set(terms)=={'stability','action_rate','failure'}
        assert -1e-7<=terms['stability']<=5.*dt*fraction+1e-7
        assert -.05*dt*fraction-1e-7<=terms['action_rate']<=1e-7
        raw_failures=-terms['failure']*batch/(100.*dt)
        integer_error=abs(raw_failures-round(raw_failures))
        max_failure_integer_error=max(max_failure_integer_error,integer_error)
        assert integer_error<1e-4
        count=round(raw_failures)
        assert 0<=count<=row['terminated']
        failures+=count
        assert failures<=entries
        error=abs(sum(terms.values())-row['mean_training_reward'])
        max_sum_error=max(max_sum_error,error)
        assert error<1e-6 and abs(row['environment_reward_mean']-row['mean_training_reward'])<1e-6
        assert row['reference_reward_removed_mean']==0
        if active==0:
            inactive_rollouts+=1
            assert all(abs(value)<1e-8 for value in terms.values())
            assert row['actor_gradient_updates']==0
    complete=(run/'complete.json').exists()
    if complete:
        manifest=json.loads((run/'complete.json').read_text())
        assert manifest['active_policy_steps']==active_total and manifest['recovery_entries']==entries
        assert manifest['iterations']==len(rows)
    return dict(run=str(run),rows_audited=len(rows),run_completed=complete,
        metrics_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        active_policy_steps=active_total,recovery_entries=entries,
        recovery_failures=failures,actor_gradient_updates=actor_updates,
        zero_active_rollouts=inactive_rollouts,max_reward_sum_error=max_sum_error,
        max_failure_count_integer_error=max_failure_integer_error,
        scope='Aggregate reward/update accounting and bounds on the recorded prefix. Failure counts derive from the failure reward; they are training events, not autonomous task success or independent survival evaluation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
