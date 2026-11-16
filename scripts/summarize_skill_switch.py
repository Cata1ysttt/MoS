"""Compare completed paired switch probes without claiming soccer task success."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
reference=None
results=[]
for path in a.runs:
    summary=json.loads((path/'summary.json').read_text())
    key=tuple(summary[name] for name in ['checkpoint_sha256','seed','steps','requested'])+(summary['switch']['tick'],summary['switch']['state_before_switch_sha256'])
    if reference is None:reference=key
    assert key==reference,'Different policy, horizon, seed, batch, or physical/history prefix'
    assert summary['one_policy_instance'] and not summary['expert_loaded']
    assert summary['switch']['physical_and_history_state_preserved']
    assert not summary['task_success_measured']
    alive=summary['switch']['alive_before_switch']
    resets=summary['resets_per_environment']
    assert len(alive)==len(resets)==summary['requested']
    assert summary['continuous_survivors']==sum(count==0 for count in resets)
    terms=collections.Counter()
    first_terms=collections.Counter()
    first_reset_environments=set()
    for event in summary.get('reset_events',[]):
        if event['tick']>=summary['switch']['tick']:
            terms.update(name for name,flag in event['terms'].items() if flag)
            if alive[event['environment']] and event['environment'] not in first_reset_environments:
                first_terms.update(name for name,flag in event['terms'].items() if flag)
                first_reset_environments.add(event['environment'])
    results.append(dict(run=str(path),summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest(),
        transition_steps=summary.get('transition_steps',0),command_changed=summary['switch']['applied'],
        requested=summary['requested'],alive_at_switch=sum(alive),continuous_survivors=summary['continuous_survivors'],
        continuous_fraction=summary['continuous_survivors']/summary['requested'],
        post_switch_reset_terms=dict(terms),first_reset_terms_for_lanes_alive_at_switch=dict(first_terms),first_reset_environments=len(first_reset_environments),termination_reasons_recorded='reset_events' in summary,
        latent_step_norm_mean=statistics.mean(summary['switch']['latent_step_norm']),
        joint_target_step_rms_mean=statistics.mean(summary['switch']['joint_target_step_rms'])))
report=dict(checkpoint_sha256=reference[0],seed=reference[1],steps=reference[2],num_envs=reference[3],switch_step=reference[4],
    shared_state_history_prefix_sha256=reference[5],results=results,
    note='Shared-scene continuous-call diagnostic. Resets invalidate a lane, including task terminations that are not falls. Neither accuracy nor completion of two consecutive soccer tasks is measured.')
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
