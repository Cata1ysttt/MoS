"""Separate pre-contact and post-contact failures without dropping failed lanes."""
import argparse
import json
from collections import Counter
from pathlib import Path
from audit_sequence_recovery import validate


def audit(run):
    result=json.loads((run/'summary.json').read_text())
    accounting=validate(result)
    contract=json.loads((run/'scene_contracts.json').read_text())['shared']
    dt=contract['sim_dt']*contract['decimation']
    assert result['initial_skill']=='pass'
    passes={int(k):v for k,v in result['pass_impacts'].items()}
    shots={int(k):v for k,v in result['shoot_impacts'].items()}
    resets={row['environment']:row for row in result['first_resets']}
    handoffs={i:row for row in result['switches'] for i in row['environments']}
    rows=[]
    for i in range(result['requested']):
        phase=('after_shot' if i in shots else 'after_handoff_before_shot'
               if i in handoffs else 'after_pass_before_handoff' if i in passes else 'before_pass')
        end=resets[i]['tick'] if i in resets else result['steps']
        row=dict(environment=i,phase_at_end=phase,failed=i in resets,end_tick=end,
                 termination_terms=resets[i]['terms'] if i in resets else {},
                 time_after_shot_s=(end-shots[i]['tick'])*dt if i in shots else None,
                 post_shot_observation_censored=i in shots and i not in resets)
        if i in handoffs:
            switch=handoffs[i];idx=switch['environments'].index(i)
            row['handoff']={key:switch[key][idx] for key in
                ['root_speed_xy','root_tilt_rad','distance_to_robot']}
        rows.append(row)
    failed=Counter(row['phase_at_end'] for row in rows if row['failed'])
    survivors=Counter(row['phase_at_end'] for row in rows if not row['failed'])
    assert sum(failed.values())+sum(survivors.values())==result['requested']
    return dict(run=str(run),checkpoint_sha256=result['checkpoint_sha256'],
        requested=result['requested'],failed_by_phase=dict(failed),survivors_by_phase=dict(survivors),
        rows=rows,accounting=accounting,
        scope='Descriptive first-reset timing. Post-shot survivors are right-censored at the horizon. Groups have different trajectories; timing differences do not establish a causal recovery effect.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    results=[audit(run) for run in a.runs]
    with a.output.open('x') as f:json.dump(results,f,indent=2)
    for result in results:
        print(result['run'],result['failed_by_phase'],result['survivors_by_phase'])
