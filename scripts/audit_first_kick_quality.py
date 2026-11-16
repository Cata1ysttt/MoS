"""Compare paired first-observed kick velocities, separate from frozen quality metrics."""
import argparse
import json
import math
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--planner',type=Path,required=True)
p.add_argument('--expert',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--relative-speed-tolerance',type=float,default=.3)
a=p.parse_args()
assert math.isfinite(a.relative_speed_tolerance) and a.relative_speed_tolerance>=0
summaries=[json.loads((path/'summary.json').read_text()) for path in [a.planner,a.expert]]
assert summaries[0]['initial_state_sha256']==summaries[1]['initial_state_sha256']
assert summaries[0]['seed']==summaries[1]['seed']
assert summaries[0]['requested']==summaries[1]['requested']
results={}
measurements={}
for label,path,summary in zip(['planner','expert'],[a.planner,a.expert],summaries):
    assert summary['completed']==summary['requested'] and not summary['expert_assisted_oracle']
    measured=json.loads((path/'first_kick_velocity.json').read_text())
    samples=measured['samples'];n=summary['requested']
    measurements[label]=samples
    ids={int(i) for i in samples}
    assert ids|set(measured['missing_environment_ids'])==set(range(n))
    assert not ids&set(measured['missing_environment_ids'])
    rows=list(samples.values())
    metrics=['forward_speed_mps','direction_error_deg','relative_forward_speed_error','target_speed_mps']
    assert all(math.isfinite(row[key]) for row in rows for key in metrics)
    direction=sum(row['direction_error_deg']<=10 for row in rows)
    joint=sum(row['direction_error_deg']<=10 and row['relative_forward_speed_error']<=a.relative_speed_tolerance for row in rows)
    results[label]=dict(run=str(path),episodes=n,observed_first_kicks=len(rows),missing=n-len(rows),
        first_observed_direction_le_10deg=direction,first_observed_direction_and_speed=joint,
        medians_conditioned_on_observed_kick={key:statistics.median(row[key] for row in rows) if rows else None for key in metrics},
        native_frozen_direction_success_count=round(summary['quality_ball_direction_success_10deg']*n))
common=set(measurements['planner'])&set(measurements['expert'])
for key in common:
    left,right=measurements['planner'][key],measurements['expert'][key]
    assert math.isclose(left['target_speed_mps'],right['target_speed_mps'],abs_tol=1e-6,rel_tol=1e-6),'Paired speed commands differ'
    assert all(math.isclose(x,y,abs_tol=1e-6,rel_tol=1e-6) for x,y in zip(left['target_direction_xy'],right['target_direction_xy'])),'Paired target directions differ'
report=dict(paired_commands_checked_for_observed_kicks=len(common),seed=summaries[0]['seed'],initial_state_sha256=summaries[0]['initial_state_sha256'],
    relative_speed_tolerance=a.relative_speed_tolerance,results=results,
    note='Passive first observed kicked state with ball speed >=0.25m/s before first resample, up to one tick delay. All episode counts retain missing kicks in the denominator. Speed-plus-direction criterion is an additional diagnostic, not the original task metric; native frozen direction metrics use a different sampling time.')
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
