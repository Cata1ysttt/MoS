"""Replay measured first-shot errors through the actual continuous reward code.

This audits the impact component only, not cumulative return, PPO advantage,
or a causal explanation of policy performance. No simulation or training runs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from fiber_soccer.continuous_shooting import ContinuousShootingPhase
from fiber_soccer.paths import ROOT


def replay(paths):
    records=[]
    for path in paths:
        summary=json.loads(path.read_text())
        assert summary['failed_lane_control']=='zero' and not summary['expert_loaded']
        for environment,row in summary['shoot_impacts'].items():
            records.append(dict(seed=summary['seed'],environment=int(environment),**row))
    assert records
    n=len(records);phase=ContinuousShootingPhase(n,'cpu');phase.start_shooting(torch.arange(n),torch.zeros(n))
    velocity=torch.tensor([[r['speed_mps']*math.cos(math.radians(r['direction_error_deg'])),
                            r['speed_mps']*math.sin(math.radians(r['direction_error_deg']))] for r in records])
    direction=torch.tensor([[1.,0.]]*n);speed=torch.tensor([r['target_speed_mps'] for r in records])
    preferred=torch.tensor([r['preferred_foot'] for r in records]);yes=torch.ones(n,dtype=torch.bool);no=~yes
    phase.observe(velocity,direction,speed,yes,preferred,yes,torch.ones(n,dtype=torch.long))
    terms=phase.rewards(torch.zeros(n),torch.zeros(n),torch.zeros(n),no,no,.02)
    for i,row in enumerate(records):
        expected=math.exp(-(row['direction_error_deg']/10)**2-(row['relative_forward_speed_error']/.3)**2)*row['preferred_foot']
        assert abs(float(phase.quality[i])-expected)<1e-5
        accurate=row['preferred_foot'] and row['direction_error_deg']<=10 and row['relative_forward_speed_error']<=.3
        assert bool(phase.accurate[i])==accurate
        row.update(accurate=accurate,quality=float(phase.quality[i]),impact_reward=float(terms[i,1]),
            preferred_foot_reward=2.*row['preferred_foot'],quality_reward=float(10*phase.quality[i]),accuracy_bonus=5.*accurate)
    def stats(selected):
        if not selected:return dict(count=0)
        values=[r['impact_reward'] for r in selected]
        return dict(count=len(selected),mean_impact_reward=sum(values)/len(values),minimum_impact_reward=min(values),
            maximum_impact_reward=max(values),mean_quality=sum(r['quality'] for r in selected)/len(selected))
    bins={}
    for lo,hi in [(0,10),(10,20),(20,40),(40,180)]:
        bins[f'preferred_foot_angle_{lo}_{hi}']=stats([r for r in records if r['preferred_foot'] and
            (r['direction_error_deg']>=lo if lo==0 else r['direction_error_deg']>lo) and r['direction_error_deg']<=hi])
    return dict(summary_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        shots=len(records),accurate=stats([r for r in records if r['accurate']]),
        inaccurate_preferred_foot=stats([r for r in records if not r['accurate'] and r['preferred_foot']]),
        wrong_foot=stats([r for r in records if not r['preferred_foot']]),
        direction_only_10_to_20_deg=stats([r for r in records if r['preferred_foot'] and r['relative_forward_speed_error']<=.3 and 10<r['direction_error_deg']<=20]),
        angle_bins=bins,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,nargs='+',required=True)
    p.add_argument('--candidate',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
    result={name:replay(paths) for name,paths in [('baseline',a.baseline),('candidate',a.candidate)]}
    result.update(reward_implementation_sha256=hashlib.sha256((ROOT/'src/fiber_soccer/continuous_shooting.py').read_bytes()).hexdigest(),
        impact_bounds=dict(accurate_minimum=7+10*math.exp(-2),accurate_maximum=17,
            inaccurate_preferred_foot_supremum=2+10*math.exp(-1),wrong_foot=0),
        scope='Fixed observed first-shot velocity errors replayed through the unchanged reward implementation. Impact component only; excludes failure/progress/recovery/timeout terms. Does not measure learning gradients or prove a causal bottleneck.')
    for group in ['baseline','candidate']:
        print(group,json.dumps({k:result[group][k] for k in ['shots','accurate','inaccurate_preferred_foot','wrong_foot','direction_only_10_to_20_deg','angle_bins']},indent=2))
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
