"""Diagnose whether internally predicted targets can be realized by frozen BFM.

Per-state latent optimization is diagnostic only. Expert targets are read solely
for reporting; optimization targets come from the frozen shared predictor.
No optimized latent is exported as a deployable policy or a training dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.latent_oracle import fit_latent,SOLVER_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4);started=time.monotonic()
    checksum=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert header['planner_arch']=='target_guided_latent_v1' and not header['config']['smoke_test']
    paths=[Path(row['path']) for row in header['config']['data']]
    _,validation,provenance=load_guidance_data(paths,smoke=True)
    assert [(v['path'],v['sha256']) for v in provenance]==[(v['path'],v['sha256']) for v in header['config']['data']]
    policy=SoccerPolicy(header,a.device)
    weights=torch.ones(29,device=a.device);weights[:12]=2
    root_weight=(weights/weights.mean()).sqrt()
    def physical(raw):return policy.default+(6*raw).clamp(-6,6)*policy.scale
    def weighted(error):return (error.square()*weights).sum(-1)/weights.sum()
    result={}
    for skill,(x,expert,teacher) in validation.items():
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==header['config']['validation_state_sha256'][skill]
        metrics={key:[] for key in ['before_mapping_mse','after_mapping_mse','before_expert_mse','after_expert_mse','latent_rms_change']}
        for begin in range(0,len(x),128):
            xb=x[begin:begin+128].to(a.device);expertb=expert[begin:begin+128].to(a.device)
            with torch.no_grad():
                initial=policy.planner.act(xb)
                desired=policy.planner.target_predictor.bounded_targets(xb)
                before=physical(policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],initial))
            solved,info=fit_latent(policy.actor,(xb[:,:64],xb[:,64:93],xb[:,93:930]),desired*root_weight,
                initial,lambda raw:physical(raw)*root_weight,iterations=64)
            with torch.no_grad():
                after=physical(policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],solved))
                metrics['before_mapping_mse'].append(weighted(before-desired).cpu())
                metrics['after_mapping_mse'].append(weighted(after-desired).cpu())
                metrics['before_expert_mse'].append(weighted(before-expertb).cpu())
                metrics['after_expert_mse'].append(weighted(after-expertb).cpu())
                metrics['latent_rms_change'].append((solved-initial).square().mean(-1).sqrt().cpu())
        values={key:torch.cat(v) for key,v in metrics.items()};groups={}
        for label,mask in [('all',torch.ones(len(x),dtype=torch.bool)),('teacher_executed',teacher.bool()),('student_executed',~teacher.bool())]:
            if not mask.any():continue
            groups[label]=dict(states=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in values.items()},
                maximum_remaining_mapping_mse=float(values['after_mapping_mse'][mask].max()),
                fraction_mapping_mse_below_1e_4=float((values['after_mapping_mse'][mask]<1e-4).float().mean()))
        result[skill]=groups;print(json.dumps({skill:groups}),flush=True)
    assert all(v.grad is None for v in policy.parameters())
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checksum
    report=dict(checkpoint_sha256=checksum,iterations=64,learning_rate=.05,solver_version=SOLVER_VERSION,
        gradient_reduction='sum_per_state',leg_weight=2,optimization_targets='Frozen predictor outputs, not expert actions',
        state_hashes=header['config']['validation_state_sha256'],results=result,elapsed=time.monotonic()-started,
        scope='Per-state feasibility diagnostic on historical validation observations. Not autonomous rollout, deployable policy, or training labels; expert targets only used to score resulting actions.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
