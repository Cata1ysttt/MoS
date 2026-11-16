"""Measure internal-target solver quality versus steps and observed call latency."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.predicted_target_solver import PredictedTargetSolverPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4);started=time.monotonic()
    checksum=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    _,validation,provenance=load_guidance_data([Path(d['path']) for d in header['config']['data']],smoke=True)
    assert provenance==header['config']['data']
    for skill,(x,_,_) in validation.items():
        assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==header['config']['validation_state_sha256'][skill]
    probes=torch.cat([v[0][:8] for v in validation.values()]).cuda()
    weight=torch.ones(29,device='cuda:0');weight[:12]=2
    result={}
    for steps in [0,4,8,16,32,64]:
        policy=PredictedTargetSolverPolicy(dict(header,predicted_target_solver_steps=steps),'cuda:0') if steps else SoccerPolicy(header,'cuda:0')
        latency={}
        with torch.inference_mode():
            for size in [1,16]:
                for _ in range(2):policy(probes[:size])
                elapsed=[];updates=[]
                for repeat in range(20):
                    xb=probes[repeat%16:repeat%16+1] if size==1 else probes
                    torch.cuda.synchronize();begin=time.perf_counter();policy(xb);torch.cuda.synchronize()
                    elapsed.append(time.perf_counter()-begin)
                    if steps:updates.append(policy.solver_records[-1]['updates'])
                latency[str(size)]=dict(calls=20,median_seconds=float(torch.tensor(elapsed).median()),
                    p95_seconds=float(torch.quantile(torch.tensor(elapsed),.95)),maximum_seconds=max(elapsed),
                    mean_updates=sum(updates)/len(updates) if updates else 0)
            quality={}
            # The 64-step full-holdout result already exists; retain it as a separate source.
            if steps<64:
                for skill,(x,expert,_) in validation.items():
                    mapping=[];target_error=[]
                    for begin in range(0,len(x),128):
                        xb=x[begin:begin+128].cuda();yb=expert[begin:begin+128].cuda()
                        desired=policy.planner.target_predictor.bounded_targets(xb)
                        physical=policy(xb)['joint_targets']
                        mapping.append((((physical-desired).square()*weight).sum(-1)/weight.sum()).cpu())
                        target_error.append((((physical-yb).square()*weight).sum(-1)/weight.sum()).cpu())
                    mapping=torch.cat(mapping);target_error=torch.cat(target_error)
                    quality[skill]=dict(states=len(x),mapping_mse=float(mapping.mean()),expert_mse=float(target_error.mean()),
                        fraction_mapping_below_1e_4=float((mapping<1e-4).float().mean()))
        assert all(p.grad is None for p in policy.parameters())
        result[str(steps)]=dict(latency=latency,historical_holdout=quality)
        print(json.dumps({steps:result[str(steps)]}),flush=True)
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==checksum
    report=dict(source_sha256=checksum,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        cpu_threads=4,results=result,elapsed=time.monotonic()-started,
        scope='Historical offline error and isolated GPU call latency; no fresh rollout or real-time system guarantee',
        full_64_step_quality_reference='reports/planner_target_guided_inversion.json')
    a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
