"""Check CUDA graph refinement against the reference solver on real AH-BFM states."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.predicted_target_solver import refine
from fiber_soccer.captured_latent_solver import CapturedLatentSolver,fixed_adam


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64,choices=[4,8,16,32,64])
    a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    checksum=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    _,validation,_=load_guidance_data([Path(v['path']) for v in header['config']['data']],smoke=True)
    policy=SoccerPolicy(header,'cuda:0')
    x=torch.cat([v[0][torch.linspace(0,len(v[0])-1,64).long()] for v in validation.values()]).cuda()
    weight=torch.ones(29,device='cuda:0');weight[:12]=2;root=(weight/weight.mean()).sqrt()
    def physical(raw):return policy.default+(6*raw).clamp(-6,6)*policy.scale
    solver=CapturedLatentSolver(policy.actor,lambda raw:physical(raw)*root,a.iterations)
    result={}
    with torch.inference_mode():
        for n in [1,16]:
            errors=[];previous_output=None;preserved=None
            for offset in [0,64,32,96]:
                xb=x[offset:offset+n];z=policy.planner.act(xb);target=policy.planner.target_predictor.bounded_targets(xb)
                reference,info=refine(policy.actor,xb,z,target,policy.default,policy.scale,a.iterations)
                rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
                actual,fast_info=solver.solve((xb[:,:64],xb[:,64:93],xb[:,93:930]),target*root,z)
                assert torch.equal(rng,torch.random.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
                with torch.inference_mode(False):
                    eager,*_=fixed_adam(policy.actor,tuple(v.clone() for v in (xb[:,:64],xb[:,64:93],xb[:,93:930])),
                        (target*root).clone(),z.clone(),lambda raw:physical(raw)*root,a.iterations)
                torch.testing.assert_close(actual,reference,atol=3e-5,rtol=1e-4)
                eager_difference=float((eager-reference).abs().max())
                capture_difference=float((actual-eager).abs().max())
                if previous_output is not None:assert torch.equal(previous_output,preserved)
                previous_output=actual;preserved=actual.clone()
                raw=policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],actual)
                raw_ref=policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],reference)
                errors.append(dict(eager_reference_maximum_latent_difference=eager_difference,capture_eager_maximum_latent_difference=capture_difference,
                    strict_latent_close=bool(torch.allclose(actual,reference,atol=3e-5,rtol=1e-4)),maximum_latent_difference=float((actual-reference).abs().max()),
                    maximum_physical_difference=float((physical(raw)-physical(raw_ref)).abs().max()),
                    reference=info,captured=fast_info))
            times=[]
            # Include the same prior, target prediction, actor decoding and synchronization as the diagnostic policy.
            for repeat in range(22):
                xb=x[(repeat%4)*16:(repeat%4)*16+n]
                torch.cuda.synchronize();begin=time.perf_counter()
                initial=policy(xb);target=policy.planner.target_predictor.bounded_targets(xb)
                z,_=solver.solve((xb[:,:64],xb[:,64:93],xb[:,93:930]),target*root,initial['latent'])
                output=physical(policy.actor(xb[:,:64],xb[:,64:93],xb[:,93:930],z))
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                assert torch.isfinite(output).all()
                if repeat>=2:times.append(elapsed)
            result[str(n)]=dict(equivalence_checks=errors,calls=len(times),median_seconds=float(torch.tensor(times).median()),
                p95_seconds=float(torch.quantile(torch.tensor(times),.95)),maximum_seconds=max(times))
            print(json.dumps({n:result[str(n)]}),flush=True)
    assert all(v.grad is None for v in policy.parameters()) and sha(a.checkpoint)==checksum
    report=dict(source_sha256=checksum,iterations=a.iterations,device=torch.cuda.get_device_name(),results=result,
        scope='Real-state numerical differences and isolated timing; no equivalence or closed-loop guarantee inferred from small errors')
    a.output.write_text(json.dumps(report,indent=2)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':main()
