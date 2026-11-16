"""Prepare and smoke-test a diagnostic controller using internal target inversion."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.predicted_target_solver import PredictedTargetSolverPolicy


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=64,choices=[4,8,16,32,64])
    p.add_argument('--backend',choices=['eager','cuda_graph'],default='eager')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    source_hash=sha(a.source)
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    assert header['planner_arch']=='target_guided_latent_v1' and not header['config']['smoke_test']
    header.update(predicted_target_solver_steps=a.steps,predicted_target_solver_backend=a.backend,diagnostic_source_sha256=source_hash,
        diagnostic_scope='Internal predicted-target latent optimization; no expert inputs; real-time capability unproven')
    torch.save(header,a.output/'diagnostic.pt')
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:16] for v in validation.values()]).cuda()
    rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    ordinary=SoccerPolicy(header,'cuda:0');after=torch.random.get_rng_state();cuda_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
    policy=PredictedTargetSolverPolicy(torch.load(a.output/'diagnostic.pt',map_location='cpu',weights_only=False),'cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(cuda_after,torch.cuda.get_rng_state())
    with torch.inference_mode():
        baseline=ordinary(x);result=policy(x)
        assert torch.allclose(result['latent'].norm(dim=-1),torch.full((len(x),),16.,device=x.device),atol=3e-6)
        assert torch.isfinite(result['joint_targets']).all()
    assert policy.solver_records[-1]['final_mse']<policy.solver_records[-1]['initial_mse']
    assert all(p.grad is None for p in policy.parameters())
    for k,v in ordinary.state_dict().items():assert torch.equal(v,policy.state_dict()[k]),k
    assert sha(a.source)==source_hash
    complete=dict(source_sha256=source_hash,checkpoint_sha256=sha(a.output/'diagnostic.pt'),steps=a.steps,backend=a.backend,
        smoke=policy.solver_records[-1],frozen_parameters_unchanged=True,loading_rng_matches_source=True,
        expert_targets_used=False,scope=header['diagnostic_scope'])
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
