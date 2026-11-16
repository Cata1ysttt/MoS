"""Prepare and smoke-test a diagnostic controller using internal target inversion."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import SoccerPolicy,load_soccer_policy
from fiber_soccer.predicted_target_solver import PredictedTargetSolverPolicy


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=64,choices=[4,8,16,32,64])
    p.add_argument('--backend',choices=['eager','cuda_graph'],default='eager')
    p.add_argument('--scope',choices=['all','initial_pass'],default='all')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    source_hash=sha(a.source)
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    assert header['planner_arch']=='target_guided_latent_v1' and not header['config']['smoke_test']
    header.update(predicted_target_solver_steps=a.steps,predicted_target_solver_backend=a.backend,diagnostic_source_sha256=source_hash,
        diagnostic_scope='Internal predicted-target latent optimization; no expert inputs; real-time capability unproven')
    if a.scope!='all':header['predicted_target_refinement_scope']=a.scope
    torch.save(header,a.output/'diagnostic.pt')
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:16] for v in validation.values()]).cuda()
    rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
    ordinary=SoccerPolicy({k:v for k,v in header.items() if k not in ['predicted_target_solver_steps','predicted_target_solver_backend','predicted_target_refinement_scope']},'cuda:0');after=torch.random.get_rng_state();cuda_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(cuda_rng)
    policy=load_soccer_policy(a.output/'diagnostic.pt','cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(cuda_after,torch.cuda.get_rng_state())
    reference=load_soccer_policy(dict(header,predicted_target_solver_backend='eager'),'cuda:0') if a.backend=='cuda_graph' else None
    with torch.inference_mode():
        baseline=ordinary(x);result=policy(x)
        assert torch.allclose(result['latent'].norm(dim=-1),torch.full((len(x),),16.,device=x.device),atol=3e-6)
        assert torch.isfinite(result['joint_targets']).all()
        commanded=policy(x,skill=x[:,-2:].argmax(-1),target_direction=x[:,-6:-3],target_speed=x[:,-3])
        assert torch.equal(commanded['latent'],result['latent'])
        assert torch.equal(commanded['joint_targets'],result['joint_targets'])
        if a.backend=='cuda_graph':
            expected=reference(x)
            assert torch.equal(expected['latent'],result['latent'])
            assert torch.equal(expected['joint_targets'],result['joint_targets'])
    smoke_record=dict(policy.solver_records[-1]);scope_checks={}
    if a.scope=='initial_pass':
        full=load_soccer_policy({k:v for k,v in header.items() if k!='predicted_target_refinement_scope'},'cuda:0')
        prior=SoccerPolicy(Path(header['config']['base']),'cuda:0')
        with torch.inference_mode():
            passing=x[:16];shooting=x[16:]
            policy.reset()
            for observed,expected,name in [(passing,full,'initial_pass'),(shooting,prior,'shoot'),(passing,prior,'recovery')]:
                actual=policy(observed);wanted=expected(observed)
                assert torch.equal(actual['latent'],wanted['latent'])
                assert torch.equal(actual['joint_targets'],wanted['joint_targets'])
                scope_checks[name+'_outputs_exact']=True
            policy.reset(torch.ones(len(passing),device=x.device,dtype=torch.bool))
            assert torch.equal(policy(passing)['joint_targets'],full(passing)['joint_targets'])
            scope_checks['reset_restores_refinement']=True
            policy.reset();mixed=torch.cat((passing[:8],shooting[:8]))
            actual=policy(mixed);all_output=full(mixed);prior_output=prior(mixed)
            for key in ['latent','joint_targets']:
                assert torch.equal(actual[key][:8],all_output[key][:8])
                assert torch.equal(actual[key][8:],prior_output[key][8:])
            scope_checks['mixed_rows_exact']=True
    assert smoke_record['final_mse']<smoke_record['initial_mse']
    assert all(p.grad is None for p in policy.parameters())
    for k,v in ordinary.state_dict().items():assert torch.equal(v,policy.state_dict()[k]),k
    assert sha(a.source)==source_hash
    complete=dict(source_sha256=source_hash,checkpoint_sha256=sha(a.output/'diagnostic.pt'),steps=a.steps,backend=a.backend,
        smoke=smoke_record,refinement_scope=a.scope,scope_checks=scope_checks,frozen_parameters_unchanged=True,loading_rng_matches_source=True,
        expert_targets_used=False,explicit_commands_match_observation=True,
        reference_solver_outputs_exact=a.backend=='cuda_graph',scope=header['diagnostic_scope'])
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(complete),flush=True)


if __name__=='__main__':main()
