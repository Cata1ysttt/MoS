"""Prepare a metadata-only standing-recovery candidate and audit real-state routing."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import load_soccer_policy


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    source_sha=sha(a.source)
    assert source_sha=='9832d86c0d9efa6e028262ff347d7512082c2a2ef976d2294a2be8bdf484f8ba'
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    assert header['predicted_target_refinement_scope']=='initial_pass' and 'standing_recovery_mode' not in header
    header['standing_recovery_mode']='initial_backward_goal'
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:8] for v in validation.values()]).cuda()
    rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    source=load_soccer_policy(a.source,'cuda:0')
    after=torch.random.get_rng_state();gpu_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(gpu_rng)
    candidate=load_soccer_policy(header,'cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(gpu_after,torch.cuda.get_rng_state())
    checks={}
    with torch.inference_mode():
        for kicked,skill,name in [(False,'pass','initial_pass'),(True,'pass','ordinary_pass_contact'),
            (False,'shoot','shoot_approach'),(True,'shoot','shoot_contact')]:
            x[:,940]=float(kicked)
            actual=candidate(x,skill=skill);expected=source(x,skill=skill)
            assert all(torch.equal(actual[k],expected[k]) for k in expected),name
            checks[name+'_outputs_exact']=True
        original_goal=candidate.initial_goal.goal.clone()
        skill=torch.ones(len(x),device=x.device,dtype=torch.long);skill[:len(x)//2]=0
        actual=candidate(x,skill=skill);expected=source(x,skill=skill)
        active=skill==0
        assert torch.equal(actual['latent'][active],original_goal[active])
        raw=candidate.actor(x[:,:64],x[:,64:93],x[:,93:930],original_goal)
        assert torch.equal(actual['action_normalized'][active],raw[active])
        assert all(torch.equal(actual[k][~active],expected[k][~active]) for k in expected)
        checks['mixed_recovery_uses_initial_goal_and_preserves_shoot']=True
    candidate.reset(active);source.reset(active)
    # Reset lanes start ordinary passing; non-reset lanes keep their shooting contact origin.
    actual=candidate(x,skill='pass');expected=source(x,skill='pass')
    assert all(torch.equal(actual[k][active],expected[k][active]) for k in expected)
    assert torch.equal(actual['latent'][~active],original_goal[~active])
    checks['partial_reset_and_inference_context_change']=True
    candidate.reset();source.reset();x[:,940]=0
    actual=candidate(x,skill='pass');expected=source(x,skill='pass')
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    checks['full_reset_restores_base']=True
    state=source.state_dict();other=candidate.state_dict();assert state.keys()==other.keys()
    for k,v in state.items():assert torch.equal(v,other[k]),k
    assert all(p.grad is None and not p.requires_grad for p in candidate.parameters())
    assert sha(a.source)==source_sha
    a.output.mkdir(parents=True);torch.save(header,a.output/'diagnostic.pt')
    complete=dict(source_sha256=source_sha,checkpoint_sha256=sha(a.output/'diagnostic.pt'),
        steps=header['predicted_target_solver_steps'],backend=header['predicted_target_solver_backend'],
        refinement_scope='initial_pass',standing_recovery_mode=header['standing_recovery_mode'],
        frozen_parameters_unchanged=True,loading_rng_matches_source=True,expert_targets_used=False,
        checks=checks,real_states=len(x),scope='Real historical states with explicit synthetic phase flags; routing proof only, not closed-loop recovery success')
    (a.output/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(complete),flush=True)

if __name__=='__main__':main()
