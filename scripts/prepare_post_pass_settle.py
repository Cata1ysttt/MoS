"""Prepare a metadata-only delayed post-pass backward-goal controller."""
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
    p.add_argument('--trigger',choices=['contact','valid_kick'],default='contact')
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    source_sha=sha(a.source);assert source_sha=='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'
    header=torch.load(a.source,map_location='cpu',weights_only=False);header['post_pass_settle_delay_steps']=10
    if a.trigger!='contact':header['post_pass_settle_trigger']=a.trigger
    rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    source=load_soccer_policy(a.source,'cuda:0');after=torch.random.get_rng_state();gpu_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(gpu_rng);candidate=load_soccer_policy(header,'cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(gpu_after,torch.cuda.get_rng_state())
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:8] for v in validation.values()]).cuda();checks={}
    def equal(observed):
        actual=candidate(observed);expected=source(observed)
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
    def standing(observed):
        goal=candidate.initial_goal.goal
        raw=candidate.actor(observed[:,:64],observed[:,64:93],observed[:,93:930],goal);scaled=(6*raw).clamp(-6,6)
        return dict(latent=goal,action_normalized=raw,action_scaled=scaled,joint_targets=candidate.default+scaled*candidate.scale)
    with torch.inference_mode():
        x=source.condition_observation(x,skill='shoot')
        for kicked in [False,True]:
            x[:,940]=float(kicked)
            for _ in range(12):equal(x);assert not candidate.last_settle_active.any()
        checks['native_shoot_outputs_exact']=True
        candidate.reset();source.reset();x=source.condition_observation(x,skill='pass');x[:,940]=0;equal(x)
        x[:,940]=1
        if a.trigger=='valid_kick':
            x[:,936:938]=-x[:,930:932]  # Contact with a stationary ball is not a valid pass.
            for _ in range(20):equal(x);assert not candidate.last_settle_active.any()
            checks['weak_contact_keeps_original_policy']=True
            x[:,936:938]=-x[:,930:932];x[:,936]+=.5
        for tick in range(10):equal(x);assert not candidate.last_settle_active.any()
        checks['passing_through_first_ten_contact_steps_exact']=True
        for _ in range(40):
            before=x.clone();actual=candidate(x);source(x);expected=standing(x)
            assert torch.equal(x,before) and candidate.last_settle_active.all()
            assert all(torch.equal(actual[k],expected[k]) for k in expected)
            assert candidate.solver_records[-1]['standing_recovery_environments']==0
        checks['delayed_passing_outputs_equal_initial_goal_actor']=True
        x=source.condition_observation(x,skill='shoot');x[:,940]=0
        for _ in range(51):equal(x);assert not candidate.last_settle_active.any()
        checks['new_ball_shooting_outputs_exact']=True
        x[:,940]=1
        for _ in range(12):equal(x);assert not candidate.last_settle_active.any()
        x=source.condition_observation(x,skill='pass');equal(x)
        assert not candidate.last_settle_active.any() and candidate.solver_records[-1]['standing_recovery_environments']==len(x)
        checks['post_shoot_recovery_preserved']=True
        candidate.reset();source.reset();x[:,940]=0;equal(x);x[:,940]=1
        for _ in range(11):candidate(x);source(x)
        mask=torch.arange(len(x),device=x.device)%2==0
        candidate.reset(mask);source.reset(mask);x[mask,940]=0
        actual=candidate(x);baseline=source(x);wanted=standing(x)
        assert torch.equal(candidate.last_settle_active,~mask)
        for key in actual:
            assert torch.equal(actual[key][mask],baseline[key][mask])
            assert torch.equal(actual[key][~mask],wanted[key][~mask])
        checks['mixed_rows_and_partial_reset_correct']=True
    original=source.state_dict();other=candidate.state_dict();assert original.keys()==other.keys()
    for key,value in original.items():assert torch.equal(value,other[key]),key
    assert all(not p.requires_grad and p.grad is None for p in candidate.parameters())
    assert sha(a.source)==source_sha
    a.output.mkdir(parents=True);torch.save(header,a.output/'diagnostic.pt')
    candidate.reset();restored=load_soccer_policy(a.output/'diagnostic.pt','cuda:0')
    with torch.inference_mode():
        actual=candidate(x);expected=restored(x)
        assert all(torch.equal(actual[k],expected[k]) for k in actual)
    checks['checkpoint_roundtrip_exact']=True
    result=dict(source_sha256=source_sha,checkpoint_sha256=sha(a.output/'diagnostic.pt'),checks=checks,
        frozen_parameters_unchanged=True,loading_rng_matches_source=True,expert_targets_used=False,real_states=len(x),
        post_pass_settle_trigger=a.trigger,post_pass_settle_delay_steps=10,steps=header['predicted_target_solver_steps'],backend=header['predicted_target_solver_backend'],
        scope='Historical states with synthetic phase flags. Passing behavior after the follow-through interval intentionally changes; no closed-loop performance claim.')
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
