"""Create a metadata-only new-ball RMA candidate and verify actual policy routing."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_rma import initialize_new_ball_history


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['initialize_shoot_rma','initialize_shoot_ball_task'],default='initialize_shoot_rma')
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    source_sha=sha(a.source)
    assert source_sha=='7c98136ec5848fbf9ca67f7132f471548d0f4c098266c94df4b67cb993361fdc'
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    assert header['standing_recovery_mode']=='initial_backward_goal' and 'new_ball_history_mode' not in header
    header['new_ball_history_mode']=a.mode
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
        for skill in ['pass','shoot']:
            source.reset();candidate.reset()
            for kicked in [False,True]:
                x[:,940]=float(kicked)
                actual=candidate(x,skill=skill);expected=source(x,skill=skill)
                assert all(torch.equal(actual[k],expected[k]) for k in expected)
                assert candidate.solver_records[-1]['new_ball_history_environments']==0
            checks['native_'+skill+'_outputs_exact']=True
        source.reset();candidate.reset();x[:,940]=0
        for kicked in [False,True]:
            x[:,940]=float(kicked)
            actual=candidate(x,skill='pass');expected=source(x,skill='pass')
            assert all(torch.equal(actual[k],expected[k]) for k in expected)
        checks['initial_pass_outputs_exact']=True
        x=source.condition_observation(x,skill='shoot');x[:,940]=0
        anchor=torch.cat([x[:,933:935],torch.ones_like(x[:,933:934])],dim=1)
        task_anchor=torch.cat((x[:,933:939],x[:,2400:2404]),dim=-1)
        maximum_change=0.
        for age in range(51):
            before=x.clone()
            transformed=initialize_new_ball_history(x,torch.full((len(x),),age,device=x.device,dtype=torch.long),anchor)
            boundary=2230
            if a.mode=='initialize_shoot_ball_task':
                from fiber_soccer.new_ball_task_history import initialize_new_ball_task_history
                transformed=initialize_new_ball_task_history(transformed,torch.full((len(x),),age,device=x.device,dtype=torch.long),task_anchor)
                boundary=2130
            assert torch.equal(transformed[:,:boundary],x[:,:boundary]) and torch.equal(transformed[:,2380:],x[:,2380:])
            actual=candidate(x);expected=source(transformed)
            assert torch.equal(x,before)
            assert all(torch.equal(actual[k],expected[k]) for k in expected),age
            assert candidate.solver_records[-1]['new_ball_history_environments']==(len(x) if age<50 else 0)
            if a.mode=='initialize_shoot_ball_task':
                assert candidate.solver_records[-1]['new_ball_task_history_environments']==(len(x) if age<10 else 0)
            if age==0:
                # Shared prior is feedforward, so this compares only the input change.
                raw=source.planner.prior.act(x)
                maximum_change=float((actual['latent']-raw).abs().max())
        checks['new_ball_outputs_equal_manual_input_only_rewrite']=True
        checks['track_expires_after_50_calls']=True
        x[:,940]=1;candidate(x);source(x)
        actual=candidate(x,skill='pass');expected=source(x,skill='pass')
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        checks['standing_recovery_preserved']=True
    candidate.reset();source.reset();x=x.clone();x[:,940]=0
    actual=candidate(x,skill='pass');expected=source(x,skill='pass')
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    checks['reset_and_context_change_preserved']=True
    state=source.state_dict();other=candidate.state_dict();assert state.keys()==other.keys()
    for k,v in state.items():assert torch.equal(v,other[k]),k
    assert all(p.grad is None and not p.requires_grad for p in candidate.parameters())
    assert sha(a.source)==source_sha
    a.output.mkdir(parents=True);torch.save(header,a.output/'diagnostic.pt')
    result=dict(source_sha256=source_sha,checkpoint_sha256=sha(a.output/'diagnostic.pt'),
        steps=header['predicted_target_solver_steps'],backend=header['predicted_target_solver_backend'],
        refinement_scope='initial_pass',standing_recovery_mode=header['standing_recovery_mode'],
        new_ball_history_mode=header['new_ball_history_mode'],frozen_parameters_unchanged=True,
        loading_rng_matches_source=True,expert_targets_used=False,checks=checks,real_states=len(x),
        handoff_maximum_latent_change=maximum_change,
        scope='Historical states with synthetic phase flags, routing/input sensitivity only; no closed-loop performance claim')
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
