"""Verify a post-pass pre-contact solver scope with fixed full-history/recovery weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_task_history import NewBallTaskHistory

SCOPE='initial_pass_and_post_pass_shoot'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    source_sha=sha(a.source)
    assert source_sha=='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    assert header['predicted_target_refinement_scope']=='initial_pass'
    assert header['new_ball_history_mode']=='initialize_shoot_ball_task'
    header['predicted_target_refinement_scope']=SCOPE
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:8] for v in validation.values()]).cuda()
    rng=torch.random.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    source=load_soccer_policy(a.source,'cuda:0')
    after=torch.random.get_rng_state();gpu_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(gpu_rng)
    candidate=load_soccer_policy(header,'cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(gpu_after,torch.cuda.get_rng_state())
    full_header={k:v for k,v in header.items() if k not in ('standing_recovery_mode','new_ball_history_mode','predicted_target_refinement_scope')}
    full=load_soccer_policy(full_header,'cuda:0')
    manual=NewBallTaskHistory().eval();checks={}
    with torch.inference_mode():
        for skill in ['pass','shoot']:
            source.reset();candidate.reset()
            for kicked in [False,True]:
                x[:,940]=float(kicked)
                actual=candidate(x,skill=skill);expected=source(x,skill=skill)
                assert all(torch.equal(actual[k],expected[k]) for k in expected)
                assert candidate.solver_records[-1]['post_pass_shoot_refined_environments']==0
            checks['native_'+skill+'_outputs_exact']=True
        source.reset();candidate.reset();manual.reset()
        for kicked in [False,True]:
            x=source.condition_observation(x,skill='pass');x[:,940]=float(kicked)
            actual=candidate(x);expected=source(x);manual(x)
            assert all(torch.equal(actual[k],expected[k]) for k in expected)
        checks['initial_pass_outputs_exact']=True
        x=source.condition_observation(x,skill='shoot');x[:,940]=0
        maximum_change=0.
        for age in range(51):
            before=x.clone();transformed=manual(x)
            actual=candidate(x);expected=full(transformed);baseline=source(x)
            assert torch.equal(x,before)
            assert all(torch.equal(actual[k],expected[k]) for k in expected),age
            assert candidate.solver_records[-1]['post_pass_shoot_refined_environments']==len(x)
            maximum_change=max(maximum_change,float((actual['latent']-baseline['latent']).abs().max()))
        checks['post_pass_shoot_equals_full_solver_on_corrected_inputs']=True
        x[:,940]=1;manual(x)
        actual=candidate(x);expected=source(x)
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        assert candidate.solver_records[-1]['post_pass_shoot_refined_environments']==0
        checks['shooting_contact_returns_to_prior']=True
        manual(source.condition_observation(x,skill='pass'))
        actual=candidate(x,skill='pass');expected=source(x,skill='pass')
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        checks['standing_recovery_preserved']=True
        # Reset half the batch; the remaining rows retain their completed pass.
        mask=torch.arange(len(x),device=x.device)%2==0
        candidate.reset(mask);source.reset(mask);manual.reset(mask)
        x=source.condition_observation(x,skill='shoot');x[:,940]=0
        actual=candidate(x);baseline=source(x);solved=full(manual(x))
        for key in actual:
            assert torch.equal(actual[key][mask],baseline[key][mask])
            assert torch.equal(actual[key][~mask],solved[key][~mask])
        assert candidate.solver_records[-1]['post_pass_shoot_refined_environments']==int((~mask).sum())
        checks['mixed_rows_and_partial_reset_preserved']=True
    state=source.state_dict();other=candidate.state_dict();assert state.keys()==other.keys()
    for k,v in state.items():assert torch.equal(v,other[k]),k
    assert all(p.grad is None and not p.requires_grad for p in candidate.parameters())
    assert sha(a.source)==source_sha
    a.output.mkdir(parents=True);torch.save(header,a.output/'diagnostic.pt')
    result=dict(source_sha256=source_sha,checkpoint_sha256=sha(a.output/'diagnostic.pt'),
        steps=header['predicted_target_solver_steps'],backend=header['predicted_target_solver_backend'],
        refinement_scope=SCOPE,standing_recovery_mode=header['standing_recovery_mode'],
        new_ball_history_mode=header['new_ball_history_mode'],frozen_parameters_unchanged=True,
        loading_rng_matches_source=True,expert_targets_used=False,checks=checks,real_states=len(x),
        maximum_latent_change=maximum_change,
        scope='Historical states with synthetic phase flags; routing validation, not closed-loop performance')
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
