"""Prepare a zero-initialized trainable correction without changing baseline behavior."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.guidance_data import load_guidance_data
from fiber_soccer.post_pass_latent_adapter import LatentResidual
from fiber_soccer.new_ball_task_history import NewBallTaskHistory

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    digest=sha(a.source);assert digest=='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'
    header=torch.load(a.source,map_location='cpu',weights_only=False)
    source=load_soccer_policy(a.source,'cuda:0')
    layer=LatentResidual(source.planner.obs_mean.cpu(),source.planner.obs_std.cpu())
    header['post_pass_latent_adapter']=dict(version=1,width=128,model={k:v.detach().clone() for k,v in layer.state_dict().items()})
    rng=torch.random.get_rng_state();gpu=torch.cuda.get_rng_state()
    source=load_soccer_policy(a.source,'cuda:0');after=torch.random.get_rng_state();gpu_after=torch.cuda.get_rng_state()
    torch.random.set_rng_state(rng);torch.cuda.set_rng_state(gpu)
    policy=load_soccer_policy(header,'cuda:0')
    assert torch.equal(after,torch.random.get_rng_state()) and torch.equal(gpu_after,torch.cuda.get_rng_state())
    _,validation,_=load_guidance_data([Path(d['path']) for d in header['config']['data']])
    x=torch.cat([v[0][:8] for v in validation.values()]).cuda();checks={};manual=NewBallTaskHistory()
    def compare(observed):
        actual=policy(observed);expected=source(observed)
        assert all(torch.equal(actual[k],expected[k]) for k in expected)
        return actual
    with torch.inference_mode():
        for skill in ['pass','shoot']:
            source.reset();policy.reset()
            for kicked in [False,True]:
                x=source.condition_observation(x,skill=skill);x[:,940]=float(kicked)
                compare(x);assert not policy.last_adapter_active.any()
            checks['native_'+skill+'_outputs_exact']=True
        source.reset();policy.reset();manual.reset()
        for kicked in [False,True]:
            x=source.condition_observation(x,skill='pass');x[:,940]=float(kicked)
            compare(x);manual(x);assert not policy.last_adapter_active.any()
        for age in range(51):
            x=source.condition_observation(x,skill='shoot');x[:,940]=0
            compare(x);transformed=manual(x)
            assert torch.equal(policy.last_adapter_input,transformed),age
            assert policy.last_adapter_active.all()
        checks['zero_adapter_continuous_outputs_exact']=True
        checks['corrected_history_reconstructed_without_double_advance']=True
        x[:,940]=1;compare(x);assert not policy.last_adapter_active.any()
        x=source.condition_observation(x,skill='pass');compare(x);assert not policy.last_adapter_active.any()
        checks['shoot_contact_and_recovery_preserved']=True
        # Deliberately nonzero weights exercise the actual gating, not just a zero network.
        policy.post_pass_adapter.net[-1].bias.fill_(.05)
        source.reset();policy.reset();x[:,940]=1
        compare(x)  # Passing contact arms only the post-pass context.
        x=source.condition_observation(x,skill='shoot');x[:,940]=0
        mask=torch.arange(len(x),device=x.device)%2==0
        source.reset(mask);policy.reset(mask)
        actual=policy(x);baseline=source(x)
        assert policy.last_adapter_active.tolist()==(~mask).tolist()
        for key in actual:assert torch.equal(actual[key][mask],baseline[key][mask])
        assert not torch.equal(actual['latent'][~mask],baseline['latent'][~mask])
        manual_z=policy.post_pass_adapter(policy.last_adapter_input,policy.last_base_latent)
        assert torch.equal(actual['latent'][~mask],manual_z[~mask])
        checks['nonzero_adapter_mixed_gate_and_partial_reset_correct']=True
        policy.post_pass_adapter.load_state_dict(header['post_pass_latent_adapter']['model'])
    original=source.state_dict();current=policy.state_dict()
    for key,value in original.items():assert torch.equal(value,current[key]),key
    assert all(not v.requires_grad and v.grad is None for v in policy.parameters())
    assert sha(a.source)==digest
    a.output.mkdir(parents=True);torch.save(header,a.output/'diagnostic.pt')
    restored=load_soccer_policy(a.output/'diagnostic.pt','cuda:0');policy.reset();source.reset()
    with torch.inference_mode():
        actual=policy(x);expected=restored(x)
        assert all(torch.equal(value,expected[key]) for key,value in actual.items())
    checks['checkpoint_roundtrip_exact']=True
    result=dict(source_sha256=digest,checkpoint_sha256=sha(a.output/'diagnostic.pt'),checks=checks,
        steps=header['predicted_target_solver_steps'],backend=header['predicted_target_solver_backend'],
        refinement_scope='initial_pass',standing_recovery_mode=header['standing_recovery_mode'],
        new_ball_history_mode=header['new_ball_history_mode'],post_pass_latent_adapter_version=1,
        real_states=len(x),frozen_parameters_unchanged=True,loading_rng_matches_source=True,expert_targets_used=False,
        adapter_parameters=sum(v.numel() for v in layer.parameters()),
        scope='Zero initialization and nonzero gating checks on historical states with synthetic phases; no trained performance claim')
    (a.output/'complete.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
