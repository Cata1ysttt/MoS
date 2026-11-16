"""Measure deterministic policy changes on fixed real post-pass shooting states.

No expert actions, inverse solver, simulation or parameter updates. Shared
states make output changes comparable; they do not establish task improvement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import load_planner
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS,ROOT

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--states',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--shorter',type=Path,required=True)
p.add_argument('--longer',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
d=torch.load(a.states,map_location='cpu',weights_only=True)
cut=d['metadata']['num_envs']*3//4
eligible=(d['environment']>=cut)&(d['x'][:,940]<.5)
selected=[]
for environment in d['environment'][eligible].unique():
    ids=(eligible&(d['environment']==environment)).nonzero().flatten()
    selected.append(ids[torch.linspace(0,len(ids)-1,min(4,len(ids))).long()])
indices=torch.cat(selected);x=d['x'][indices].to(a.device)
assert (x[:,-2:].argmax(-1)==1).all()
actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
meta=json.loads((ASSETS/'actor_meta.json').read_text())
default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
z={};targets={};hashes={};base_tensors=None
with torch.no_grad():
    for label,path in [('base',a.base),('shorter',a.shorter),('longer',a.longer)]:
        header=torch.load(path,map_location='cpu',weights_only=False)
        hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
        if label=='base':base_tensors=header['planner']
        else:
            assert header['shoot_adapter_after_pass_only'] and header['adapter_skill']==1
            assert header['config']['actor_sha256']==hashlib.sha256((ROOT/'AH-BFM/AH-BFM.pt').read_bytes()).hexdigest()
            for key,value in base_tensors.items():assert torch.equal(value,header['planner'][key]),key
        model=load_planner(header,a.device).requires_grad_(False)
        z[label]=model.act(x)
        action=actor(x[:,:64],x[:,64:93],x[:,93:930],z[label])
        targets[label]=default+(6*action).clamp(-6,6)*scale
        del model,header
    comparisons={}
    for old,new in [('base','shorter'),('base','longer'),('shorter','longer')]:
        delta=z[new]-z[old];state_rms=delta.square().mean(-1).sqrt()
        cosine=torch.nn.functional.cosine_similarity(z[new],z[old],dim=-1).clamp(-1,1)
        joint_delta=targets[new]-targets[old]
        comparisons[f'{old}_to_{new}']=dict(latent_rms=float(delta.square().mean().sqrt()),
            mean_latent_angle_deg=float(torch.rad2deg(torch.acos(cosine)).mean()),
            p90_state_latent_rms=float(torch.quantile(state_rms,.9)),
            mean_raw_gaussian_kl_at_std_0_2=float(.5*(delta/.2).square().sum(-1).mean()),
            joint_target_mse=float(joint_delta.square().mean()),
            maximum_absolute_joint_target_change=float(joint_delta.abs().max()))
assert all(v.grad is None for v in actor.parameters())
result=dict(states=len(indices),environments=int(d['environment'][indices].unique().numel()),
    state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
    source_sha256=hashlib.sha256(a.states.read_bytes()).hexdigest(),checkpoint_sha256=hashes,
    comparisons=comparisons,shared_tensors_exact=True,expert_actions_used=False,
    state_rows=[dict(environment=int(d['environment'][i]),tick=int(d['tick'][i])) for i in indices],
    scope='Four temporal quantiles per available environment in the reserved last quarter of an earlier r3 continuous-state collection, before shooting contact and including later failures. Same historical states for all models; stateless planner core represents the armed post-pass context. Not current candidate rollouts, expert fitting, learning gradients, or proof of task improvement.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='state_rows'},indent=2))
