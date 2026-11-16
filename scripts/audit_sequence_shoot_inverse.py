"""Diagnose latent fitting on held-out post-pass shooting states; no rollout or training labels."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import load_planner
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.latent_oracle import fit_latent,SOLVER_VERSION

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--labels',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--iterations',type=int,default=64)
a=p.parse_args()
assert a.iterations>0 and not a.output.exists()
torch.set_num_threads(4)
data=torch.load(a.labels,map_location='cpu',weights_only=True)
assert data['metadata']['physical_target_mapping']=='raw_action_clamp_10_then_affine_no_physical_clip'
assert data['metadata']['purpose']=='training'
cut=data['metadata']['num_envs']*3//4
eligible=(data['environment']>=cut)&(data['x'][:,940]<.5)
selected=[]
for environment in data['environment'][eligible].unique():
    ids=(eligible&(data['environment']==environment)).nonzero().flatten()
    selected.append(ids[torch.linspace(0,len(ids)-1,min(4,len(ids))).long()])
ids=torch.cat(selected);x=data['x'][ids].to(a.device);target=data['target'][ids].to(a.device)
actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
meta=json.loads((ASSETS/'actor_meta.json').read_text())
default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
weights=torch.ones(29,device=a.device);weights[:12]=2
state=(x[:,:64],x[:,64:93],x[:,93:930])
def physical(action):return default+(action*6).clamp(-6,6)*scale
errors={};z=None
with torch.no_grad():
    for name,path in [('base',a.base),('candidate',a.candidate)]:
        model=load_planner(torch.load(path,map_location='cpu',weights_only=False),a.device).requires_grad_(False)
        z=model.act(x)
        errors[name]=(physical(actor(*state,z))-target).square()
refined,solver=fit_latent(actor,state,target,z,physical,a.iterations)
with torch.no_grad():errors['offline_inverse']=(physical(actor(*state,refined))-target).square()
assert all(v.grad is None for v in actor.parameters())
errors['joint_target_box_lower_bound']=(target-target.clamp(default-6*scale,default+6*scale)).square()
results={}
for name,error in errors.items():
    results[name]=dict(weighted_action_mse=float((error*weights).sum(-1).div(weights.sum()).mean()),
        action_mse=float(error.mean()),leg_mse=float(error[:,:12].mean()),
        maximum_state_action_mse=float(error.mean(-1).max()))
report=dict(states=len(ids),environments=int(data['environment'][ids].unique().numel()),
    state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),iterations=a.iterations,
    solver_version=SOLVER_VERSION,solver=solver,results=results,
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('labels',a.labels),('base',a.base),('candidate',a.candidate)]},
    state_rows=[dict(environment=int(data['environment'][i]),tick=int(data['tick'][i])) for i in ids],
    scope='Up to four temporal quantiles per held-out environment before shooting contact, including later-failed trajectories. Independent expert-assisted latent fitting on fixed states, not autonomous execution or proof of continuous control. No inverse labels or optimized latents are saved for training.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['state_rows']},indent=2))
