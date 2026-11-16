"""Offline expert-assisted inversion on recorded autonomous first-impact states."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.latent_oracle import fit_latent,SOLVER_VERSION

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--iterations',type=int,default=64)
p.add_argument('--frames-before',type=int,default=20)
a=p.parse_args()
assert a.iterations>0 and a.frames_before>0 and not a.output.exists()
torch.set_num_threads(4)
summary=json.loads((a.evaluation/'summary.json').read_text())
assert summary['completed']==summary['requested'] and not summary['expert_assisted_oracle']
assert summary['controller']=='unified_latent_planner'
impacts=json.loads((a.evaluation/'first_kick_velocity.json').read_text())
episodes={row['environment']:row for row in json.loads((a.evaluation/'episodes.json').read_text())}
rows=torch.load(a.evaluation/'action_diagnostics.pt',map_location='cpu',weights_only=True)
selected=[]
for row in rows:
    for key,impact in impacts['samples'].items():
        i=int(key)
        if row['active'][i] and impact['tick']-a.frames_before<=row['tick']<impact['tick']:
            selected.append((row,i,impact))
assert selected,'No recorded states before first impacts'
state=tuple(torch.stack([row['bfm_inputs'][k][i] for row,i,_ in selected]).to(a.device) for k in range(3))
z=torch.stack([row['z'][i] for row,i,_ in selected]).to(a.device)
target=torch.stack([row['expert_targets'][i] for row,i,_ in selected]).to(a.device)
recorded=torch.stack([row['bfm_targets'][i] for row,i,_ in selected]).to(a.device)
actor=NativeActor(load_native(a.device))
meta=json.loads((ASSETS/'actor_meta.json').read_text())
default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale=torch.tensor(meta['action_rescale'],device=a.device)*.25

def to_targets(action):return default+(action*6).clamp(-6,6)*scale

with torch.no_grad():
    replay=to_targets(actor(*state,z))
    torch.testing.assert_close(replay,recorded,atol=2e-6,rtol=1e-5)
    before=(replay-target).square()
refined,solver=fit_latent(actor,state,target,z,to_targets,a.iterations)
with torch.no_grad():after=(to_targets(actor(*state,refined))-target).square()
assert all(parameter.grad is None for parameter in actor.parameters())
weights=torch.ones(29,device=a.device);weights[:12]=2
successful=torch.tensor([episodes[i]['metrics']['quality_ball_direction_success_10deg']>.5 for _,i,_ in selected],device=a.device)
results={}
for name,mask in [('all',torch.ones(len(selected),dtype=torch.bool,device=a.device)),('native_direction_success',successful),('native_direction_failure',~successful)]:
    if not mask.any():
        results[name]=dict(states=0)
        continue
    errors={}
    for label,value in [('planner',before),('offline_inverse',after)]:
        sub=value[mask]
        errors[label]=dict(action_mse=float(sub.mean()),leg_mse=float(sub[:,:12].mean()),
            weighted_action_mse=float((sub*weights).sum(-1).div(weights.sum()).mean()),
            per_joint_mse=dict(zip(meta['joint_names'],sub.mean(0).cpu().tolist())))
    results[name]=dict(states=int(mask.sum()),errors=errors)
report=dict(evaluation=str(a.evaluation),seed=summary['seed'],episodes=summary['requested'],
    sampled_environments=sorted({i for _,i,_ in selected}),missing_first_impact_environments=impacts['missing_environment_ids'],
    frames_before=a.frames_before,diagnostic_stride=5,iterations=a.iterations,solver_version=SOLVER_VERSION,solver=solver,
    executed_target_replay_max_abs_error=float((replay-recorded).abs().max()),results=results,
    source_hashes={name:hashlib.sha256((a.evaluation/name).read_bytes()).hexdigest() for name in ['summary.json','episodes.json','first_kick_velocity.json','action_diagnostics.pt']},
    state_rows=[dict(tick=row['tick'],environment=i,impact_tick=impact['tick']) for row,i,impact in selected],
    scope='Uses expert action targets on recorded evaluation states only. Independent per-state inversion, not an autonomous rollout or a guarantee of continuous control. No validation states or inverse labels are added to training.')
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps({key:{k:v for k,v in value.items() if k!='errors'}|{'errors':{role:{k:v for k,v in error.items() if k!='per_joint_mse'} for role,error in value.get('errors',{}).items()}} for key,value in results.items()},indent=2))
