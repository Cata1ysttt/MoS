"""Measure speed-conditioning sensitivity on fixed held-out post-pass states.

This is an offline input intervention, not a rollout or a recovery controller.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS
from fiber_soccer.planner import load_planner,with_task_history

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cpu')
a=p.parse_args()
assert not a.output.exists()
torch.set_num_threads(4)
selected=[]
for path in a.data:
    assert (path.parent/'complete.json').exists()
    data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['metadata']['skill']=='pass'
    rows=data['rows'];cut=len(rows[0]['x'])*3//4
    x=torch.stack([row['x'][cut:] for row in rows])
    history=torch.stack([row['expert_observation'][cut:] for row in rows])
    x=with_task_history(x,history)
    mask=x[:,:,940]>.5
    values=x[mask]
    assert len(values)>0
    indices=torch.linspace(0,len(values)-1,min(512,len(values))).long()
    selected.append(values[indices])
    del data,rows,x,history
x=torch.cat(selected)
assert torch.all(x[:,-2]==1) and torch.all(x[:,-1]==0)
actor=NativeActor(load_native(a.device)).eval().requires_grad_(False)
meta=json.loads((ASSETS/'actor_meta.json').read_text())
default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
scale=torch.tensor(meta['action_rescale'],device=a.device)*1.5
results={}
with torch.inference_mode():
    for path in a.checkpoints:
        assert (path.parent/'complete.json').exists()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        model=load_planner(torch.load(path,map_location='cpu',weights_only=False),a.device).eval()
        outputs={}
        for speed in ['recorded',2.5,6.0]:
            targets=[];latents=[]
            for start in range(0,len(x),128):
                inputs=x[start:start+128].to(a.device).clone()
                if speed!='recorded':inputs[:,-3]=speed
                z=model.act(inputs)
                action=actor(inputs[:,:64],inputs[:,64:93],inputs[:,93:930],z)
                targets.append((default+action.clamp(-1,1)*scale).cpu());latents.append(z.cpu())
            outputs[str(speed)]=(torch.cat(targets),torch.cat(latents))
        changes={}
        for speed in ['2.5','6.0']:
            delta=outputs[speed][0]-outputs['recorded'][0]
            per_state=delta.square().mean(-1).sqrt()
            changes[speed]=dict(joint_target_rms_rad=float(delta.square().mean().sqrt()),
                leg_target_rms_rad=float(delta[:,:12].square().mean().sqrt()),
                state_target_rms_quantiles_rad=torch.quantile(per_state,torch.tensor([.5,.9,.99])).tolist(),
                latent_rms=float((outputs[speed][1]-outputs['recorded'][1]).square().mean().sqrt()))
        results[str(path)]=dict(checkpoint_sha256=digest,changes_from_recorded_speed=changes)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
report=dict(states=len(x),state_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    recorded_speed_range=[float(x[:,-3].min()),float(x[:,-3].max())],data=list(map(str,a.data)),results=results,
    scope='Held-out environment quarter; up to 512 evenly spaced kicked-state rows per dataset. Only the final planner speed command changes. Historical observations remain recorded, so this is an immediate input sensitivity diagnostic, not a counterfactual rollout or proof of post-shoot recovery behavior. No parameter updates.')
with a.output.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
