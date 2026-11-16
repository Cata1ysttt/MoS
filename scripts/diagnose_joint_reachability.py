"""Action inversion on training replay only; never an autonomous success metric."""
import argparse,json
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.paths import ASSETS
from fiber_soccer.multiskill import load_projector
from fiber_soccer.latent_oracle import fit_latent
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(811340)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=load_projector(ck,'cpu').eval().requires_grad_(False)
data=torch.load(a.replay,map_location='cpu',weights_only=True)
ids=torch.randperm(len(data['data'][0]))[:128]
x,s,last,h,target=[v[ids] for v in data['data']]
actor=convert(str(ASSETS/'actor.onnx')).eval().requires_grad_(False)
meta=json.loads((ASSETS/'actor_meta.json').read_text());scene=json.loads((ASSETS/'pickup_scene.json').read_text())
default=torch.tensor(scene['default_q']);scale=torch.tensor(meta['action_rescale'])*.25
def targets(action):return default+(action*6).clamp(-6,6)*scale
with torch.no_grad():z=model(x)
_,audit=fit_latent(actor,(s,last,h),target,z,targets,iterations=128)
audit.update(skill=data['skill'],samples=len(ids),checkpoint=str(a.checkpoint),replay=str(a.replay),expert_assisted=True,interpretation='Per-state action reachability on training replay, not an executable policy or a closed-loop success rate.')
with torch.no_grad():
    bounded=torch.maximum(torch.minimum(target,default+6*scale),default-6*scale)
    excess=(target-bounded).square()
    per_joint=excess.mean(0)
    audit['box_constraint_mse_lower_bound']=float(excess.mean())
    audit['target_joint_fraction_outside_actor_box']=float((excess>1e-10).float().mean())
    audit['box_error_top_joints']=[dict(joint=meta['joint_names'][i],mse=float(per_joint[i])) for i in per_joint.argsort(descending=True)[:5].tolist()]
a.output.write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
