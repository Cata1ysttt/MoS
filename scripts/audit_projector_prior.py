"""Compare retained skill projectors on the same saved validation indices as planners."""
import argparse,hashlib,json,torch
from pathlib import Path
from fiber_soccer.multiskill import load_projector,condition
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.paths import ASSETS,ROOT
p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
root=ROOT/'runs';device=a.device;torch.set_num_threads(4)
protocol=json.loads(a.selection.read_text());selection=protocol['indices']
assert protocol['sampling_horizon']==5 and protocol['samples_per_task']==256
actor=NativeActor(load_native(device));scale=torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=device)*1.5;default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=device);weight=torch.ones(29,device=device);weight[:12]=2
result={};sample_states=[]
for skill,name in [('pass','passing_50hz_long_s1'),('shoot','joint_v2_r2_pass_train')]:
 d=torch.load(root/'planner_recovery_r2_semantic'/f'{skill}_trajectory.pt',weights_only=True);rows=d['rows'];cut=len(rows[0]['x'])*3//4;length=len(rows)-4;ids=torch.tensor(selection[skill])
 x,z,h,target=[torch.stack([r[key][cut:] for r in rows[:length]]).flatten(0,1)[ids].to(device) for key in ['x','label_z','expert_observation','target']]
 sample_states.append(x.cpu())
 ck=torch.load(root/name/'latest.pt',map_location='cpu',weights_only=False);p=load_projector(ck,device).eval()
 with torch.no_grad():
  prior=p(condition(h,skill) if ck.get('multiskill') else h)
  state=(x[:,:64],x[:,64:93],x[:,93:930]);teacher=actor(*state,z).clamp(-1,1);pred=actor(*state,prior).clamp(-1,1)
  error=(((pred-teacher)*scale).square()*weight).sum(-1)/weight.sum();confidence=1/(1+(default+teacher*scale-target).square().mean(-1)/.01);near=(x[:,939]>.5)&(x[:,940]<.5)
  result[skill]=dict(weighted_action_mse=float((error*confidence).sum()/confidence.sum()),launched_before_contact_mse=float((error[near]*confidence[near]).sum()/confidence[near].sum()),latent_mse=float((prior-z).square().mean()),source=name,checkpoint_sha256=hashlib.sha256((root/name/'latest.pt').read_bytes()).hexdigest())
assert hashlib.sha256(torch.cat(sample_states).numpy().tobytes()).hexdigest()==protocol['state_sha256']
a.output.write_text(json.dumps(dict(selection=str(a.selection),results=result,note='Frozen retained projectors evaluated on r2 held-out recorded states; not autonomous performance'),indent=2));print(json.dumps(result,indent=2))
