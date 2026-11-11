"""Frozen-actor action distillation on oracle and student states, environment split."""
import os,json
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
import torch
from onnx2torch import convert
from fiber_soccer.lpp import Projector
r=Path(__file__).resolve().parents[1];out=r/'runs/oracle_action_distill_pilot';out.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(810172);device='cuda:0'
actor=convert(str(r/'assets/actor.onnx')).to(device).eval().requires_grad_(False)
meta=json.loads((r/'assets/actor_meta.json').read_text());scene=json.loads((r/'assets/pickup_scene.json').read_text())
scale=torch.tensor(meta['action_rescale'],device=device)*.25;default=torch.tensor(scene['default_q'],device=device)
def targets(a):return default+(a*6).clamp(-6,6)*scale
sets=[]
for name in ['oracle_diag_assisted16','oracle_diag_student16']:
 rows=torch.load(r/'runs'/name/'action_diagnostics.pt',map_location='cpu',weights_only=True)
 x=torch.stack([a['projector_obs'] for a in rows]);y=torch.stack([a['expert_targets'] for a in rows]);s=[torch.stack([a['bfm_inputs'][k] for a in rows]) for k in range(3)];m=torch.stack([a['active'] for a in rows])
 parts=[]
 for training in [True,False]:
  mask=m.clone()
  if training:mask[:,12:]=False
  else:mask[:,:12]=False
  parts.append(tuple(v[mask].to(device) for v in [x,y,*s]))
 sets.append(parts)
ck=torch.load(r/'runs/passing_50hz_long_s1/latest.pt',map_location=device,weights_only=False)
p=Projector(ck['obs_dim']).to(device);p.load_state_dict(ck['projector']);opt=torch.optim.AdamW(p.parameters(),lr=1e-5,weight_decay=1e-4)
def loss(data,idx):
 x,y,*s=data
 return (targets(actor(*(v[idx] for v in s),p(x[idx])))-y[idx]).square().mean()
def validate():
 with torch.no_grad():
  return [sum(float(loss(pair[1],ids))*len(ids) for ids in torch.arange(len(pair[1][0]),device=device).split(64))/len(pair[1][0]) for pair in sets]
initial=validate();best=sum(initial);beststate={k:v.cpu().clone() for k,v in p.state_dict().items()};history=[dict(update=0,validation=initial)]
for i in range(1,301):
 opt.zero_grad();total=0
 for pair in sets:
  data=pair[0];ids=torch.randint(len(data[0]),(32,),device=device);v=loss(data,ids)/2;v.backward();total+=float(v.detach())
 torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step()
 if i%25==0:
  scores=validate();row=dict(update=i,loss=total,validation=scores);history.append(row);print(row,flush=True)
  if sum(scores)<best:best=sum(scores);beststate={k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
assert all(v.grad is None for v in actor.parameters())
torch.save(dict(projector=beststate,obs_dim=ck['obs_dim'],latent_interval=1,latent_execution='slerp',method='mixed_oracle_student_action_distillation'),out/'latest.pt')
(out/'training.json').write_text(json.dumps(dict(history=history,frozen_actor=True,train_envs=list(range(12)),validation_envs=list(range(12,16)),selection='Mean action validation MSE; task performance unverified'),indent=2))
