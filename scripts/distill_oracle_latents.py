"""Pilot oracle latent distillation; split by environment, validate autonomously later."""
import json,os
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','3')
import torch
from fiber_soccer.lpp import Projector
root=Path(__file__).resolve().parents[1];out=root/'runs/oracle_distill_pilot'
out.mkdir(exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(810171);device='cuda:0'
rows=torch.load(root/'runs/oracle_diag_assisted16/action_diagnostics.pt',map_location='cpu',weights_only=True)
x=torch.stack([r['projector_obs'] for r in rows]);z=torch.stack([r['z'] for r in rows]);valid=torch.stack([r['active'] for r in rows])
train=valid.clone();train[:,12:]=False
val=valid.clone();val[:,:12]=False
xt=x[train].to(device);zt=z[train].to(device);xv=x[val].to(device);zv=z[val].to(device)
ck=torch.load(root/'runs/passing_50hz_long_s1/latest.pt',map_location=device,weights_only=False)
p=Projector(ck['obs_dim']).to(device);p.load_state_dict(ck['projector'])
opt=torch.optim.AdamW(p.parameters(),lr=1e-5,weight_decay=1e-4)
def validation():
 with torch.no_grad():return sum(float((p(a)-b).square().sum(-1).sum()) for a,b in zip(xv.split(64),zv.split(64)))/len(xv)
best=validation();history=[dict(update=0,validation=best)];best_state={k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
for i in range(1,201):
 ids=torch.randint(len(xt),(64,),device=device);loss=(p(xt[ids])-zt[ids]).square().sum(-1).mean()/256
 opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step()
 if i%20==0:
  score=validation();history.append(dict(update=i,loss=float(loss),validation=score));print(history[-1],flush=True)
  if score<best:best=score;best_state={k:v.detach().cpu().clone() for k,v in p.state_dict().items()}
# Optimizer intentionally omitted: this is a deployment pilot, not a resume checkpoint.
torch.save(dict(projector=best_state,obs_dim=ck['obs_dim'],latent_interval=1,latent_execution='slerp',method='oracle_latent_distillation'),out/'latest.pt')
(out/'training.json').write_text(json.dumps(dict(train_samples=len(xt),validation_samples=len(xv),history=history,selection='held-out environment latent error; not task success'),indent=2))
