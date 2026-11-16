"""Balanced offline action-aligned two-expert pilot; no full-FIBER claim."""
import argparse,json,time
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.paths import ASSETS
from fiber_soccer.lpp import Projector
from fiber_soccer.multiskill import condition
p=argparse.ArgumentParser();p.add_argument('--pass-data',type=Path,required=True);p.add_argument('--shoot-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=1000);p.add_argument('--device',default='cuda:0');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(811313);a.output.mkdir(parents=True,exist_ok=False)
data={}
for skill,path in [('pass',a.pass_data),('shoot',a.shoot_data)]:
 rows=torch.load(path,weights_only=True)
 xs=[];ys=[];ss=[[],[],[]];lanes=[]
 for row in rows:
  x=row['projector_obs'] if skill=='pass' else torch.cat((row['policy_obs'],row['rma_obs']),-1)
  states=row['bfm_inputs'] if skill=='pass' else row['inputs']
  y=row['targets'] if skill=='pass' else row['target']
  xs.append(condition(x,skill));ys.append(y);lanes.append(torch.arange(len(x)))
  for k in range(3):ss[k].append(states[k])
 lane=torch.cat(lanes);data[skill]=dict(x=torch.cat(xs),y=torch.cat(ys),s=tuple(torch.cat(s) for s in ss),train=torch.where(lane<12)[0],validation=torch.where(lane>=12)[0])
actor=convert(str(ASSETS/'actor.onnx')).eval().to(a.device).requires_grad_(False)
meta=json.loads((ASSETS/'actor_meta.json').read_text());scene=json.loads((ASSETS/'pickup_scene.json').read_text())
default=torch.tensor(scene['default_q'],device=a.device);scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
model=Projector(1202).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
config=dict(method='joint_action_only_projector_offline_pilot',skills=['pass','shoot'],backbone='ah_bfm',seed=811313,updates=a.updates,batch_per_skill=64,holdout='lanes12:16; no heldout lanes used for gradients',source={s:str(p) for s,p in [('pass',a.pass_data),('shoot',a.shoot_data)]},limitations=['not full LPP; no paired backward encoder','not FIBER flow planner','offline teacher states only; autonomous evaluation required'])
(a.output/'config.json').write_text(json.dumps(config,indent=2))
import swanlab
swanlab.init(project='MoS',name=a.output.name,config=config,mode='online')
def loss(skill,split,count):
 d=data[skill];pool=d[split];ids=pool[torch.randint(len(pool),(count,))]
 x=d['x'][ids].to(a.device);states=tuple(v[ids].to(a.device) for v in d['s']);y=d['y'][ids].to(a.device)
 action=actor(*states,model(x));pred=default+(action*6).clamp(-6,6)*scale
 return (pred-y).square().mean()
start=time.monotonic()
for update in range(a.updates):
 model.train();opt.zero_grad();row={}
 for skill in data:
  l=loss(skill,'train',64);(l*.5).backward();row[skill+'_train_mse']=float(l.detach())
 torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 if update%50==0 or update+1==a.updates:
  model.eval()
  with torch.no_grad():
   for skill in data:row[skill+'_validation_mse']=float(loss(skill,'validation',256))
  row.update(update=update+1,elapsed=time.monotonic()-start);print(row,flush=True);swanlab.log(row,step=update+1)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  torch.save(dict(projector=model.state_dict(),obs_dim=1202,multiskill=True,skills=['pass','shoot'],latent_interval=1,latent_execution='hold',updates=update+1),a.output/'latest.tmp');(a.output/'latest.tmp').replace(a.output/'latest.pt')
assert all(v.grad is None for v in actor.parameters())
swanlab.finish()
