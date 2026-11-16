"""Balanced supervised latent sequence planner; held out complete lanes."""
import argparse,json,time
from pathlib import Path
import torch
from fiber_soccer.planner import LatentPlanner,SCHEMA,trajectory_windows,load_planner
from fiber_soccer.native_bfm import load_native,NativeActor
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--updates',type=int,default=3000);p.add_argument('--seed',type=int,default=811410);p.add_argument('--device',default='cuda:0');p.add_argument('--horizon',type=int,default=5);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(a.seed)
train={};valid={};provenance=[]
for path in a.data:
 d=torch.load(path,map_location='cpu',weights_only=True);m=d['metadata'];assert m['schema']==SCHEMA
 assert m['seed']!=811314,'Development evaluation must not enter training'
 provenance.append(m);rows=d['rows'];x=torch.stack([r['x'] for r in rows]);z=torch.stack([r.get('label_z',r['z']) for r in rows]);episode=torch.stack([r['episode'] for r in rows]);target=torch.stack([r['target'] for r in rows]);T,N=x.shape[:2]
 parts=trajectory_windows(x,z,episode,target,a.horizon)
 split=max(1,N*3//4)
 for pool,section in [(train,slice(None,split)),(valid,slice(split,None))]:
  values=tuple(v[:,section].flatten(0,1) for v in parts)
  if m['skill'] in pool:values=tuple(torch.cat((old,new)) for old,new in zip(pool[m['skill']],values))
  pool[m['skill']]=values
assert set(train)=={'pass','shoot'}
if a.resume:
 model=load_planner(torch.load(a.resume,map_location='cpu',weights_only=False),a.device)
 assert model.horizon==a.horizon
else:
 model=LatentPlanner(a.horizon).to(a.device)
 all_x=torch.cat([v[0] for v in train.values()]);model.obs_mean.copy_(all_x.mean(0));model.obs_std.copy_(all_x.std(0).clamp_min(.05))
actor=NativeActor(load_native(a.device))
meta=json.loads((ASSETS/'actor_meta.json').read_text());scene=json.loads((ASSETS/'pickup_scene.json').read_text());default=torch.tensor(scene['default_q'],device=a.device);scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
onset={id(pool):{skill:torch.where(values[0][:,93:930].abs().sum(-1)<1e-8)[0] for skill,values in pool.items()} for pool in (train,valid)}
config=dict(reset_sample_fraction=.25,method='unified_latent_sequence_regression',seed=a.seed,resume=str(a.resume),data=provenance,horizon=a.horizon,width=model.width,execution='replan_every_20ms_first_latent',assisted_training_labels=True,validation='held-out environment lanes, not task success')
(a.output/'config.json').write_text(json.dumps(config,indent=2));start=time.monotonic();best=float('inf')
def batch(pool,size):
 out=[]
 for skill in ['pass','shoot']:
  values=pool[skill];start=onset[id(pool)][skill]
  n_start=size//4 if len(start) else 0
  ids=torch.randint(len(values[0]),(size-n_start,))
  if n_start:ids=torch.cat((ids,start[torch.randint(len(start),(n_start,))]))
  out.append(tuple(v[ids].to(a.device) for v in values))
 return tuple(torch.cat([v[k] for v in out]) for k in range(4))
def loss(batch):
 x,z,mask,target=batch;pred=model(x)
 latent=((pred-z).square().mean(-1)*mask).sum()/mask.sum()
 action=actor(x[:,:64],x[:,64:93],x[:,93:930],pred[:,0]);physical=default+(action*6).clamp(-6,6)*scale
 action_loss=(physical-target).square().mean()
 return latent+10*action_loss,latent,action_loss
for update in range(1,a.updates+1):
 model.train();total,latent,action=loss(batch(train,64));opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 if update%100==0 or update==1:
  model.eval()
  # Fixed sample indices at every validation checkpoint for comparable selection.
  state=torch.random.get_rng_state();torch.manual_seed(811411)
  with torch.no_grad():v,vl,va=loss(batch(valid,256))
  torch.random.set_rng_state(state)
  row=dict(update=update,loss=float(total.detach()),latent=float(latent.detach()),action=float(action.detach()),validation=float(v),validation_latent=float(vl),validation_action=float(va),elapsed=time.monotonic()-start)
  print(row,flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  payload=dict(planner=model.state_dict(),observation_schema=SCHEMA,horizon=a.horizon,width=model.width,latent_interval=1,latent_execution='hold',updates=update,config=config)
  torch.save(payload,a.output/'latest.pt')
  if float(v)<best:best=float(v);torch.save(payload,a.output/'best.pt')
assert all(p.grad is None for p in actor.parameters())
(a.output/'complete.json').write_text(json.dumps(dict(updates=a.updates,best_validation=best)))
