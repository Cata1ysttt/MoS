"""Conditional endpoint flow with optional decoded-action supervision."""
import argparse,json,time,math
from pathlib import Path
import torch
from fiber_soccer.planner import trajectory_windows,load_planner,balanced_skill_mean
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--label-error-scale',type=float,default=0.,help='Weight labels by 1/(1+physical expert-action MSE/scale); zero disables');p.add_argument('--action-weight',type=float,default=0.);p.add_argument('--leg-weight',type=float,default=1.);p.add_argument('--no-command-conditioning',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--updates',type=int,default=20000);p.add_argument('--seed',type=int,default=812150);p.add_argument('--device',default='cuda:0');p.add_argument('--horizon',type=int,default=5);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(a.seed)
if not math.isfinite(a.label_error_scale) or a.label_error_scale<0:raise ValueError('Label error scale must be finite and nonnegative')
actor=None
if a.action_weight>0 or a.label_error_scale>0:
 from fiber_soccer.native_bfm import NativeActor,load_native
 actor=NativeActor(load_native(a.device))
 action_scale=torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=a.device)*1.5
 default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
 joint_weight=torch.ones(29,device=a.device);joint_weight[:12]=a.leg_weight
train={};valid={};provenance=[];schema=None;label_audits=[]
for path in a.data:
 d=torch.load(path,map_location='cpu',weights_only=True);m=d['metadata']
 if schema is None:schema=m['schema']
 assert m['schema']==schema,'Mixed observation schemas require fresh aligned collection'
 assert m['seed']!=811314,'Development evaluation must not enter training'
 provenance.append(m);rows=d['rows'];x=torch.stack([r['x'] for r in rows]);z=torch.stack([r.get('label_z',r['z']) for r in rows]);episode=torch.stack([r['episode'] for r in rows]);target=torch.stack([r['target'] for r in rows]);T,N=x.shape[:2]
 confidence=None
 if a.label_error_scale>0:
  flat=x.flatten(0,1);labels=z.flatten(0,1);targets=target.flatten(0,1);errors=[]
  with torch.no_grad():
   for begin in range(0,len(flat),1024):
    xb=flat[begin:begin+1024].to(a.device);zb=labels[begin:begin+1024].to(a.device)
    predicted=default+actor(xb[:,:64],xb[:,64:93],xb[:,93:930],zb).clamp(-1,1)*action_scale
    errors.append((predicted-targets[begin:begin+1024].to(a.device)).square().mean(-1).cpu())
  error=torch.cat(errors).reshape(T,N);confidence=1/(1+error/a.label_error_scale)
  label_audits.append(dict(path=str(path),mean_action_mse=float(error.mean()),mean_confidence=float(confidence.mean()),fraction_below_half=float((confidence<.5).float().mean())))
 parts=trajectory_windows(x,z,episode,target,a.horizon,confidence)
 split=max(1,N*3//4)
 for pool,section in [(train,slice(None,split)),(valid,slice(split,None))]:
  values=tuple(v[:,section].flatten(0,1) for v in parts)
  if m['skill'] in pool:values=tuple(torch.cat((old,new)) for old,new in zip(pool[m['skill']],values))
  pool[m['skill']]=values
assert set(train)=={'pass','shoot'}
if a.resume:
 assert torch.load(a.resume,map_location='cpu',weights_only=False).get('planner_arch')=='flow_endpoint_v1'
 model=load_planner(torch.load(a.resume,map_location='cpu',weights_only=False),a.device)
 assert model.horizon==a.horizon and model.schema==schema
else:
 model=FlowLatentPlanner(a.horizon,schema=schema,command_conditioning=not a.no_command_conditioning).to(a.device)
 all_x=torch.cat([v[0] for v in train.values()]);model.obs_mean.copy_(all_x.mean(0));model.obs_std.copy_(all_x.std(0).clamp_min(.05))
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
onset={id(pool):{skill:torch.where(values[0][:,93:930].abs().sum(-1)<1e-8)[0] for skill,values in pool.items()} for pool in (train,valid)}
config=dict(confidence_normalization='within_each_skill' if a.label_error_scale>0 else 'disabled',label_error_scale=a.label_error_scale,label_audits=label_audits,method='conditional_endpoint_flow_mlp',seed=a.seed,data=provenance,horizon=a.horizon,width=model.width,
    action_auxiliary=a.action_weight>0,action_weight=a.action_weight,leg_weight=a.leg_weight,auxiliary_path='full_8_step_deployed_sampler',command_conditioning=model.command_conditioning,reset_sample_fraction=.25,flow_steps=model.flow_steps,noise_seed=model.noise_seed,
    execution='fixed_noise_replan_every_20ms_first_latent',validation='held-out lanes, not autonomous success')
(a.output/'config.json').write_text(json.dumps(config,indent=2));start=time.monotonic();best=float('inf')
def batch(pool,size):
 out=[]
 for skill in ['pass','shoot']:
  values=pool[skill];starts=onset[id(pool)][skill];count=size//4 if len(starts) else 0
  ids=torch.randint(len(values[0]),(size-count,))
  if count:ids=torch.cat((ids,starts[torch.randint(len(starts),(count,))]))
  out.append(tuple(v[ids].to(a.device) for v in values))
 return tuple(torch.cat([v[k] for v in out]) for k in range(4))
def loss(values):
 x,target,mask,_=values
 noise=torch.randn_like(target);time=torch.rand(len(x),device=a.device)
 mixed=(1-time[:,None,None])*noise+time[:,None,None]*target
 prediction=model.endpoint(mixed,time,x)
 squared=(prediction-target).square().mean(-1)
 flow_loss=balanced_skill_mean(squared,mask) if a.label_error_scale>0 else (squared*mask).sum()/mask.sum()
 action_loss=flow_loss.new_zeros(())
 if a.action_weight>0:
  sampled=model(x)
  predicted_action=actor(x[:,:64],x[:,64:93],x[:,93:930],sampled[:,0])
  with torch.no_grad():teacher_action=actor(x[:,:64],x[:,64:93],x[:,93:930],target[:,0])
  error=((predicted_action.clamp(-1,1)-teacher_action.clamp(-1,1))*action_scale).square()
  per_sample=(error*joint_weight).sum(-1).div(joint_weight.sum())
  action_loss=balanced_skill_mean(per_sample,mask[:,0]) if a.label_error_scale>0 else per_sample.mean()
 return flow_loss+a.action_weight*action_loss,flow_loss,action_loss
for update in range(1,a.updates+1):
 model.train();value,flow_loss,action_loss=loss(batch(train,64));opt.zero_grad();value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
 if update%200==0 or update==1:
  model.eval()
  with torch.random.fork_rng(devices=[torch.device(a.device).index or 0] if a.device.startswith('cuda') else []):
   torch.manual_seed(812151)
   with torch.no_grad():
    values=batch(valid,256);score,validation_flow,validation_action=loss(values)
    prediction=model(values[0]);sample_error=((prediction-values[1]).square().mean(-1)*values[2]).sum()/values[2].sum()
  row=dict(update=update,loss=float(value.detach()),flow_loss=float(flow_loss.detach()),action_loss=float(action_loss.detach()),validation_objective=float(score),validation_endpoint=float(validation_flow),validation_action=float(validation_action),validation_sample=float(sample_error),elapsed=time.monotonic()-start)
  print(row,flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  payload=dict(planner=model.state_dict(),planner_arch='flow_endpoint_v1',observation_schema=schema,
      horizon=a.horizon,width=model.width,command_conditioning=model.command_conditioning,flow_steps=model.flow_steps,noise_seed=model.noise_seed,
      latent_interval=1,latent_execution='hold',updates=update,config=config)
  torch.save(payload,a.output/'latest.pt')
  if float(score)<best:best=float(score);torch.save(payload,a.output/'best.pt')
if actor is not None:assert all(p.grad is None for p in actor.parameters())
(a.output/'complete.json').write_text(json.dumps(dict(updates=a.updates,best_validation_objective=best)))
