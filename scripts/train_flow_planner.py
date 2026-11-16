"""Conditional endpoint flow with optional decoded-action supervision."""
import argparse,json,time,math
from pathlib import Path
import torch
from fiber_soccer.planner import trajectory_windows,load_planner,balanced_skill_mean,observed_contact_window,SCHEMA_V3,SCHEMA_V4,with_task_history,DirectLatentPlanner
from fiber_soccer.flow_planner import FlowLatentPlanner,expand_task_history
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--action-target',choices=['decoded_label','expert'],default='decoded_label',help='Physical target for action alignment; expert uses the recorded original expert joint targets');p.add_argument('--pass-approach-sample-fraction',type=float,default=0.,help='Training-only pass samples within 0.6m, launched and not kicked, outside the contact window');p.add_argument('--architecture',choices=['flow','direct'],default='flow');p.add_argument('--task-history',action='store_true',help='Add saved observation-manager histories to the shared latent planner');p.add_argument('--data',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--contact-sample-fraction',type=float,default=0.,help='Training-only contact-window sampling; validation remains unchanged');p.add_argument('--label-error-scale',type=float,default=0.,help='Weight labels by 1/(1+physical expert-action MSE/scale); zero disables');p.add_argument('--flow-weight',type=float,default=1.,help='Zero runs decoded-action-only alignment from a warm planner');p.add_argument('--action-weight',type=float,default=0.);p.add_argument('--leg-weight',type=float,default=1.);p.add_argument('--no-command-conditioning',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--updates',type=int,default=20000);p.add_argument('--seed',type=int,default=812150);p.add_argument('--device',default='cuda:0');p.add_argument('--horizon',type=int,default=5);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(a.seed)
if not math.isfinite(a.label_error_scale) or a.label_error_scale<0:raise ValueError('Label error scale must be finite and nonnegative')
if not math.isfinite(a.pass_approach_sample_fraction) or not 0 <= a.pass_approach_sample_fraction <= .75-a.contact_sample_fraction:raise ValueError('Approach and contact fractions must be nonnegative and total at most .75')
if not math.isfinite(a.contact_sample_fraction) or not 0<=a.contact_sample_fraction<=.75:raise ValueError('Contact sample fraction must be in [0, .75]')
if not math.isfinite(a.flow_weight) or a.flow_weight<0:raise ValueError('Flow weight must be finite and nonnegative')
if a.flow_weight==0 and a.action_weight<=0:raise ValueError('At least one training objective must be active')
actor=None
if a.action_weight>0 or a.label_error_scale>0:
 from fiber_soccer.native_bfm import NativeActor,load_native
 actor=NativeActor(load_native(a.device))
 action_scale=torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=a.device)*1.5
 default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device)
 joint_weight=torch.ones(29,device=a.device);joint_weight[:12]=a.leg_weight
train={};valid={};provenance=[];schema=None;label_audits=[]
contact_flags={id(train):{},id(valid):{}}
approach_flags={id(train):{},id(valid):{}}
for path in a.data:
 d=torch.load(path,map_location='cpu',weights_only=True);m=d['metadata']
 source_schema=m['schema']
 expected=SCHEMA_V4 if a.task_history else source_schema
 if schema is None:schema=expected
 assert expected==schema,'Mixed observation schemas require fresh aligned collection'
 if a.task_history:assert source_schema==SCHEMA_V3,'History expansion requires v3 source states'
 assert m['seed']!=811314,'Development evaluation must not enter training'
 provenance.append(m);rows=d['rows'];x=torch.stack([r['x'] for r in rows]);z=torch.stack([r.get('label_z',r['z']) for r in rows]);episode=torch.stack([r['episode'] for r in rows]);target=torch.stack([r['target'] for r in rows]);T,N=x.shape[:2]
 if a.task_history:
  assert all('expert_observation' in row for row in rows),'Task histories are missing from this source'
  x=with_task_history(x,torch.stack([row['expert_observation'] for row in rows]))
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
 contact=torch.zeros_like(episode,dtype=torch.bool)
 if a.contact_sample_fraction>0 or a.pass_approach_sample_fraction>0:
  assert schema in ('soccer_state_v2_950',SCHEMA_V3,SCHEMA_V4),'Contact sampling requires an explicit kicked-phase observation'
  contact=observed_contact_window(x,episode)
 approach=(x[:,:,933:935].norm(dim=-1)<=.6)&(x[:,:,939]>.5)&(x[:,:,940]<.5)&~contact if a.pass_approach_sample_fraction>0 and m['skill']=='pass' else torch.zeros_like(contact)
 split=max(1,N*3//4)
 for pool,section in [(train,slice(None,split)),(valid,slice(split,None))]:
  values=tuple(v[:,section].flatten(0,1) for v in parts)
  if m['skill'] in pool:values=tuple(torch.cat((old,new)) for old,new in zip(pool[m['skill']],values))
  pool[m['skill']]=values
  marked=contact[:len(parts[0]),section].flatten()
  flags=contact_flags[id(pool)]
  flags[m['skill']]=torch.cat((flags[m['skill']],marked)) if m['skill'] in flags else marked
  marked_approach=approach[:len(parts[0]),section].flatten()
  aflags=approach_flags[id(pool)]
  aflags[m['skill']]=torch.cat((aflags[m['skill']],marked_approach)) if m['skill'] in aflags else marked_approach
assert set(train)=={'pass','shoot'}
if a.resume:
 header=torch.load(a.resume,map_location='cpu',weights_only=False)
 assert header.get('planner_arch') in ('flow_endpoint_v1','direct_latent_v1')
 if a.architecture=='flow':assert header['planner_arch']=='flow_endpoint_v1'
 model=load_planner(header,a.device)
 if schema==SCHEMA_V4 and model.schema==SCHEMA_V3:
  assert header['planner_arch']=='flow_endpoint_v1','Input expansion currently requires a flow source'
  histories=torch.cat([v[0][:,1200:2400] for v in train.values()])
  model=expand_task_history(model,histories.mean(0),histories.std(0))
 if a.architecture=='direct' and header['planner_arch']=='flow_endpoint_v1':
  model=DirectLatentPlanner.from_context(model,a.horizon)
 assert model.horizon==a.horizon and model.schema==schema
else:
 model=(DirectLatentPlanner if a.architecture=='direct' else FlowLatentPlanner)(a.horizon,schema=schema,command_conditioning=not a.no_command_conditioning).to(a.device)
 all_x=torch.cat([v[0] for v in train.values()]);model.obs_mean.copy_(all_x.mean(0));model.obs_std.copy_(all_x.std(0).clamp_min(.05))
opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
onset={id(pool):{skill:torch.where(values[0][:,93:930].abs().sum(-1)<1e-8)[0] for skill,values in pool.items()} for pool in (train,valid)}
contact_indices={key:{skill:torch.where(flags)[0] for skill,flags in tasks.items()} for key,tasks in contact_flags.items()}
approach_indices={key:{skill:torch.where(flags)[0] for skill,flags in tasks.items()} for key,tasks in approach_flags.items()}
contact_pool_sizes={name:{skill:len(ids) for skill,ids in contact_indices[id(pool)].items()} for name,pool in [('train',train),('valid',valid)]}
config=dict(action_target=a.action_target,pass_approach_sample_fraction=a.pass_approach_sample_fraction,approach_radius=.6,approach_sampling_scope='pass_training_only_launched_not_kicked_outside_contact_window',approach_pool_sizes={name:{skill:len(ids) for skill,ids in approach_indices[id(pool)].items()} for name,pool in [('train',train),('valid',valid)]},task_history_input=a.task_history,observation_schema=schema,flow_weight=a.flow_weight,resume=str(a.resume),contact_pool_sizes=contact_pool_sizes,contact_sample_fraction=a.contact_sample_fraction,contact_sampling_scope='training_only',contact_window_frames_before=10,contact_window_frames_after=5,confidence_normalization='within_each_skill' if a.label_error_scale>0 else 'disabled',label_error_scale=a.label_error_scale,label_audits=label_audits,method='direct_latent_regression' if a.architecture=='direct' else 'conditional_endpoint_flow_mlp',seed=a.seed,data=provenance,horizon=a.horizon,width=model.width,
    action_auxiliary=a.action_weight>0,action_weight=a.action_weight,leg_weight=a.leg_weight,auxiliary_path='direct_latent_prediction' if a.architecture=='direct' else 'full_8_step_deployed_sampler',command_conditioning=model.command_conditioning,reset_sample_fraction=.25,flow_steps=getattr(model,'flow_steps',None),noise_seed=getattr(model,'noise_seed',None),
    execution='direct_replan_every_20ms_first_latent' if a.architecture=='direct' else 'fixed_noise_replan_every_20ms_first_latent',validation='held-out lanes, not autonomous success')
(a.output/'config.json').write_text(json.dumps(config,indent=2));start=time.monotonic();best=float('inf')
def batch(pool,size):
 out=[]
 for skill in ['pass','shoot']:
  values=pool[skill];starts=onset[id(pool)][skill];count=size//4 if len(starts) else 0
  contacts=contact_indices[id(pool)][skill]
  contact_count=int(size*a.contact_sample_fraction) if pool is train and len(contacts) else 0
  approaches=approach_indices[id(pool)][skill]
  approach_count=int(size*a.pass_approach_sample_fraction) if pool is train and skill=='pass' and len(approaches) else 0
  ids=torch.randint(len(values[0]),(size-count-contact_count-approach_count,))
  if count:ids=torch.cat((ids,starts[torch.randint(len(starts),(count,))]))
  if contact_count:ids=torch.cat((ids,contacts[torch.randint(len(contacts),(contact_count,))]))
  if approach_count:ids=torch.cat((ids,approaches[torch.randint(len(approaches),(approach_count,))]))
  out.append(tuple(v[ids].to(a.device) for v in values))
 return tuple(torch.cat([v[k] for v in out]) for k in range(4))
def loss(values):
 x,target,mask,expert_targets=values
 with torch.set_grad_enabled(torch.is_grad_enabled() and a.flow_weight>0):
  if a.architecture=='direct':
   prediction=model.raw(x)
  else:
   noise=torch.randn_like(target);time=torch.rand(len(x),device=a.device)
   mixed=(1-time[:,None,None])*noise+time[:,None,None]*target
   prediction=model.endpoint(mixed,time,x)
 squared=(prediction-target).square().mean(-1)
 flow_loss=balanced_skill_mean(squared,mask) if a.label_error_scale>0 else (squared*mask).sum()/mask.sum()
 action_loss=flow_loss.new_zeros(())
 if a.action_weight>0:
  sampled=model(x)
  predicted_action=actor(x[:,:64],x[:,64:93],x[:,93:930],sampled[:,0])
  if a.action_target=='expert':
   error=(default+predicted_action.clamp(-1,1)*action_scale-expert_targets).square()
  else:
   with torch.no_grad():teacher_action=actor(x[:,:64],x[:,64:93],x[:,93:930],target[:,0])
   error=((predicted_action.clamp(-1,1)-teacher_action.clamp(-1,1))*action_scale).square()
  per_sample=(error*joint_weight).sum(-1).div(joint_weight.sum())
  action_loss=balanced_skill_mean(per_sample,mask[:,0]) if a.label_error_scale>0 else per_sample.mean()
 return a.flow_weight*flow_loss+a.action_weight*action_loss,flow_loss,action_loss
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
  if a.architecture=='direct':
   row['latent_regression_loss']=row.pop('flow_loss');row['validation_latent']=row.pop('validation_endpoint')
  print(row,flush=True)
  with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  payload=dict(planner=model.state_dict(),planner_arch='direct_latent_v1' if a.architecture=='direct' else 'flow_endpoint_v1',observation_schema=schema,
      horizon=a.horizon,width=model.width,command_conditioning=model.command_conditioning,flow_steps=getattr(model,'flow_steps',None),noise_seed=getattr(model,'noise_seed',None),
      latent_interval=1,latent_execution='hold',updates=update,config=config)
  torch.save(payload,a.output/'latest.pt')
  if float(score)<best:best=float(score);torch.save(payload,a.output/'best.pt')
if actor is not None:assert all(p.grad is None for p in actor.parameters())
(a.output/'complete.json').write_text(json.dumps(dict(updates=a.updates,best_validation_objective=best)))
