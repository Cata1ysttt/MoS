"""On-policy configurable-window action-only LPP. Expert and BFM remain frozen."""
import json,time,os
from pathlib import Path
import torch
from onnx2torch import convert
from .bridge import Bridge
from .paths import ROOT, ASSETS
from .lpp import Projector,project,execute_latent,alignment_loss
from .expert_data import expert_windows,replay_error
from .supervision import proximity_weights
from .diagnostic_replay import load_replay, action_loss

def train(env,expert,obs,args):
 root=ROOT;device=env.unwrapped.device;torch.set_num_threads(4)
 evidence=json.loads((ASSETS/'verification.json').read_text())
 real=json.loads((ASSETS/'real_verification.json').read_text())
 # Engineering gate: retain strict failures in audit; bound physical target error to <0.01 deg.
 if not (real['strict_violations']==0 and evidence['max_abs_error']<5e-5):raise RuntimeError('Actor engineering equivalence gate not passed')
 bfm=convert(str(ASSETS/'actor.onnx')).eval().to(device).requires_grad_(False)
 bridge=Bridge(env.unwrapped,ASSETS);n=env.unwrapped.num_envs
 interval=args.latent_interval
 pobs=lambda o:torch.cat((o['policy'],o['rma_obs']),-1)
 projector=Projector(pobs(obs).shape[-1]).to(device)
 if args.resume_projector:
  resume=torch.load(args.resume_projector,map_location=device,weights_only=False)
  projector.load_state_dict(resume['projector'])
 optimizer=torch.optim.AdamW(projector.parameters(),lr=3e-4,weight_decay=1e-4)
 stand=torch.tensor(json.loads((ASSETS/'z_bank.json').read_text())['stand'],device=device).expand(n,-1)
 if args.resume_projector:optimizer.load_state_dict(resume['optimizer'])
 if args.learning_rate is not None:
  for group in optimizer.param_groups:group['lr']=args.learning_rate
 diagnostic=load_replay(args.diagnostic_replay,device) if args.diagnostic_replay else None
 replay=None
 if args.expert_replay:
  replay=expert_windows(torch.load(args.expert_replay/'expert_dense.pt',weights_only=True),interval=interval)
  replay={k:tuple(x.to(device) for x in v) if k=='states' else v.to(device) for k,v in replay.items()}
  split=max(1,int(replay['env_ids'].max())+1-3)
  replay_ids=torch.where(replay['env_ids']<split)[0]
  replay_weights=torch.where(replay['contact'][replay_ids],4.,1.)
 previous=stand.clone();steps=0;iteration=0;start=time.monotonic();initial_parameter=next(projector.parameters()).detach().clone()
 metadata=dict(method='action_only_lpp',environments=n,expert=str(args.checkpoint),resume=str(args.resume_projector),additional_budget=args.steps,source_steps=int(resume.get('cumulative_steps',resume['steps'])) if args.resume_projector else 0,control_dt=env.unwrapped.step_dt,latent_interval=interval,optimization_passes=args.optimization_passes,latent_execution=args.latent_execution)
 metadata.update(near_ball_weight=args.near_ball_weight,near_ball_radius=args.near_ball_radius,replay_target_samples_per_update=320,expert_replay=str(args.expert_replay),expert_replay_weight=args.expert_replay_weight if replay is not None else 0,learning_rate=optimizer.param_groups[0]['lr'])
 metadata.update(diagnostic_replay=str(args.diagnostic_replay),diagnostic_replay_weight=args.diagnostic_replay_weight,diagnostic_reserved_envs=8)
 (args.output/'config.json').write_text(json.dumps(metadata,indent=2))
 import swanlab
 swanlab.init(project=os.environ.get('SWANLAB_PROJECT', 'MoS'),workspace=os.environ.get('SWANLAB_WORKSPACE') or None,name=args.output.name,config=metadata)
 total_terminated=0;total_timeouts=0
 while steps<args.steps:
  records=[]
  for window in range(50//interval):
   initial=pobs(obs).detach();old=previous.clone();valid=torch.ones(n,device=device);states=[];labels=[];masks=[];weights=[]
   with torch.no_grad():endpoint=project(projector(initial)+torch.randn_like(old)*.1)
   for h in range(interval):
    with torch.no_grad():
     weights.append(proximity_weights(bridge.robot.data.root_pos_w,env.unwrapped.scene['ball'].data.root_pos_w,args.near_ball_radius,args.near_ball_weight))
     state=bridge.observe();a=bfm(*state,torch.where(valid[:,None].bool(),execute_latent(old,endpoint,(h+1)/interval,args.latent_execution),stand))
     teacher=expert.act_inference(obs['policy'],obs['rma_obs'])
     labels.append(bridge.expert_targets(teacher).detach());states.append(tuple(x.detach().clone() for x in state));masks.append(valid.clone())
     obs,r,term,trunc,info=env.step(bridge.env_action(bridge.targets(a)));done=term|trunc
     total_terminated+=int(term.sum());total_timeouts+=int(trunc.sum())
     bridge.advance(a,done);valid=valid*(~done);steps+=n
   previous=torch.where(valid[:,None].bool(),endpoint,stand).detach()
   records.append((initial,old,states,torch.stack(labels,1),torch.stack(masks,1),torch.stack(weights,1)))
  # Shuffle interval/environment pairs; batch is a whole masked five-step window.
  x=torch.cat([r[0] for r in records]);old=torch.cat([r[1] for r in records]);target=torch.cat([r[3] for r in records]);mask=torch.cat([r[4] for r in records])*torch.cat([r[5] for r in records])
  states=[tuple(torch.cat([r[2][h][k] for r in records]) for k in range(3)) for h in range(interval)]
  losses=[];replay_losses=[];diagnostic_losses=[]
  for epoch in range(args.optimization_passes):
   for idx in torch.randperm(len(x),device=device).chunk(4):
    optimizer.zero_grad();batch_loss=0.;denominator=(29*mask[idx].sum()).clamp_min(1)
    # Bound actor activation memory while preserving the full minibatch objective.
    for micro in idx.split(64):
     z=projector(x[micro]);pred=[]
     for h in range(interval):pred.append(bridge.targets(bfm(*(v[micro] for v in states[h]),execute_latent(old[micro],z,(h+1)/interval,args.latent_execution))))
     loss=((target[micro]-torch.stack(pred,1)).square().sum(-1)*mask[micro]).sum()/denominator
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite LPP loss')
     loss.backward();batch_loss+=float(loss.detach())
    if replay is not None:
     ridx=replay_ids[torch.multinomial(replay_weights,320//interval,replacement=True)]
     rloss=replay_error(projector,bfm,replay,ridx,bridge.targets,stand[:1],args.latent_execution).mean()
     if not torch.isfinite(rloss):raise RuntimeError('Nonfinite replay loss')
     (rloss*args.expert_replay_weight).backward();replay_losses.append(float(rloss.detach()))
    if diagnostic is not None:
     dloss=action_loss(projector,bfm,diagnostic,bridge.targets,batch_size=64)
     if not torch.isfinite(dloss):raise RuntimeError('Nonfinite diagnostic replay loss')
     (dloss*args.diagnostic_replay_weight).backward();diagnostic_losses.append(float(dloss.detach()))
    torch.nn.utils.clip_grad_norm_(projector.parameters(),1);optimizer.step();losses.append(batch_loss)
  iteration+=1;row=dict(steps=steps,iteration=iteration,loss=sum(losses)/len(losses),elapsed=time.monotonic()-start,terminated=total_terminated,timeouts=total_timeouts)
  # Observe whether proximity weighting actually activates without changing gradients.
  valid_frames=torch.cat([r[4] for r in records])
  frame_weights=torch.cat([r[5] for r in records])
  valid_count=valid_frames.sum().clamp_min(1)
  row['upweighted_valid_fraction']=float(((frame_weights>1)*valid_frames).sum()/valid_count)
  row['mean_valid_supervision_weight']=float((frame_weights*valid_frames).sum()/valid_count)
  if diagnostic_losses:row['diagnostic_replay_loss']=sum(diagnostic_losses)/len(diagnostic_losses)
  if replay_losses:row['replay_loss']=sum(replay_losses)/len(replay_losses)
  with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(row,flush=True);swanlab.log(row,step=steps)
  if iteration%20==0 or steps>=args.steps:
   checkpoint=dict(latent_interval=interval,latent_execution=args.latent_execution,cumulative_steps=metadata['source_steps']+steps,projector=projector.state_dict(),optimizer=optimizer.state_dict(),steps=steps,obs_dim=x.shape[-1])
   torch.save(checkpoint,args.output/'latest.tmp');(args.output/'latest.tmp').replace(args.output/'latest.pt')
   if iteration%80==0 or steps>=args.steps:
    temp=args.output/f'checkpoint_{steps}.tmp';torch.save(checkpoint,temp);temp.replace(args.output/f'checkpoint_{steps}.pt')
 delta=float((next(projector.parameters()).detach()-initial_parameter).abs().max())
 assert delta>0 and all(v.grad is None for v in bfm.parameters())
 (args.output/'parameter_audit.json').write_text(json.dumps(dict(projector_parameter_delta=delta,bfm_gradients_absent=True)))
 swanlab.finish()
