"""First episode per environment; capture terminal state before autoreset."""
import json,types,hashlib
from pathlib import Path
import torch
from onnx2torch import convert
from .bridge import Bridge
from .paths import ROOT, ASSETS
from .lpp import Projector,execute_latent
from .latent_oracle import fit_latent
from .pass_measurement import first_kick_samples

@torch.no_grad()
def evaluate(env,expert,obs,args):
 torch.set_num_threads(4)
 oracle_steps=getattr(args,"oracle_latent_steps",0)
 semantic_weight=getattr(args,'oracle_semantic_weight',0.)
 semantic_mode=oracle_steps>0 and semantic_weight>0
 controlled=bool(args.projector_checkpoint or oracle_steps)
 oracle_records=[]
 root=ROOT;core=env.unwrapped;device=core.device;n=core.num_envs
 initial_state=torch.cat((core.scene['robot'].data.root_state_w,core.scene['robot'].data.joint_pos,core.scene['ball'].data.root_state_w),-1).detach().cpu()
 torch.save(initial_state,args.output/'initial_state.pt')
 scenario_hash=hashlib.sha256(initial_state.numpy().tobytes()).hexdigest()
 ufo_mode=False;planner_mode=False
 if args.projector_checkpoint:
  header=torch.load(args.projector_checkpoint,map_location='cpu',weights_only=False)
  ufo_mode=header.get('backbone')=='ufo'
  planner_mode='planner' in header
 if ufo_mode:
  from .ufo import UFOBridge,Actor,load_export
  bridge=UFOBridge(core)
 else:bridge=Bridge(core,ASSETS)
 rows={};original=core._reset_idx;trace=[]
 clipping_error=torch.zeros(n,device=device);tracking_sum=torch.zeros(n,device=device);tracking_count=torch.zeros(n,device=device);clipping_count=torch.zeros(n,device=device);active=torch.ones(n,dtype=torch.bool,device=device);diagnostics=[];windows=[]
 impact_samples={}
 def measure_impact():
  cmd=core.command_manager.get_term('motion')
  eligible=active.clone()
  if impact_samples:eligible[list(impact_samples)]=False
  # Never substitute a later ball when the first event lacked a valid sample.
  if hasattr(cmd,'_quality_ball_resample_count'):eligible &= cmd._quality_ball_resample_count==0
  valid,values=first_kick_samples(core.scene['ball'].data.root_lin_vel_w[:,:2],cmd.ball_target_dir_w[:,:2],cmd.ball_target_speed,cmd._ball_kicked,eligible)
  for i in valid.nonzero().flatten().tolist():
   impact_samples[i]=dict(tick=tick,**dict(zip(('speed_mps','forward_speed_mps','direction_error_deg','target_speed_mps','relative_forward_speed_error'),values[i].tolist())),
       ball_velocity_xy=core.scene['ball'].data.root_lin_vel_w[i,:2].tolist(),
       target_direction_xy=cmd.ball_target_dir_w[i,:2].tolist(),
       robot_root_quaternion_wxyz=core.scene['robot'].data.root_quat_w[i].tolist())
 def capture(self,ids):
  measure_impact()
  cmd=self.command_manager.get_term('motion')
  for i in ids.tolist():
   if i in rows:continue
   metrics={k:float(v[i]) for k,v in cmd.metrics.items() if k.startswith('quality_') or 'settle' in k}
   event_counts={name:int(getattr(cmd,name)[i]) for name in ('_quality_evaluated_kick_count','_quality_accurate_kick_count','_quality_kick_count') if hasattr(cmd,name)}
   rows[i]=dict(clipping_target_mse=float(clipping_error[i]/tracking_count[i].clamp_min(1)),action_target_mse=float(tracking_sum[i]/tracking_count[i].clamp_min(1)),target_clipped_fraction=float(clipping_count[i]/tracking_count[i].clamp_min(1)),event_counts=event_counts,environment=i,terminated=bool(self.reset_terminated[i]),timeout=bool(self.reset_time_outs[i]),duration=float(self.episode_length_buf[i]*self.step_dt),termination_terms={name:bool(self.termination_manager.get_term(name)[i]) for name in self.termination_manager.active_terms},metrics=metrics)
  return original(ids)
 core._reset_idx=types.MethodType(capture,core)
 if controlled:
  interval=1 if oracle_steps else 5
  execution='oracle_50hz' if oracle_steps else 'slerp'
  if args.projector_checkpoint and (not oracle_steps or args.oracle_init_projector):
   ck=torch.load(args.projector_checkpoint,map_location=device,weights_only=False)
   execution='oracle_projector_init_50hz' if oracle_steps else ck.get('latent_execution','slerp')
   interval=1 if oracle_steps else ck.get('latent_interval',5)
   if interval not in (1,5):raise ValueError('Unsupported checkpoint interval')
   if planner_mode:
    from .planner import observation as planner_observation
    from .planner_policy import load_soccer_policy
    if interval!=1 or oracle_steps:raise ValueError('Unified planner evaluation requires autonomous 50Hz execution')
    unified_policy=load_soccer_policy(ck,device,post_kick_adapter_delay=getattr(args,'post_kick_adapter_delay',0),recovery_adapter_shoot_only=getattr(args,'recovery_adapter_shoot_only',False))
    p=unified_policy.planner
   else:
    from .multiskill import load_projector
    p=load_projector(ck,device).eval()
   if ck.get('multiskill'):
    from .multiskill import ConditionedProjector
    p=ConditionedProjector(p,getattr(args,'skill','pass'))
  if ufo_mode:
   actor=Actor(device)
   encoder=load_export(ASSETS/'ufo/g1_policy/exported/backward_encoder.onnx',device)
   stand=encoder(*bridge.encoder_inputs()).detach()
  else:
   if semantic_mode:
    from .native_bfm import load_native,NativeActor
    semantic_model=load_native(device);actor=NativeActor(semantic_model)
   else:actor=unified_policy.actor if planner_mode else convert(str(ASSETS/'actor.onnx')).to(device).eval().requires_grad_(False)
   stand=torch.tensor(json.loads((ASSETS/'z_bank.json').read_text())['stand'],device=device).expand(n,-1)
  previous=stand.clone();endpoint=stand.clone();alive=torch.ones(n,dtype=torch.bool,device=device)
 for tick in range(args.steps):
  measure_impact()
  if controlled:
   teacher=bridge.expert_targets(expert.act_inference(obs['policy'],obs['rma_obs']))
   if oracle_steps:
    if args.oracle_init_projector:endpoint=p(torch.cat((obs['policy'],obs['rma_obs']),-1))
    if semantic_mode:
     from .native_backward import privileged_from_env
     from .semantic_oracle import fit_regularized_latent
     prealign=getattr(args,'oracle_prealign_steps',0)
     if prealign:endpoint,_=fit_latent(actor,bridge.observe(),teacher,endpoint,bridge.targets,prealign)
     prior=semantic_model.backward_map(dict(state=bridge.observe()[0],privileged_state=privileged_from_env(core,bridge)))
     endpoint,fit=fit_regularized_latent(actor,bridge.observe(),teacher,endpoint,prior,bridge.targets,semantic_weight,oracle_steps)
    else:endpoint,fit=fit_latent(actor,bridge.observe(),teacher,endpoint,bridge.targets,oracle_steps)
    oracle_records.append(dict(tick=tick,**fit))
    a=actor(*bridge.observe(),endpoint)
   else:
    if tick%interval==0:
     previous=endpoint.clone()
     if planner_mode:
      planner_command=unified_policy(planner_observation(core,bridge,getattr(args,'skill','pass'),schema=p.schema,backward_model=unified_policy.actor.model,task_observation=torch.cat((obs['policy'],obs['rma_obs']),-1)))
      endpoint=planner_command['latent']
     else:endpoint=p(torch.cat((obs['policy'],obs['rma_obs']),-1))
     current_window=dict(execution=execution,previous=previous.cpu().clone(),endpoint=endpoint.cpu().clone(),states=[],targets=[],masks=[])
    a=planner_command['action_normalized'] if planner_mode else actor(*bridge.observe(),execute_latent(previous,endpoint,(tick%interval+1)/interval,execution))
   desired=bridge.targets(a);action=bridge.env_action(desired)
   realized=bridge.expert_targets(action)
   tracking_sum+=active*(realized-teacher).square().mean(-1)
   clipping_error+=active*(realized-desired).square().mean(-1)
   clipping_count+=active*((realized-desired).abs().amax(-1)>1e-5)
   tracking_count+=active
   if not oracle_steps:
    current_window["states"].append(tuple(v.cpu().clone() for v in bridge.observe()))
    current_window["targets"].append(teacher.cpu().clone())
    current_window["masks"].append(active.cpu().clone())
    if tick%interval==interval-1:windows.append(current_window)
   if tick%5==0:
    cmd=core.command_manager.get_term('motion')
    diagnostics.append(dict(tick=tick,active=active.cpu().clone(),projector_obs=torch.cat((obs['policy'],obs['rma_obs']),-1).cpu(),bfm_inputs=tuple(v.cpu() for v in bridge.observe()),expert_targets=teacher.cpu(),bfm_targets=desired.cpu(),effective_targets=realized.cpu(),z=endpoint.cpu(),ball_position=core.scene['ball'].data.root_pos_w.cpu().clone(),robot_position=core.scene['robot'].data.root_pos_w.cpu().clone()))
  else:action=expert.act_inference(obs['policy'],obs['rma_obs'])
  robot=core.scene['robot'].data;ball=core.scene['ball'].data
  trace.append(torch.cat((robot.root_pos_w,robot.root_quat_w,robot.joint_pos,ball.root_pos_w),-1).cpu())
  obs,_,term,trunc,_=env.step(action);done=term|trunc
  active &= ~done
  if controlled:
   # Validate affine/clipping diagnostics against actual processed commands.
   # Delay is intentionally excluded: compare the queued command, not its execution time.
   if active.any():
    torch.testing.assert_close(realized[active], bridge.term.processed_actions[:,bridge.order][active], rtol=1e-5, atol=1e-6)
   bridge.advance(a,done);previous[done]=stand[done];endpoint[done]=stand[done]
   if planner_mode:unified_policy.reset(done)
  if tick%100==0:print('EVAL_PROGRESS',tick,len(rows),flush=True)
  if len(rows)==n:break
 core._reset_idx=original
 result=list(rows.values());summary=dict(initial_state_sha256=scenario_hash,seed=args.seed,checkpoint=str(args.projector_checkpoint or args.checkpoint),completed=len(result),requested=n,steps=tick+1,terminated=sum(r['terminated'] for r in result))
 summary['recovery_adapter_shoot_only']=getattr(args,'recovery_adapter_shoot_only',False)
 if planner_mode and unified_policy.shoot_adapter_after_pass_only:summary['shoot_adapter_after_pass_only']=True
 summary['post_kick_adapter_delay']=getattr(args,'post_kick_adapter_delay',0)
 summary['controller']='unified_latent_planner' if planner_mode else ('projector' if controlled else 'expert')
 summary['backbone']=('ufo' if ufo_mode else 'ah_bfm') if controlled else 'expert'
 summary['latent_interval']=interval if controlled else None
 summary['latent_execution']=execution if controlled else None
 summary['expert_assisted_oracle']=bool(oracle_steps)
 if planner_mode and hasattr(unified_policy,'solver_records'):
  records=unified_policy.solver_records
  summary['controller']='predicted_target_latent_solver'
  summary['predicted_target_solver_steps']=unified_policy.solver_steps
  summary['predicted_target_solver_backend']=unified_policy.solver_backend
  if unified_policy.refinement_scope!='all':summary['predicted_target_refinement_scope']=unified_policy.refinement_scope
  summary['predicted_target_solver_expert_targets_used']=False
  latency=torch.tensor([r['latency_seconds'] for r in records])
  summary['policy_latency_seconds']={k:float(v) for k,v in zip(['median','p95','maximum'],torch.quantile(latency,torch.tensor([.5,.95,1.])))}
  summary['real_time_50hz_verified']=False
  (args.output/'predicted_target_solver.json').write_text(json.dumps(records))
 summary['oracle_optimization_steps']=oracle_steps
 summary['oracle_semantic_weight']=semantic_weight
 summary['oracle_prealign_steps']=getattr(args,'oracle_prealign_steps',0)
 summary['oracle_init_projector']=bool(args.oracle_init_projector)
 summary['event_counts']={key:sum(r.get('event_counts',{}).get(key,0) for r in result) for key in ('_quality_evaluated_kick_count','_quality_accurate_kick_count','_quality_kick_count')}
 for key in ['quality_kick_success','quality_ball_direction_success_10deg','quality_post_kick_survival_ratio','quality_wrong_foot_kick']:
  summary[key]=sum(r['metrics'].get(key,0) for r in result)/max(1,len(result))
 (args.output/'episodes.json').write_text(json.dumps(result,indent=2));(args.output/'summary.json').write_text(json.dumps(summary,indent=2))
 (args.output/'first_kick_velocity.json').write_text(json.dumps(dict(samples=impact_samples,missing_environment_ids=sorted(set(range(n))-set(impact_samples)),sampling='First observed kicked state with horizontal ball speed >=0.25m/s, before first resample; passive diagnostic, up to one control tick delay'),indent=2))
 if oracle_records:(args.output/'oracle_fits.json').write_text(json.dumps(oracle_records))
 if diagnostics:torch.save(diagnostics,args.output/'action_diagnostics.pt')
 if windows:torch.save(windows,args.output/'latent_windows.pt')
 torch.save(dict(states=torch.stack(trace),joint_names=core.scene['robot'].joint_names,dt=core.step_dt),args.output/'trace.pt')
 print('EVALUATION',summary,flush=True)
