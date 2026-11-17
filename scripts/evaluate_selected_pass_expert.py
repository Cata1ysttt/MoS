"""Load exact expert and saved environment; no training or checkpoint export."""
import argparse,pickle,json,faulthandler
faulthandler.enable()
from pathlib import Path
from fiber_soccer.paths import ASSETS
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--env-config',type=Path,required=True);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps',type=int,default=600)
p.add_argument('--seed',type=int,default=810123);p.add_argument('--resume-projector',type=Path);p.add_argument('--evaluate',action='store_true');p.add_argument('--projector-checkpoint',type=Path);p.add_argument('--train-lpp',action='store_true');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--ufo-native-pd',action='store_true',help='PD-only diagnostic override, reapplied after reset')
p.add_argument('--ufo-train',action='store_true')
p.add_argument('--ufo-teacher-collect',action='store_true')
p.add_argument('--semantic-weight',type=float,default=0.1)
p.add_argument('--collect-dense',action='store_true')
p.add_argument('--oracle-init-projector',action='store_true',help='Diagnostic: initialize each oracle solve from autonomous projector')
p.add_argument('--oracle-latent-steps',type=int,default=0,help='Expert-assisted evaluation or planner training-label collection')
p.add_argument('--expert-replay',type=Path,help='Dense expert trajectory for train-env-only replay')
p.add_argument('--expert-replay-weight',type=float,default=1.0)
p.add_argument('--diagnostic-replay',type=Path,help='Training-only action_diagnostics.pt; reserves final 8 environments')
p.add_argument('--diagnostic-replay-weight',type=float,default=0.5)
p.add_argument('--learning-rate',type=float,help='Explicit override after optimizer restore')
p.add_argument('--latent-execution',choices=['slerp','hold'],default='slerp')
p.add_argument('--latent-interval',type=int,choices=[1,5],default=5)
p.add_argument('--optimization-passes',type=int,default=5)
p.add_argument('--near-ball-weight',type=float,default=1.0,help='Online supervision weight within near-ball-radius; 1 preserves baseline')
p.add_argument('--near-ball-radius',type=float,default=0.6)
p.add_argument('--oracle-semantic-weight',type=float,default=0.)
p.add_argument('--oracle-prealign-steps',type=int,default=0)
p.add_argument('--planner-collect',action='store_true')
p.add_argument('--planner-observation-schema',choices=['soccer_state_v1_945','soccer_state_v2_950','soccer_state_v3_backward_1206'],default='soccer_state_v1_945')
p.add_argument('--planner-student',type=Path,help='Collect labels on unified planner states')
p.add_argument('--planner-teacher-ratio',type=float,default=0.)
p.add_argument('--joint-train',action='store_true')
p.add_argument('--joint-resume',type=Path)
p.add_argument('--joint-replay',type=Path,nargs='*',default=[])
p.add_argument('--pass-anchor',type=Path,default=Path('/workspace/fiber-soccer/runs/passing_50hz_long_s1/latest.pt'))
p.add_argument('--teacher-ratio',type=float,default=.25)
p.add_argument('--post-kick-adapter-delay',type=int,default=0,help='Control steps after observed contact before enabling a post-kick-only adapter')
p.add_argument('--recovery-adapter-shoot-only',action='store_true',help='Enable recovery correction only after a contact first observed under the shooting skill')
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
import math
if not math.isfinite(args.near_ball_weight) or args.near_ball_weight<1:p.error('--near-ball-weight must be finite and >=1')
if not math.isfinite(args.near_ball_radius) or args.near_ball_radius<=0:p.error('--near-ball-radius must be finite and positive')
if args.optimization_passes<1:p.error('--optimization-passes must be positive')
if args.oracle_latent_steps<0 or (args.oracle_latent_steps and not (args.evaluate or args.planner_collect)):p.error('Latent oracle requires --evaluate or --planner-collect and positive steps')
if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate<=0):p.error('learning-rate must be finite and positive')
if not math.isfinite(args.diagnostic_replay_weight) or args.diagnostic_replay_weight<0:p.error('diagnostic-replay-weight must be finite and nonnegative')
if args.diagnostic_replay and args.latent_interval!=1:p.error('Diagnostic replay currently requires latent-interval 1')
if args.oracle_init_projector and (not args.oracle_latent_steps or not args.projector_checkpoint):p.error('oracle-init-projector requires oracle steps and projector checkpoint')
app=AppLauncher(args).app
import torch,gymnasium as gym
import whole_body_tracking.tasks
from fiber_soccer.pass_expert_compat import install
install()
from whole_body_tracking.utils.actor_multi_critic import ActorMultiCritic
try:
 args.output.mkdir(parents=True,exist_ok=False)
 with args.env_config.open('rb') as f:cfg=pickle.load(f)
 with (args.checkpoint.parent/'params/agent.pkl').open('rb') as f:agent_cfg=pickle.load(f)
 if 'mpc_ik_reference_based' not in vars(cfg):cfg.mpc_ik_reference_based=False
 if 'mpc_ik_use_pelvis_velocity' not in vars(cfg):cfg.mpc_ik_use_pelvis_velocity=False
 import whole_body_tracking.tasks.tracking.mdp.rewards as rewards
 code=(ASSETS/'expert_rewards.py').read_text()
 exec(compile(code,'expert_rewards.py','exec'),rewards.__dict__)
 cfg.rewards.ready_pose.func=rewards.rolling_pass_ready_pose_l2
 print('Saved configuration and archived reward implementation loaded',flush=True)
 cfg.commands.motion.debug_vis=False
 cfg.scene.num_envs=args.num_envs;cfg.sim.device=args.device;cfg.seed=args.seed
 env=gym.make('Tracking-Flat-G1-Rolling-Pass-Event-v1-stage2',cfg=cfg)
 print('Environment constructed',flush=True)
 core=env.unwrapped;ball=core.scene['ball']
 def tensor_summary(v):
  v=v.detach().cpu();return dict(shape=list(v.shape),min=float(v.min()),max=float(v.max()),values=v.tolist())
 audit=dict(mass=tensor_summary(ball.root_physx_view.get_masses()),materials=tensor_summary(ball.root_physx_view.get_material_properties()),ball_asset=str(cfg.scene.ball.spawn.usd_path),command={k:v for k,v in vars(cfg.commands.motion).items() if k.startswith('ball_') and isinstance(v,(str,bool,int,float,list,tuple,type(None)))})
 drag=getattr(core,'_booster_ball_drag_forces',None)
 if drag is not None:audit['drag_force']=tensor_summary(drag)
 (args.output/'ball_runtime_audit.json').write_text(json.dumps(audit,indent=2))
 obs,_=env.reset()
 if args.ufo_native_pd:
  from fiber_soccer.ufo_gains import install
  install(core,args.output)
 ac=agent_cfg.to_dict()['policy'];ac.pop('class_name')
 policy=ActorMultiCritic(obs['policy'].shape[-1],obs['critic'].shape[-1],obs['rma_obs'].shape[-1],env.unwrapped.action_manager.total_action_dim,**ac).to(args.device)
 checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
 policy.load_state_dict(checkpoint['model_state_dict'],strict=True);policy.eval().requires_grad_(False)
 if args.planner_collect:
  from fiber_soccer.planner_collection import collect
  args.skill='pass'
  collect(env,policy,obs,args)
  env.close()
  raise SystemExit(0)
 if args.joint_train:
  from fiber_soccer.joint_training import train as train_joint
  args.skill='pass'
  train_joint(env,policy,obs,args)
  env.close()
  raise SystemExit(0)
 if args.ufo_train:
  from fiber_soccer.ufo_training import train
  train(env,policy,obs,args)
  env.close()
  raise SystemExit(0)
 if args.evaluate:
  from fiber_soccer.evaluation import evaluate
  evaluate(env,policy,obs,args)
  env.close()
  raise SystemExit(0)
 if args.train_lpp:
  from fiber_soccer.training import train
  train(env,policy,obs,args)
  env.close()
  raise SystemExit(0)
 from fiber_soccer.bridge import Bridge
 bridge=Bridge(env.unwrapped,ASSETS)
 samples=[];teacher_targets=[];dense=[]
 rows=[]
 for tick in range(args.steps):
  with torch.no_grad():action=policy.act_inference(obs['policy'],obs['rma_obs'])
  if args.collect_dense:
   cmd=env.unwrapped.command_manager.get_term('motion')
   dense.append(dict(tick=tick,projector_obs=torch.cat((obs['policy'],obs['rma_obs']),-1).detach().cpu().clone(),bfm_inputs=tuple(x.detach().cpu().clone() for x in bridge.observe()),targets=bridge.expert_targets(action).detach().cpu().clone(),ball_position=env.unwrapped.scene['ball'].data.root_pos_w.detach().cpu().clone(),robot_position=env.unwrapped.scene['robot'].data.root_pos_w.detach().cpu().clone(),kick_count=cmd._quality_kick_count.detach().cpu().clone(),accurate_count=cmd._quality_accurate_kick_count.detach().cpu().clone()))
  if tick%10==0:
   samples.append(tuple(x.detach().cpu().clone() for x in bridge.observe()))
   teacher_targets.append(bridge.expert_targets(action).detach().cpu())
  obs,reward,terminated,truncated,info=env.step(action)
  if args.collect_dense:dense[-1]['done']=(terminated|truncated).detach().cpu().clone()
  equivalent=(bridge.expert_targets(action)-bridge.default)/(bridge.scale*6)
  bridge.advance(equivalent,terminated|truncated)
  if (terminated|truncated).any():
   rows.append(dict(tick=tick,terminated=int(terminated.sum()),truncated=int(truncated.sum()),info={k:str(v) for k,v in info.items() if k=='log'}))
 if dense:torch.save(dense,args.output/'expert_dense.pt')
 torch.save(samples,args.output/'actor_inputs.pt')
 torch.save(teacher_targets,args.output/'expert_targets.pt')
 (args.output/'episodes_raw.json').write_text(json.dumps(rows,indent=2))
 (args.output/'contract.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),obs={k:list(v.shape) for k,v in obs.items()},control_dt=env.unwrapped.step_dt,joints=env.unwrapped.scene['robot'].joint_names),indent=2))
 env.close()
except Exception:
 import traceback,os
 traceback.print_exc()
 os._exit(1)
finally:app.close()
