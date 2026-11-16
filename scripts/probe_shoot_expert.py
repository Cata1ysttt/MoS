"""Validate the supplied shooting export against the retained kick configuration."""
import argparse, importlib.util, json, sys
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--projector-checkpoint',type=Path);p.add_argument('--evaluate',action='store_true');p.add_argument('--seed',type=int,default=811313)
p.add_argument('--oracle-latent-steps',type=int,default=0,help='Expert-assisted diagnostic; not autonomous policy performance')
p.add_argument('--oracle-init-projector',action='store_true')
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
AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
try:
 import torch, gymnasium as gym, onnxruntime as ort
 import whole_body_tracking.tasks
 from whole_body_tracking.tasks.tracking import mdp
 from isaaclab.managers import ObservationTermCfg as ObsTerm
 from isaaclab.utils import configclass
 path='/workspace/batchmpc_issaclab/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/kick_env_cfg.py'
 spec=importlib.util.spec_from_file_location('retained_shoot_cfg',path);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 @configclass
 class Obs(m.PassEventObservationsCfg):
  @configclass
  class PolicyCfg(m.PassEventObservationsCfg.PolicyCfg):
   ball_pos=ObsTerm(func=mdp.ball_position_body_yaw)
   ball_vel=ObsTerm(func=mdp.ball_relative_velocity_body_yaw)
   ball_target_dir=ObsTerm(func=mdp.ball_target_direction_body_yaw)
  @configclass
  class RmaObsCfg(m.PassEventObservationsCfg.RmaObsCfg):
   ball_obs=ObsTerm(func=mdp.ball_rma_observation_body_yaw)
  policy:PolicyCfg=PolicyCfg()
  rma_obs:RmaObsCfg=RmaObsCfg()
 cfg=m.G1FlatKickRefEnvCfg();cfg.observations=Obs();cfg.scene.num_envs=a.num_envs;cfg.sim.device=a.device;cfg.seed=a.seed;cfg.commands.motion.debug_vis=False
 a.output.mkdir(parents=True,exist_ok=False)
 from whole_body_tracking.tasks.tracking.tracking_kick_env import TrackingKickEnv
 env=TrackingKickEnv(cfg=cfg);obs,_=env.reset()
 options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
 session=ort.InferenceSession('/workspace/fiber-soccer/body_yaw_49999.onnx',sess_options=options,providers=['CPUExecutionProvider'])
 meta=session.get_modelmeta().custom_metadata_map
 audit={'obs_shapes':{k:list(v.shape) for k,v in obs.items()},'metadata':meta,'source_config':path}
 (a.output/'audit.json').write_text(json.dumps(audit,indent=2))
 assert obs['policy'].shape[1]==1030 and obs['rma_obs'].shape[1]==150,audit['obs_shapes']
 import numpy as np
 class Expert:
  def act_inference(self,policy_obs,rma_obs):
   result=np.concatenate([session.run(['actions'],{'obs':policy_obs[i:i+1].cpu().numpy(),'rma_obs':rma_obs[i:i+1].cpu().numpy(),'time_step':np.zeros((1,1),np.float32)})[0] for i in range(len(policy_obs))])
   return torch.tensor(result,device=env.device)
 if a.planner_collect:
  from fiber_soccer.planner_collection import collect
  a.skill='shoot'
  collect(env,Expert(),obs,a)
  env.close()
  raise SystemExit(0)
 if a.joint_train:
  from fiber_soccer.joint_training import train as train_joint
  a.skill='shoot'
  train_joint(env,Expert(),obs,a)
  env.close()
  raise SystemExit(0)
 if a.evaluate:
  from fiber_soccer.evaluation import evaluate
  a.checkpoint=Path('/workspace/fiber-soccer/body_yaw_49999.onnx');a.skill='shoot';a.latent_interval=1;a.latent_execution='hold'
  evaluate(env,Expert(),obs,a)
  env.close()
  raise SystemExit(0)
 from fiber_soccer.bridge import Bridge
 from fiber_soccer.paths import ASSETS
 b=Bridge(env,ASSETS);records=[];done_count=0
 for t in range(a.steps):
  with torch.no_grad():
   act=np.concatenate([session.run(['actions'],{'obs':obs['policy'][i:i+1].cpu().numpy(),'rma_obs':obs['rma_obs'][i:i+1].cpu().numpy(),'time_step':np.zeros((1,1),np.float32)})[0] for i in range(a.num_envs)])
   act=torch.tensor(act,device=env.device)
   records.append(dict(inputs=tuple(x.cpu().clone() for x in b.observe()),target=b.expert_targets(act).cpu(),policy_obs=obs['policy'].cpu().clone(),rma_obs=obs['rma_obs'].cpu().clone()))
   obs,_,term,trunc,_=env.step(act);done=term|trunc;done_count+=int(term.sum())
   b.advance((b.expert_targets(act)-b.default)/(b.scale*6),done)
   if t%100==0:print('PROGRESS',t,done_count,flush=True)
 torch.save(records,a.output/'teacher_records.pt')
 cmd=env.command_manager.get_term('motion')
 summary={'steps':a.steps,'terminations':done_count,'metrics':{k:str(v) for k,v in cmd.metrics.items() if 'quality' in k or 'kick' in k}}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True);env.close()
finally:app.close()
