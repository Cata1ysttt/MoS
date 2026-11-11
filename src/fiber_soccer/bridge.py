"""Website-exact AH-BFM state/history and physical-target mapping."""
import json
from pathlib import Path
import torch

class Bridge:
 def __init__(self,env,assets):
  self.env=env;self.robot=env.scene['robot'];self.device=env.device
  meta=json.loads((Path(assets)/'actor_meta.json').read_text());scene=json.loads((Path(assets)/'pickup_scene.json').read_text())
  self.ids=[self.robot.joint_names.index(n) for n in meta['joint_names']]
  self.default=torch.tensor(scene['default_q'],device=self.device);self.scale=torch.tensor(meta['action_rescale'],device=self.device)*.25
  self.last=torch.zeros(env.num_envs,29,device=self.device)
  self.history=torch.zeros(env.num_envs,10,93,device=self.device)
  names=env.action_manager.active_terms
  if len(names)!=1:raise ValueError(names)
  self.term=env.action_manager.get_term(names[0])
  self.order=[self.term._joint_names.index(n) for n in meta['joint_names']]
  self.inverse=torch.argsort(torch.tensor(self.order,device=self.device))
  self.observe(append=True)
 def observe(self,append=False):
  d=self.robot.data;q=d.joint_pos[:,self.ids]-self.default;v=d.joint_vel[:,self.ids];w=d.root_ang_vel_b*.25;g=d.projected_gravity_b
  frame=torch.cat((self.last,w,q,v,g),-1)
  if append:self.history=torch.cat((frame[:,None],self.history[:,:9]),1)
  hist=torch.cat([self.history[:,1:,a:b].flatten(1) for a,b in [(0,29),(29,32),(32,61),(61,90),(90,93)]],-1)
  return torch.cat((q,v,g,w),-1),self.last.clone(),hist
 def targets(self,action):return self.default+(action*6).clamp(-6,6)*self.scale
 def expert_targets(self,action):
  # JointAction applies affine scale then physical clipping.
  target=action.clamp(-10,10)*self.term._scale+self.term._offset
  if self.term.cfg.clip is not None:target=torch.clamp(target,self.term._clip[:,:,0],self.term._clip[:,:,1])
  return target[:,self.order]
 def env_action(self,target):return (target[:,self.inverse]-self.term._offset)/self.term._scale
 def advance(self,action,done):
  self.last=(action*6).clamp(-6,6).detach();self.last[done]=0;self.history[done]=0
  self.observe(append=True)
