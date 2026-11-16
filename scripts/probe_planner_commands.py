"""Frozen-state command sensitivity, not a task success measurement."""
import argparse,json
import torch
from fiber_soccer.planner_policy import SoccerPolicy
from fiber_soccer.planner import SCHEMA_V4,with_task_history
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
policy=SoccerPolicy(a.checkpoint);d=torch.load(a.data,map_location='cpu',weights_only=True)
rows=d['rows'];inputs=[with_task_history(r['x'],r['expert_observation']) if policy.planner.schema==SCHEMA_V4 else r['x'] for r in rows]
ids=torch.linspace(0,len(rows)-1,8).long().tolist();x=torch.cat([inputs[i][:12] for i in ids])
with torch.no_grad():
 first=policy(x,skill='pass',target_speed=2.);second=policy(x,skill='shoot',target_speed=6.)
 skill_only=policy(x,skill='pass');baseline=policy(x,skill='shoot')
 result=dict(samples=len(x),checkpoint=a.checkpoint,observation_schema=policy.planner.schema,task_history_held_fixed=policy.planner.schema==SCHEMA_V4,
  skill_only_action_rms=float((skill_only['joint_targets']-baseline['joint_targets']).square().mean().sqrt()),
  skill_and_speed_action_rms=float((first['joint_targets']-second['joint_targets']).square().mean().sqrt()),
  skill_and_speed_latent_rms=float((first['latent']-second['latent']).square().mean().sqrt()))
# Probe target direction on pre-contact held-out lanes, preserving robot state,
# ball state, speed and skill. This measures sensitivity, not correct ball control.
all_x=torch.cat([x[len(x)*3//4:] for x in inputs])
near=(all_x[:,933:935].norm(dim=-1)<1.) & (all_x[:,940]<.5)
if policy.planner.schema not in ('soccer_state_v2_950','soccer_state_v3_backward_1206',SCHEMA_V4):near.zero_()
precontact=all_x[near]
if len(precontact):
 selected=precontact[torch.linspace(0,len(precontact)-1,min(256,len(precontact))).long()]
 direction=selected[:,-6:-3];angle=torch.tensor(10.*torch.pi/180)
 def rotated(sign):
  c=angle.cos();s=angle.sin()*sign
  return torch.stack((c*direction[:,0]-s*direction[:,1],s*direction[:,0]+c*direction[:,1],direction[:,2]),-1)
 with torch.no_grad():
  left=policy(selected,target_direction=rotated(1))['joint_targets']
  right=policy(selected,target_direction=rotated(-1))['joint_targets']
  delta=left-right
  headings=torch.atan2(precontact[:,-5],precontact[:,-6])*180/torch.pi
  result['direction_probe']=dict(samples=len(selected),rotation_each_side_degrees=10.,
    observed_goal_heading_quantiles_degrees={str(q):float(headings.quantile(q)) for q in [.1,.5,.9]},
    action_rms=float(delta.square().mean().sqrt()),leg_action_rms=float(delta[:,:12].square().mean().sqrt()),
    mean_absolute_joint_delta=dict(zip(policy.joint_names,delta.abs().mean(0).tolist())),
    interpretation='Frozen pre-contact state sensitivity; not a rollout or proof of correct aiming')
from pathlib import Path
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
