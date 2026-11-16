"""Frozen-state command sensitivity, not a task success measurement."""
import argparse,json
import torch
from fiber_soccer.planner_policy import SoccerPolicy
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
policy=SoccerPolicy(a.checkpoint);d=torch.load(a.data,map_location='cpu',weights_only=True)
rows=d['rows'];ids=torch.linspace(0,len(rows)-1,8).long().tolist();x=torch.cat([rows[i]['x'][:12] for i in ids])
with torch.no_grad():
 first=policy(x,skill='pass',target_speed=2.);second=policy(x,skill='shoot',target_speed=6.)
 skill_only=policy(x,skill='pass');baseline=policy(x,skill='shoot')
 result=dict(samples=len(x),checkpoint=a.checkpoint,
  skill_only_action_rms=float((skill_only['joint_targets']-baseline['joint_targets']).square().mean().sqrt()),
  skill_and_speed_action_rms=float((first['joint_targets']-second['joint_targets']).square().mean().sqrt()),
  skill_and_speed_latent_rms=float((first['latent']-second['latent']).square().mean().sqrt()))
from pathlib import Path
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
