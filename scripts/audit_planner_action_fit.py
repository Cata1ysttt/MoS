"""Evaluate decoded action fidelity on held-out collection lanes (not task success)."""
import argparse,json
import torch
from pathlib import Path
from fiber_soccer.planner_policy import SoccerPolicy
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
policy=SoccerPolicy(a.checkpoint);d=torch.load(a.data,map_location='cpu',weights_only=True);rows=d['rows'];result=[]
for tick in [0,1,25,100,250,500,750]:
 if tick>=len(rows):continue
 r=rows[tick];x=r['x'];label=r.get('label_z',r['z'])
 with torch.no_grad():
  pred=policy(x);teacher=policy.default+(policy.actor(x[:,:64],x[:,64:93],x[:,93:930],label)*6).clamp(-6,6)*policy.scale
  error=(pred['joint_targets']-teacher).square();n=len(x);cut=n*3//4
  result.append(dict(tick=tick,train_action_mse=float(error[:cut].mean()),validation_action_mse=float(error[cut:].mean()),validation_leg_mse=float(error[cut:,:12].mean()),validation_arm_mse=float(error[cut:,15:].mean())))
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
