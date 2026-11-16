"""Measure recorded expert physical targets clipped by a shared scene's limits.

This is a command-contract diagnostic, not autonomous task performance.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import torch
from fiber_soccer.paths import ASSETS

p=argparse.ArgumentParser()
p.add_argument('--scene-contract',type=Path,required=True)
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4)
contracts=json.loads(a.scene_contract.read_text())
clip=contracts['shared']['actions']['joint_pos']['clip']
assert clip is not None,'Shared action configuration has no physical target clipping'
names=json.loads((ASSETS/'actor_meta.json').read_text())['joint_names']
bounds=[]
for name in names:
    matches=[value for pattern,value in clip.items() if re.fullmatch(pattern,name)]
    assert len(matches)==1,(name,matches)
    bounds.append(matches[0])
bounds=torch.tensor(bounds)
results=[]
for path in a.data:
    data=torch.load(path,map_location='cpu',weights_only=True)
    targets=torch.stack([row['target'] for row in data['rows']]).flatten(0,1)
    assert targets.shape[-1]==len(names) and torch.isfinite(targets).all()
    error=targets-targets.clamp(bounds[:,0],bounds[:,1])
    results.append(dict(data=str(path),skill=data['metadata']['skill'],states=len(targets),
        fraction_any_clip=float((error.abs()>1e-6).any(-1).float().mean()),
        mean_physical_mse=float(error.square().mean()),mean_leg_mse=float(error[:,:12].square().mean()),
        joint_clip_fraction={name:float(value) for name,value in zip(names,(error.abs()>1e-6).float().mean(0))}))
report=dict(scene_contract=str(a.scene_contract),scene_contract_sha256=hashlib.sha256(a.scene_contract.read_bytes()).hexdigest(),
    results=results,scope='Recorded expert physical targets in AH joint order; all recorded states, not only successful expert episodes. No simulated success is measured.')
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
