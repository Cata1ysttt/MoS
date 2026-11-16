"""Derive a spatially gated adapter checkpoint without modifying learned weights."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import load_planner

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--inner',type=float,default=.4)
p.add_argument('--outer',type=float,default=1.2)
a=p.parse_args()
assert (a.checkpoint.parent/'complete.json').exists(),'Source training must be complete'
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert ck['planner_arch']=='flow_endpoint_adapter_v1'
model=load_planner(ck,'cpu')
model.set_adapter_ball_distance_gate(a.inner,a.outer)
for key,value in ck['planner'].items():assert torch.equal(value,model.state_dict()[key])
provenance=dict(operation='ball_distance_gate_without_training',source=str(a.checkpoint),
    source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),inner=a.inner,outer=a.outer)
ck['adapter_ball_distance_gate']=[a.inner,a.outer]
ck['config']=dict(ck['config'],derived=provenance,adapter_ball_distance_gate=[a.inner,a.outer])
a.output.mkdir(parents=True,exist_ok=False)
torch.save(ck,a.output/'best.pt')
(a.output/'config.json').write_text(json.dumps(ck['config'],indent=2))
(a.output/'complete.json').write_text(json.dumps(provenance,indent=2))
print(json.dumps(provenance,indent=2))
