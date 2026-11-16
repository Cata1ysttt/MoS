"""Audit deployment origin gating on held-out real observations; no rollout claim."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner import with_task_history
from fiber_soccer.planner_policy import SoccerPolicy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args()
assert not a.output.exists()
torch.set_num_threads(4)
digest=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
data=torch.load(a.data,map_location='cpu',weights_only=True)
for row in data['rows']:
    mask=(torch.arange(len(row['x']))>=len(row['x'])*3//4)&(row['x'][:,940]>.5)
    if mask.sum()>=8:
        x=with_task_history(row['x'][mask][:8],row['expert_observation'][mask][:8]).to(a.device)
        break
else:
    raise RuntimeError('No held-out contact batch')
assert (x[:,-2:].argmax(-1)==0).all()
untouched=x.clone()
base=SoccerPolicy(a.base,a.device)
full=SoccerPolicy(a.checkpoint,a.device)
policy=SoccerPolicy(a.checkpoint,a.device,post_kick_adapter_delay=10,recovery_adapter_shoot_only=True)
original=base(x);adapted=full(x)
assert not torch.equal(original['latent'],adapted['latent'])
keys=['latent','joint_targets','action_normalized']
for _ in range(12):
    actual=policy(x)
    for key in keys:
        assert torch.equal(actual[key],original[key]),key
# Start a new contact event with half the slots commanded to shoot. These command
# overrides test routing only; the physical states/history are held fixed.
policy.reset()
commands=torch.arange(len(x),device=x.device)%2
for _ in range(11):
    policy(x,skill=commands)
actual=policy(x,skill='pass')
for key in keys:
    assert torch.equal(actual[key][::2],original[key][::2]),key
    assert torch.equal(actual[key][1::2],adapted[key][1::2]),key
policy.reset(torch.ones(len(x),device=x.device,dtype=torch.bool))
for _ in range(12):
    actual=policy(x)
    for key in keys:
        assert torch.equal(actual[key],original[key]),key
assert torch.equal(x,untouched)
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()==digest
report=dict(checkpoint_sha256=digest,states=len(x),state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
    passing_contact_matches_base=True,shoot_origin_recovery_matches_adapted=True,reset_clears_origin=True,
    input_unchanged=True,scope='Fixed real passing post-contact observations with controlled command history; deployment routing audit, not physical shooting recovery or rollout success.')
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
