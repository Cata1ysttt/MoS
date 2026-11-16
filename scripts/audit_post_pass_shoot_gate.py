"""Check the full policy's context gating on fixed real continuous shooting inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner_policy import SoccerPolicy
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--gated',type=Path,required=True)
p.add_argument('--labels',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args();assert not a.output.exists()
torch.set_num_threads(4)
d=torch.load(a.labels,map_location='cpu',weights_only=True)
mask=(d['environment']>=d['metadata']['num_envs']*3//4)&(d['x'][:,940]<.5)
x=d['x'][mask][::max(1,int(mask.sum())//8)][:8].to(a.device)
assert len(x)==8 and (x[:,-2:].argmax(-1)==1).all()
original=x.clone()
base=SoccerPolicy(a.base,a.device);source=SoccerPolicy(a.source,a.device)
policy=SoccerPolicy(a.gated,a.device)
assert policy.shoot_adapter_after_pass_only and not source.shoot_adapter_after_pass_only
old=base(x);adapted=source(x);assert not torch.equal(old['latent'],adapted['latent'])
keys=['latent','joint_targets','action_normalized']
def same(actual,expected):
    for key in keys:assert torch.equal(actual[key],expected[key]),key
same(policy(x),old)
contact=x.clone();contact[:,940]=1
# A shooting contact followed by recovery passing must not arm the adapter.
policy(contact,skill='shoot');policy(contact,skill='pass');same(policy(x),old)
policy.reset()
# Half the slots pass, half shoot, at the first observed contact.
skills=torch.arange(len(x),device=x.device)%2
policy(contact,skill=skills)
actual=policy(x)
for key in keys:
    assert torch.equal(actual[key][::2],adapted[key][::2]),key
    assert torch.equal(actual[key][1::2],old[key][1::2]),key
reset=torch.zeros(len(x),device=x.device,dtype=torch.bool);reset[0]=True
policy.reset(reset);actual=policy(x)
for key in keys:
    assert torch.equal(actual[key][0],old[key][0]),key
    assert torch.equal(actual[key][2::2],adapted[key][2::2]),key
assert torch.equal(x,original)
report=dict(states=len(x),state_sha256=hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest(),
    gated_checkpoint_sha256=hashlib.sha256(a.gated.read_bytes()).hexdigest(),
    standalone_shoot_matches_base=True,post_pass_shoot_matches_source=True,
    post_shot_recovery_does_not_arm=True,partial_reset_clears_history=True,input_unchanged=True,
    scope='Full SoccerPolicy latent/action/target comparison on real states with controlled command/contact histories; verifies routing, not physical rollouts.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
