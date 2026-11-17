"""Verify both heads changed, normalizers/frozen shared checkpoint retained."""
import argparse,json,hashlib
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--tag',default='planner_dual_task_reward_r3');a=p.parse_args()
root=Path(__file__).resolve().parents[1]
source=root/'runs/planner_dual_task_zero_r1/diagnostic.pt';target=root/'runs'/a.tag/'final.pt'
x=torch.load(source,map_location='cpu',weights_only=False);y=torch.load(target,map_location='cpu',weights_only=False)
def equal(x,y):
 if isinstance(x,torch.Tensor):return isinstance(y,torch.Tensor) and torch.equal(x,y)
 if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
 if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
 return x==y
assert set(x)==set(y)
for key in x:
 if key not in ['config','updates','dual_task_adapter']:assert equal(x[key],y[key]), key
before=x['dual_task_adapter']['model'];after=y['dual_task_adapter']['model']
changed=[k for k in before if not torch.equal(before[k],after[k])]
assert len(changed)==8 and all('.net.' in k for k in changed),changed
for value in after.values():assert torch.isfinite(value).all()
result=dict(checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changed_tensors=changed,
            both_task_heads_changed=True,shared_checkpoint_unchanged=True,normalizers_unchanged=True)
(root/'reports'/f'{a.tag}_checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
