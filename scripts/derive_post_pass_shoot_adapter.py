"""Limit a trained shooting adapter to episodes with an observed passing contact."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source_sha=hashlib.sha256(a.source.read_bytes()).hexdigest()
complete=json.loads((a.source.parent/'complete.json').read_text())
assert complete['checkpoint_sha256']==source_sha and not complete['smoke_test']
header=torch.load(a.source,map_location='cpu',weights_only=False)
assert header['adapter_skill']==1 and not header.get('adapter_post_kick_only',False)
assert not header.get('shoot_adapter_after_pass_only',False)
a.output.mkdir(parents=True,exist_ok=False)
payload=dict(header,shoot_adapter_after_pass_only=True)
torch.save(payload,a.output/'final.pt')
restored=torch.load(a.output/'final.pt',map_location='cpu',weights_only=False)
assert set(restored)==set(header)|{'shoot_adapter_after_pass_only'}
for key,value in header.items():
    if key=='planner':
        assert set(value)==set(restored[key])
        assert all(torch.equal(v,restored[key][k]) for k,v in value.items())
    else:assert value==restored[key],key
manifest=dict(method='post_pass_shooting_runtime_gate',source=str(a.source),source_sha256=source_sha,
    checkpoint_sha256=hashlib.sha256((a.output/'final.pt').read_bytes()).hexdigest(),
    weights_and_training_metadata_unchanged=True,shoot_adapter_after_pass_only=True,
    smoke_test=False,new_training_updates=0,
    scope='Runtime-context metadata only. Native tasks and continuous trajectory equivalence require separate verification.')
(a.output/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
