"""Reproduce the source-clamp correction audit for the first continuous shooting dataset."""
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from fiber_soccer.bridge import Bridge
from fiber_soccer.paths import ROOT
root=ROOT
p=root/'runs/planner_sequence_shoot_r1_training'
old=torch.load(p/'shoot_labels.pt',map_location='cpu',weights_only=True)
new=torch.load(p/'shoot_labels_clamped.pt',map_location='cpu',weights_only=True)
for key in ['x','z','tick','environment','expert_raw_action','action_scale','action_offset']:
 assert torch.equal(old[key],new[key]),key
order=new['metadata']['joint_order'];inverse=torch.argsort(torch.tensor(order))
bridge=Bridge.__new__(Bridge);bridge.order=order
bridge.term=SimpleNamespace(_scale=new['action_scale'][:,inverse],_offset=new['action_offset'][:,inverse],cfg=SimpleNamespace(clip=None))
expected=bridge.expert_targets(new['expert_raw_action'])
assert torch.equal(expected,new['target'])
changed=old['target']!=new['target']
assert int(changed.sum())==int((new['expert_raw_action'].abs()>10).sum())==8
source=Path('/workspace/g1-rollingball/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_kick_env.py')
assert 'action = torch.clamp(action.to(self.device), -10.0, 10.0)' in source.read_text()
report=dict(states=len(new['x']),changed_target_components=int(changed.sum()),changed_states=int(changed.any(-1).sum()),
 max_target_correction_rad=float((new['target']-old['target']).abs().max()),
 corrected_targets_match_native_bridge=True,inputs_and_raw_expert_outputs_preserved=True,
 source_environment_file=str(source),source_environment_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 old_label_sha256=hashlib.sha256((p/'shoot_labels.pt').read_bytes()).hexdigest(),
 corrected_label_sha256=hashlib.sha256((p/'shoot_labels_clamped.pt').read_bytes()).hexdigest())
output=root/'runs/planner_sequence_shoot_label_clamp_audit.json'
if output.exists():
 assert json.loads(output.read_text())==report,'Preserve a differing prior audit'
else:
 output.write_text(json.dumps(report,indent=2))
print(report)
