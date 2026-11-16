"""Label recorded continuous shooting states offline; never execute expert actions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from fiber_soccer.planner import SCHEMA_V4

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--states',type=Path,required=True)
p.add_argument('--expert',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert not a.output.exists(),'Preserve existing labels'
data=torch.load(a.states,map_location='cpu',weights_only=True)
m=data['metadata'];x=data['x']
assert m['format']=='flat_sequence_shoot_states_v1' and m['schema']==SCHEMA_V4
assert not m['expert_loaded'] and not m['expert_actions_executed']
assert m['initial_skill']=='pass' and m['scene_base']=='shoot'
assert m['physical_target_mapping']=='raw_action_affine_no_clip'
assert (x[:,-2:].argmax(-1)==1).all() and x.shape[-1]==2406
assert (x[:,2380:2400]==0).all(),'Shooting history padding changed'
options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
session=ort.InferenceSession(str(a.expert),sess_options=options,providers=['CPUExecutionProvider'])
rows=[]
for i in range(len(x)):
    raw=session.run(['actions'],dict(obs=x[i:i+1,1200:2230].numpy(),
        rma_obs=x[i:i+1,2230:2380].numpy(),time_step=np.zeros((1,1),np.float32)))[0]
    assert raw.shape==(1,29) and np.isfinite(raw).all()
    rows.append(torch.from_numpy(raw.copy()))
    if (i+1)%1024==0:print('LABEL_PROGRESS',i+1,len(x),flush=True)
raw=torch.cat(rows)
# The source JointPositionAction applies raw scale/offset without the bridge's
# optional +/-10 diagnostic clamp. Its actual clip=None contract was recorded.
target=raw[:,m['joint_order']]*data['action_scale']+data['action_offset']
assert torch.isfinite(target).all()
data['target']=target;data['expert_raw_action']=raw
m.update(label_source='original_shoot_expert_offline',offline_expert_loaded=True,
    expert_sha256=hashlib.sha256(a.expert.read_bytes()).hexdigest(),
    source_states_sha256=hashlib.sha256(a.states.read_bytes()).hexdigest())
torch.save(data,a.output)
report=dict(states=len(x),environments=int(data['environment'].unique().numel()),
    precontact_states=int((x[:,940]<.5).sum()),postcontact_states=int((x[:,940]>.5).sum()),
    expert_sha256=m['expert_sha256'],source_states_sha256=m['source_states_sha256'],
    label_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
    maximum_raw_action=float(raw.abs().max()),raw_action_values_above_10=int((raw.abs()>10).sum()),
    purpose=m['purpose'],scope='Offline teacher targets on autonomous post-pass shooting states; no expert actions executed and no new success-rate measurement.')
a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
