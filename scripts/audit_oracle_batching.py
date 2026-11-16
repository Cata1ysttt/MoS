"""Verify per-state latent fitting stability across batches using a recorded training state."""
import argparse,json,torch
from pathlib import Path
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.semantic_oracle import fit_regularized_latent
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
torch.set_num_threads(4);device=a.device
d=torch.load(a.data,map_location='cpu',weights_only=True)
for tick,row in enumerate(d['rows']):
 cut=len(row['x'])*3//4
 x=row['x'];mask=row['teacher_executed']&(x[:,939]>.5)&(x[:,940]<.5)&(x[:,933:935].norm(dim=-1)<.6)
 mask[cut:]=False
 if mask.any():
  lane=int(mask.nonzero()[0]);break
else:raise ValueError('No teacher-executed near-ball training state found')
x=x[lane:lane+1].to(device);state=(x[:,:64],x[:,64:93],x[:,93:930]);z=row['previous_label_z'][lane:lane+1].to(device);target=row['target'][lane:lane+1].to(device);prior=x[:,944:1200]
actor=NativeActor(load_native(device));default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=device);scale=torch.tensor(json.loads((ASSETS/'actor_meta.json').read_text())['action_rescale'],device=device)*1.5
physical=lambda value:default+value.clamp(-1,1)*scale
reference=None;result={}
for size in [1,2048]:
 fitted,audit=fit_regularized_latent(actor,tuple(v.repeat(size,1) for v in state),target.repeat(size,1),z.repeat(size,1),prior.repeat(size,1),physical,.01,32)
 if reference is None:reference=fitted[:1].clone()
 with torch.no_grad():
  difference=physical(actor(*state,fitted[:1]))-physical(actor(*state,reference))
 result[str(size)]=dict(**audit,latent_max_difference=float((fitted[:1]-reference).abs().max()),physical_target_rms_difference=float(difference.square().mean().sqrt()))
assert result['2048']['physical_target_rms_difference']<1e-4, 'Batch-dependent physical targets remain'
report=dict(data=str(a.data),training_lanes_only=True,tick=tick,lane=lane,repeated_identical_state=True,results=result)
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
