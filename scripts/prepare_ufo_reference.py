"""Convert audited UFO motion export to explicit Isaac reset/reference tensors."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/ufo_reference_export'));p.add_argument('--assets',type=Path,default=Path('assets/ufo/g1_policy'));p.add_argument('--output',type=Path,default=Path('runs/ufo_reference_export/isaac_reference.npz'));a=p.parse_args()
if a.output.exists():raise RuntimeError('Refuse overwrite')
r=torch.load(a.source/'reference.pt',map_location='cpu',weights_only=True)['reference']
z=np.asarray(joblib.load(a.source/'tracking_inference/zs_7.pkl'),dtype=np.float32)
meta=json.loads((a.assets/'exported/FBcprAuxModel.meta.json').read_text())
assert len(r['dof_pos'])==len(z)+1
# Official reset uses frame0; command0 is inferred from reference frame1.
root=torch.cat((r['ref_body_pos'][:,0],r['ref_body_rots'][:,0][:,[3,0,1,2]],r['ref_body_vels'][:,0],r['ref_body_angular_vels'][:,0]),-1).numpy()
np.savez(a.output,root_state_wxyz=root,joint_pos=r['dof_pos'].numpy(),joint_vel=r['ref_dof_vel'].numpy(),z=z,joint_names=np.asarray(meta['control_joint_names']),dt=np.array(.02),latent_target_offset_frames=np.array(1))
print(json.dumps(dict(output=str(a.output),frames=len(root),latent_frames=len(z),quaternion='wxyz',initial_frame=0,latent_target_offset_frames=1)))
