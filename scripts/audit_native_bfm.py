"""Check author package provenance and equivalence on actual saved robot states."""
import hashlib,json
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.paths import ROOT,ASSETS
from fiber_soccer.native_bfm import load_native,NativeActor
from fiber_soccer.lpp import project

torch.set_num_threads(4);torch.manual_seed(811400)
pkg=ROOT/'AH-BFM';manifest=json.loads((pkg/'bfm_manifest.json').read_text())
sha=hashlib.sha256((pkg/'AH-BFM.pt').read_bytes()).hexdigest()
assert sha==manifest['export_sha256']
meta=json.loads((pkg/'control_metadata.json').read_text());old=json.loads((ASSETS/'actor_meta.json').read_text())
for k in ['joint_names','action_rescale','control_dt']:assert meta[k]==old[k],k
assert meta['default_joint_positions']==json.loads((ASSETS/'pickup_scene.json').read_text())['default_q']
model=load_native();actor=NativeActor(model)
onnx=convert(str(ASSETS/'actor.onnx')).eval().requires_grad_(False)
checks=[]
for skill,path in [('pass','joint_v2_r2_pass_eval'),('shoot','joint_v2_r2_shoot_eval')]:
 rows=torch.load(ROOT/'runs'/path/'action_diagnostics.pt',map_location='cpu',weights_only=True)
 ids=torch.linspace(0,len(rows)-1,min(16,len(rows))).long().tolist()
 state=tuple(torch.cat([rows[i]['bfm_inputs'][k] for i in ids]) for k in range(3))
 z=torch.cat([rows[i]['z'] for i in ids])
 with torch.no_grad():
  a=actor(*state,z);b=onnx(*state,z)
 checks.append(dict(skill=skill,samples=len(z),max_abs=float((a-b).abs().max()),mse=float((a-b).square().mean())))
 torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
z=project(torch.randn(2,256)).requires_grad_();out=actor(*(x[:2] for x in state),z);out.square().sum().backward()
assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
assert all(p.grad is None for p in model.parameters())
# Synthetic privileged input verifies only API, not simulator semantic mapping.
obs={k:torch.zeros(2,v) for k,v in model.config['obs_dims'].items()}
b=model.backward_map(obs);torch.testing.assert_close(b.norm(dim=-1),torch.full((2,),16.))
result=dict(sha256=sha,actor_checks=checks,latent_gradient=True,backward_api=True,backward_simulator_mapping_verified=False)
out=ROOT/'runs/native_bfm_audit.json';out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
