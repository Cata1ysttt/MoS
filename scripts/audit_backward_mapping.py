"""Numerical plausibility audit; explicitly does not establish author equivalence."""
import argparse,json
import torch
from fiber_soccer.native_bfm import load_native,NativeActor
from fiber_soccer.native_backward import backward_observation,MAPPING
from fiber_soccer.paths import ROOT,ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(4)
d=torch.load(a.data,map_location='cpu',weights_only=True);meta=json.loads((ASSETS/'actor_meta.json').read_text());rows=d['rows'];ids=torch.linspace(0,len(rows)-1,min(16,len(rows))).long().tolist()
fields={key:torch.cat([rows[i][key] for i in ids]) for key in ['x','body_pos','body_quat','body_vel','body_omega']}
b=backward_observation(d['metadata']['body_names'],*(fields[k] for k in ['body_pos','body_quat','body_vel','body_omega']),meta['joint_names'])
model=load_native();actor=NativeActor(model);x=fields['x']
with torch.no_grad():
 z=model.backward_map(dict(state=x[:,:64],privileged_state=b))
 normalized=model._obs_normalizer(dict(privileged_state=b))['privileged_state']
 shift=fields['body_pos'].new_tensor([12.,-7.,0.]);b2=backward_observation(d['metadata']['body_names'],fields['body_pos']+shift,*(fields[k] for k in ['body_quat','body_vel','body_omega']),meta['joint_names'])
 torch.testing.assert_close(b,b2,atol=3e-6,rtol=1e-5)
 default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q']);scale=torch.tensor(meta['action_rescale'])*.25
 action=actor(x[:,:64],x[:,64:93],x[:,93:930],z)
 targets=torch.cat([rows[i]['target'] for i in ids])
 mse=float((default+(action*6).clamp(-6,6)*scale-targets).square().mean())
 labels=torch.cat([rows[i].get('label_z',rows[i]['z']) for i in ids])
 cos=float((z*labels).sum(-1).div(256).mean())
result=dict(mapping=MAPPING,author_equivalence_verified=False,samples=len(x),xy_translation_invariant=True,
 backward_norm_min=float(z.norm(dim=-1).min()),backward_norm_max=float(z.norm(dim=-1).max()),
 normalized_abs_median=float(normalized.abs().median()),normalized_abs_p99=float(normalized.abs().quantile(.99)),
 normalized_fraction_above_10=float((normalized.abs()>10).float().mean()),backward_action_mse=mse,mean_label_cosine=cos,
 warning='Frozen-state plausibility only; no autonomous capability or original mapping equivalence established.')
from pathlib import Path
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
