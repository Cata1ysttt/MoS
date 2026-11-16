"""Compare candidate backward priors on training states only."""
import argparse,json
from pathlib import Path
import torch
from fiber_soccer.native_bfm import load_native,NativeActor
from fiber_soccer.native_backward import backward_observation,MAPPING
from fiber_soccer.semantic_oracle import fit_regularized_latent
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(811460)
d=torch.load(a.data,map_location='cpu',weights_only=True);rows=d['rows'];chosen=torch.linspace(0,len(rows)-1,8).long().tolist()
values={key:torch.cat([rows[i][key][:12] for i in chosen]) for key in ['x','target','body_pos','body_quat','body_vel','body_omega']}
initial=torch.cat([rows[i].get('label_z',rows[i]['z'])[:12] for i in chosen])
model=load_native();actor=NativeActor(model);meta=json.loads((ASSETS/'actor_meta.json').read_text());default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q']);scale=torch.tensor(meta['action_rescale'])*.25
x=values['x'];state=(x[:,:64],x[:,64:93],x[:,93:930]);b=backward_observation(d['metadata']['body_names'],*(values[k] for k in ['body_pos','body_quat','body_vel','body_omega']),meta['joint_names'])
prior=model.backward_map(dict(state=state[0],privileged_state=b))
def physical(a):return default+(a*6).clamp(-6,6)*scale
results=[]
for init_name,seed in [('teacher',initial),('backward',prior)]:
 for weight in [0.,.001,.01]:
  z,metric=fit_regularized_latent(actor,state,values['target'],seed,prior,physical,weight,64)
  metric.update(initialization=init_name,prior_cosine=float((z*prior).sum(-1).div(256).mean()))
  results.append(metric);print(metric,flush=True)
Path(a.output).write_text(json.dumps(dict(mapping=MAPPING,author_mapping_equivalence_verified=False,samples=len(x),training_seed=d['metadata']['seed'],results=results),indent=2))
