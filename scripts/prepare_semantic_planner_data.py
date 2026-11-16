"""Derive explicitly experimental backward-conditioned labels from training data."""
import argparse,json,time
from pathlib import Path
import torch
from fiber_soccer.native_bfm import NativeActor,load_native
from fiber_soccer.native_backward import backward_observation,MAPPING
from fiber_soccer.semantic_oracle import fit_regularized_latent
from fiber_soccer.planner import SCHEMA_V2,SCHEMA_V3
from fiber_soccer.paths import ASSETS
p=argparse.ArgumentParser();p.add_argument('--data',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--weight',type=float,default=.01);p.add_argument('--iterations',type=int,default=64);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
model=load_native(a.device);actor=NativeActor(model);meta=json.loads((ASSETS/'actor_meta.json').read_text());default=torch.tensor(json.loads((ASSETS/'pickup_scene.json').read_text())['default_q'],device=a.device);scale=torch.tensor(meta['action_rescale'],device=a.device)*.25
physical=lambda action:default+(action*6).clamp(-6,6)*scale
start=time.monotonic();summaries=[]
for path in a.data:
 assert (path.parent/'complete.json').exists(),'Wait for collection completion'
 data=torch.load(path,map_location='cpu',weights_only=True);assert data['metadata']['schema']==SCHEMA_V2
 assert data['metadata']['seed']!=811314
 if data['metadata'].get('oracle_steps',0)>0:
  data['metadata'].setdefault('oracle_solver_version','sum_per_state_batch_stop_v1' if data['metadata'].get('oracle_gradient_reduction')=='sum_per_state' else 'mean_batch_stop_legacy')
 rows=data['rows'];stats=[];per_state_errors=[]
 for begin in range(0,len(rows),16):
  group=rows[begin:begin+16];n=len(group[0]['x'])
  x=torch.cat([r['x'] for r in group]).to(a.device)
  body=[torch.cat([r[k] for r in group]).to(a.device) for k in ['body_pos','body_quat','body_vel','body_omega']]
  b=backward_observation(data['metadata']['body_names'],*body,meta['joint_names'])
  with torch.no_grad():prior=model.backward_map(dict(state=x[:,:64],privileged_state=b))
  initial=torch.cat([r.get('label_z',r['z']) for r in group]).to(a.device)
  target=torch.cat([r['target'] for r in group]).to(a.device)
  z,fit=fit_regularized_latent(actor,(x[:,:64],x[:,64:93],x[:,93:930]),target,initial,prior,physical,a.weight,a.iterations)
  with torch.no_grad():
   action_error=(physical(actor(x[:,:64],x[:,64:93],x[:,93:930],z))-target).square().mean(-1).cpu()
  per_state_errors.append(action_error)
  for j,row in enumerate(group):
   row['x']=torch.cat((x[j*n:(j+1)*n,:-6],prior[j*n:(j+1)*n],x[j*n:(j+1)*n,-6:]),-1).cpu()
   row['previous_label_z']=row.get('label_z',row['z'])
   row['label_z']=z[j*n:(j+1)*n].cpu()
   row['label_action_mse']=action_error[j*n:(j+1)*n]
  stats.append(fit)
  if begin%160==0:print(path.parent.name,begin,fit,'seconds',time.monotonic()-start,flush=True)
 data['metadata'].update(schema=SCHEMA_V3,backward_mapping=MAPPING,author_mapping_equivalence_verified=False,
   semantic_weight=a.weight,semantic_iterations=a.iterations,semantic_gradient_reduction='sum_per_state',original_trajectory=str(path),
   label_source='offline_regularized_expert_action_inversion',expert_assisted=True)
 # z remains the actual executed latent; only supervised label_z was replaced.
 out=a.output/(data['metadata']['skill']+'_trajectory.pt');torch.save(data,out)
 errors=torch.cat(per_state_errors)
 summary=dict(skill=data['metadata']['skill'],frames=len(rows),states=len(rows)*n,
              label_action_mse_quantiles={str(q):float(errors.quantile(q)) for q in [0.,.5,.9,.99,1.]},
              fraction_label_action_mse_above_001=float((errors>.01).float().mean()),
              mean_fitted_action_mse=sum(v['action_mse'] for v in stats)/len(stats),
              mean_semantic_mse=sum(v['semantic_mse'] for v in stats)/len(stats),file=str(out))
 summaries.append(summary);print(summary,flush=True)
(a.output/'complete.json').write_text(json.dumps(dict(mapping=MAPPING,author_mapping_equivalence_verified=False,results=summaries),indent=2))
