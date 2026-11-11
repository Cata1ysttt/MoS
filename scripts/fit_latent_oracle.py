"""Offline per-state latent upper-bound diagnostic; never a deployable policy."""
import argparse,json
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.lpp import project
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--iterations',type=int,default=200);p.add_argument('--max-samples',type=int);p.add_argument('--output',type=Path);p.add_argument('--device',default='cuda:0');a=p.parse_args()
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4);torch.manual_seed(42)
actor=convert(str(root/'assets/actor.onnx')).eval().to(a.device).requires_grad_(False)
inputs=torch.load(a.data/'actor_inputs.pt',weights_only=True);targets=torch.cat(torch.load(a.data/'expert_targets.pt',weights_only=True)).to(a.device)
x=[torch.cat([r[k] for r in inputs]).to(a.device) for k in range(3)]
if a.max_samples:
 targets=targets[:a.max_samples];x=[v[:a.max_samples] for v in x]
meta=json.loads((root/'assets/actor_meta.json').read_text());scene=json.loads((root/'assets/pickup_scene.json').read_text())
scale=torch.tensor(meta['action_rescale'],device=a.device)*1.5;default=torch.tensor(scene['default_q'],device=a.device)
results=[];joint_errors=[];latents=[]
for start in range(0,len(targets),32):
 end=min(start+32,len(targets));z=torch.nn.Parameter(torch.randn(end-start,256,device=a.device));opt=torch.optim.Adam([z],lr=.05)
 for step in range(a.iterations+1):
  pred=default+actor(*(v[start:end] for v in x),project(z))*scale
  loss=(pred-targets[start:end]).square().mean()
  if step==0:initial=float(loss.detach())
  if step==a.iterations:break
  opt.zero_grad();loss.backward();opt.step()
 joint_errors.append((pred.detach()-targets[start:end]).square().cpu());latents.append(project(z.detach()).cpu())
 results.append(dict(start=start,initial_mse=initial,final_mse=float(loss.detach())))
 print(results[-1],flush=True)
output=a.output or a.data/'oracle.json'
errors=torch.cat(joint_errors)
torch.save(dict(z=torch.cat(latents),per_joint_squared_error=errors),output.with_suffix('.pt'))
output.write_text(json.dumps(dict(description='Independent latent per visited expert state; not closed-loop skill',iterations=a.iterations,batches=results,mean_mse=float(errors.mean()),per_joint_mse=dict(zip(meta['joint_names'],errors.mean(0).tolist()))),indent=2))
