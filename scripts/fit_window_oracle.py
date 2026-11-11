"""Fixed-state five-step oracle: diagnose temporal latent constraints, not success."""
import argparse,json
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.lpp import slerp,project

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--iterations',type=int,default=500)
p.add_argument('--max-windows',type=int,default=128)
p.add_argument('--device',default='cuda:0')
p.add_argument('--initialization',choices=['policy','random'],default='policy')
p.add_argument('--mode',choices=['slerp','hold','independent'],default='slerp')
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
root=Path(__file__).resolve().parents[1];device=a.device
rows=torch.load(a.run/'latent_windows.pt',weights_only=True)
old=torch.cat([r['previous'] for r in rows]).to(device)
initial=torch.cat([r['endpoint'] for r in rows]).to(device)
mask=torch.cat([torch.stack(r['masks'],1) for r in rows]).to(device)
targets=torch.cat([torch.stack(r['targets'],1) for r in rows]).to(device)
states=[tuple(torch.cat([r['states'][h][k] for r in rows]).to(device) for k in range(3)) for h in range(5)]
valid=torch.where(mask.sum(1)>0)[0]
# Deterministic uniform coverage across the evaluation, without success selection.
idx=valid[torch.linspace(0,len(valid)-1,min(a.max_windows,len(valid)),device=device).long()]
actor=convert(str(root/'assets/actor.onnx')).eval().to(device).requires_grad_(False)
meta=json.loads((root/'assets/actor_meta.json').read_text());scene=json.loads((root/'assets/pickup_scene.json').read_text())
default=torch.tensor(scene['default_q'],device=device);scale=torch.tensor(meta['action_rescale'],device=device)*1.5
results=[]
for batch in idx.split(32):
 seed=initial[batch].clone()
 if a.mode=='independent':seed=torch.stack([slerp(old[batch],seed,(h+1)/5) for h in range(5)],1)
 if a.initialization=='random':seed=project(torch.randn_like(seed))
 z=torch.nn.Parameter(seed);opt=torch.optim.Adam([z],lr=.05)
 for step in range(a.iterations+1):
  pred=torch.stack([default+actor(*(x[batch] for x in states[h]),(slerp(old[batch],project(z),(h+1)/5) if a.mode=='slerp' else project(z[:,h] if a.mode=='independent' else z))).clamp(-1,1)*scale for h in range(5)],1)
  loss=((pred-targets[batch]).square().sum(-1)*mask[batch]).sum()/(29*mask[batch].sum())
  if step==0:before=loss.item()
  if step==a.iterations:break
  opt.zero_grad();loss.backward();opt.step()
 results.append(dict(windows=len(batch),valid_steps=int(mask[batch].sum()),before=before,after=loss.item()))
 print(results[-1],flush=True)
count=sum(r['valid_steps'] for r in results)
report=dict(description='Fixed visited states; offline temporal constraint diagnostic, not closed-loop',mode=a.mode,initialization=a.initialization,iterations=a.iterations,batches=results,weighted_before=sum(r['before']*r['valid_steps'] for r in results)/count,weighted_after=sum(r['after']*r['valid_steps'] for r in results)/count)
(a.run/(f'window_oracle_{a.mode}_{a.iterations}'+('' if a.initialization=='policy' else '_random')+'.json')).write_text(json.dumps(report,indent=2))
