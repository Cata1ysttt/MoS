"""Build contiguous expert windows without crossing resets; retain outcome labels."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
rows=torch.load(a.run/'expert_dense.pt',weights_only=True)
windows=[]
for start in range(0,len(rows)-4,5):
 chunk=rows[start:start+5]
 assert [r['tick'] for r in chunk]==list(range(start,start+5))
 valid=torch.ones_like(chunk[0]['done'],dtype=torch.bool);masks=[]
 for r in chunk:
  masks.append(valid.clone());valid &= ~r['done']
 windows.append(dict(obs=chunk[0]['projector_obs'],states=[r['bfm_inputs'] for r in chunk],targets=torch.stack([r['targets'] for r in chunk],1),mask=torch.stack(masks,1),distance=torch.stack([(r['ball_position'][:,:2]-r['robot_position'][:,:2]).norm(dim=-1) for r in chunk],1)))
torch.save(windows,a.run/'expert_windows.pt')
report=dict(windows=len(windows),environments=rows[0]['done'].numel(),valid_steps=sum(int(w['mask'].sum()) for w in windows),note='Expert trajectories, not success-filtered; history action inferred from expert physical targets, distribution differs from BFM execution.')
(a.run/'windows_summary.json').write_text(json.dumps(report,indent=2));print(report)
