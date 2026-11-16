"""Quantify future labels crossing observable ball/command boundaries."""
import argparse
import json
from pathlib import Path
import torch
from fiber_soccer.planner import trajectory_event_ids

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--horizon',type=int,default=5)
a=p.parse_args()
assert a.horizon>=1
torch.set_num_threads(4)
results=[]
for path in a.data:
    assert (path.parent/'complete.json').exists()
    data=torch.load(path,map_location='cpu',weights_only=True)
    rows=data['rows'];x=torch.stack([r['x'] for r in rows]);episode=torch.stack([r['episode'] for r in rows])
    present=['ball_episode' in row for row in rows]
    assert all(present) or not any(present)
    ball=torch.stack([r['ball_episode'] for r in rows]) if all(present) else None
    events=trajectory_event_ids(x,episode,ball)
    length=len(rows)-a.horizon+1
    assert length>0
    old=torch.stack([episode[j:j+length]==episode[:length] for j in range(a.horizon)],-1)
    new=old&torch.stack([events[j:j+length]==events[:length] for j in range(a.horizon)],-1)
    assert torch.equal(old[:,:,0],new[:,:,0]),'Current-step supervision changed'
    removed=old&~new
    result=dict(path=str(path),skill=data['metadata']['skill'],horizon=a.horizon,
        boundary_source='recorded_ball_ids_and_speed' if ball is not None else 'legacy_phase_and_speed_fallback',
        additional_events=int((events[1:]!=events[:-1]).sum()-(episode[1:]!=episode[:-1]).sum()),
        old_valid_future_labels=int(old.sum()),removed_future_labels=int(removed.sum()),
        removed_fraction=float(removed.sum()/old.sum()),affected_windows=int(removed.any(-1).sum()),
        current_step_mask_unchanged=True)
    results.append(result);print(json.dumps(result),flush=True)
a.output.write_text(json.dumps(dict(results=results,scope='Legacy phase/speed boundaries are conservative; these counts do not establish the cause of closed-loop failures.'),indent=2))
