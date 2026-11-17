"""Inspect whether a new shooting ball inherits the previous ball's RMA history."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def stats(v):
    return dict(count=v.numel(),mean=float(v.mean()),median=float(v.median()),minimum=float(v.min()),maximum=float(v.max())) if v.numel() else dict(count=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    path=a.collection/'shoot_states.pt';data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['metadata']['schema']=='soccer_state_v4_task_history_2406'
    summary=json.loads((a.collection/'summary.json').read_text())
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    first=[];second=[];eligible=[]
    for i in sorted(starts):
        ids=(data['environment']==i).nonzero().flatten();assert len(ids)>0
        index=int(ids[0]);assert int(data['tick'][index])==starts[i]
        first.append(index)
        if len(ids)>1 and int(data['tick'][ids[1]])==starts[i]+1:
            second.append(int(ids[1]));eligible.append(index)
    x=data['x'][first];rma=x[:,2230:2380].reshape(-1,50,3)
    next_x=data['x'][second];next_rma=next_x[:,2230:2380].reshape(-1,50,3)
    previous=data['x'][eligible,2230:2380].reshape(-1,50,3)
    oldest_first=torch.equal(previous[:,1:],next_rma[:,:-1])
    newest_first=torch.equal(previous[:,:-1],next_rma[:,1:])
    assert oldest_first and not newest_first,'Unexpected RMA time layout'
    current_gap=(rma[:,-1,:2]-x[:,933:935]).norm(dim=-1)
    next_gap=(next_rma[:,-1,:2]-next_x[:,933:935]).norm(dim=-1)
    assert float(next_gap.max())<1e-5,'Latest post-step RMA observation disagrees with current ball'
    native=torch.load(a.native/'action_diagnostics.pt',map_location='cpu',weights_only=True)[0]
    assert native['tick']==0 and native['projector_obs'].shape[-1]==1180
    native_rma=native['projector_obs'][:,1030:].reshape(-1,50,3)
    native_repeated=torch.equal(native_rma,native_rma[:,-1:].expand_as(native_rma))
    failures={v['environment']:v for v in summary['first_resets']};shots={int(k) for k in summary['shoot_impacts']}
    before=[dict(environment=i,steps=failures[i]['tick']-tick) for i,tick in starts.items() if i in failures and i not in shots]
    result=dict(collection=str(a.collection),collection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native=str(a.native),recorded_handoffs=len(first),handoffs_with_next_step=len(second),
        rma_shape=[50,3],rma_history_order='oldest_first',history_seconds=50*data['metadata']['dt'],
        at_handoff_latest_history_to_current_ball_gap_m=stats(current_gap),
        handoffs_with_gap_over_half_meter=int((current_gap>.5).sum()),
        next_control_step_latest_history_to_current_ball_gap_m=stats(next_gap),
        exact_shift_retains_49_previous_ball_samples=oldest_first,
        native_reset_history_all_50_frames_repeat_initial_measurement=native_repeated,
        failed_before_shot=len(before),failed_before_shot_within_50_steps=sum(v['steps']<50 for v in before),
        pre_shot_failures=before,expert_loaded_for_audit=False,weights_modified=False,
        scope='Historical training collection, descriptive input audit. Does not prove stale history causes failure or that replacing it improves control. Robot/AH histories and test scene remain unchanged.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['pre_shot_failures','scope','collection_sha256']}))

if __name__=='__main__':main()
