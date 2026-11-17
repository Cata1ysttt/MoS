"""Audit the ten-frame shooting policy ball/task terms separately from RMA."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

TERMS={'ball_position':(2130,2160,933,936),'ball_relative_velocity':(2160,2190,936,939),
       'target_direction':(2190,2220,2400,2403),'target_speed':(2220,2230,2403,2404)}


def stats(v):
    return dict(count=v.numel(),mean=float(v.mean()),median=float(v.median()),maximum=float(v.max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    path=a.collection/'shoot_states.pt';data=torch.load(path,map_location='cpu',weights_only=True)
    summary=json.loads((a.collection/'summary.json').read_text())
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    first=[];second=[]
    for i,tick in starts.items():
        ids=(data['environment']==i).nonzero().flatten();assert len(ids)>1
        assert int(data['tick'][ids[0]])==tick and int(data['tick'][ids[1]])==tick+1
        first.append(int(ids[0]));second.append(int(ids[1]))
    x=data['x'][first];nx=data['x'][second]
    diagnostic=torch.load(a.native/'action_diagnostics.pt',map_location='cpu',weights_only=True)[0]
    assert diagnostic['tick']==0 and diagnostic['projector_obs'].shape[-1]==1180
    native=diagnostic['projector_obs'];terms={}
    for name,(start,end,curstart,curend) in TERMS.items():
        width=curend-curstart
        history=x[:,start:end].reshape(-1,10,width);following=nx[:,start:end].reshape(-1,10,width)
        shifted=torch.equal(history[:,1:],following[:,:-1]);assert shifted,name
        gap=(history[:,-1]-x[:,curstart:curend]).norm(dim=-1)
        nextgap=(following[:,-1]-nx[:,curstart:curend]).norm(dim=-1)
        assert float(nextgap.max())<1e-5,(name,float(nextgap.max()))
        reset=native[:,start-1200:end-1200].reshape(-1,10,width)
        repeated=torch.equal(reset,reset[:,-1:].expand_as(reset));assert repeated,name
        terms[name]=dict(input_range=[start,end],frames=10,width=width,
            handoff_latest_to_current_gap=stats(gap),handoffs_with_gap_over_0_1=int((gap>.1).sum()),
            next_step_latest_to_current_gap=stats(nextgap),retains_nine_previous_object_frames=shifted,
            native_reset_repeats_initial_measurement=repeated)
    result=dict(collection_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),handoffs=len(first),
        dt=data['metadata']['dt'],terms=terms,expert_loaded=False,weights_modified=False,
        current_rma_candidate_rewrites_these_terms=False,
        scope='Historical training input audit, not proof of causal failure. The running RMA-only trial is unchanged. These ball/task histories are separate from robot proprioceptive and AH histories.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
