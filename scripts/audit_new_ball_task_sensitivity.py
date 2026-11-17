"""Compare RMA-only and full ball/task initialization on identical recorded states."""
import argparse
import json
from pathlib import Path
import torch
from audit_new_ball_history_sensitivity import sha,stats
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_rma import initialize_new_ball_history
from fiber_soccer.new_ball_task_history import initialize_new_ball_task_history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    data=torch.load(a.collection/'shoot_states.pt',map_location='cpu',weights_only=True)
    summary=json.loads((a.collection/'summary.json').read_text())
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    x=data['x'];env=data['environment'].long();tick=data['tick'].long()
    assert x.shape[1]==2406 and (x[:,-2:].argmax(-1)==1).all()
    ages=torch.empty(len(x),dtype=torch.long);anchors=torch.empty(len(x),3);tasks=torch.empty(len(x),10)
    assert set(env.tolist())==set(starts)
    for i,start in starts.items():
        ids=(env==i).nonzero().flatten();assert len(ids) and int(tick[ids[0]])==start
        first=x[ids[0]];ages[ids]=tick[ids]-start
        anchors[ids,:2]=first[933:935];anchors[ids,2]=1
        tasks[ids]=torch.cat((first[933:939],first[2400:2404]))
    assert (ages>=0).all()
    digest=sha(a.checkpoint);policy=load_soccer_policy(a.checkpoint,'cuda:0')
    metrics={k:[] for k in ['latent_l2','joint_target_rms_rad','maximum_joint_change_rad']}
    with torch.inference_mode():
        for start in range(0,len(x),256):
            stop=min(start+256,len(x));raw=x[start:stop].cuda();age=ages[start:stop].cuda()
            rma=initialize_new_ball_history(raw,age,anchors[start:stop].cuda())
            full=initialize_new_ball_task_history(rma,age,tasks[start:stop].cuda())
            assert torch.equal(rma[:,:2130],full[:,:2130]) and torch.equal(rma[:,2230:],full[:,2230:])
            rz=policy.planner.prior.act(rma);fz=policy.planner.prior.act(full)
            ra=policy.actor(raw[:,:64],raw[:,64:93],raw[:,93:930],rz)
            fa=policy.actor(raw[:,:64],raw[:,64:93],raw[:,93:930],fz)
            delta=((6*fa).clamp(-6,6)-(6*ra).clamp(-6,6))*policy.scale
            values=dict(latent_l2=(fz-rz).norm(dim=-1),joint_target_rms_rad=delta.square().mean(-1).sqrt(),
                maximum_joint_change_rad=delta.abs().amax(-1))
            for k,v in values.items():metrics[k].append(v.cpu())
    metrics={k:torch.cat(v) for k,v in metrics.items()}
    assert all((v[ages>=10]==0).all() for v in metrics.values())
    groups={'handoff':ages==0,'age_1_to_9':(ages>=1)&(ages<10),'age_10_plus':ages>=10}
    assert sha(a.checkpoint)==digest
    result=dict(collection_sha256=sha(a.collection/'shoot_states.pt'),checkpoint_sha256=digest,
        handoffs=len(starts),states=len(x),only_ball_task_history_differs=True,
        all_outputs_exact_after_ten_frames=True,expert_loaded=False,weights_modified=False,
        groups={name:{key:stats(v[mask]) for key,v in metrics.items()} for name,mask in groups.items()},
        scope='Full versus RMA-only input counterfactuals on historical training states. No modified action executed and no closed-loop success claim.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
