"""Measure input-only RMA counterfactuals on recorded post-pass states, not rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_rma import initialize_new_ball_history


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(v):
    v=v.float()
    return dict(count=v.numel(),mean=float(v.mean()),median=float(v.median()),p95=float(v.quantile(.95)),maximum=float(v.max())) if v.numel() else dict(count=0)


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
    ages=torch.empty(len(x),dtype=torch.long);anchors=torch.empty(len(x),3)
    for i,start in starts.items():
        ids=(env==i).nonzero().flatten();assert len(ids) and int(tick[ids[0]])==start
        ages[ids]=tick[ids]-start;anchors[ids,:2]=x[ids[0],933:935];anchors[ids,2]=1
    assert set(env.tolist())==set(starts) and (ages>=0).all()
    checkpoint_sha=sha(a.checkpoint)
    policy=load_soccer_policy(a.checkpoint,'cuda:0')
    assert policy.refinement_scope=='initial_pass'
    metrics={k:[] for k in ['latent_l2','latent_cosine','joint_target_rms_rad','leg_target_rms_rad','maximum_joint_change_rad']}
    with torch.inference_mode():
        for start in range(0,len(x),256):
            stop=min(start+256,len(x));old=x[start:stop].cuda()
            new=initialize_new_ball_history(old,ages[start:stop].cuda(),anchors[start:stop].cuda())
            assert torch.equal(old[:,:2230],new[:,:2230]) and torch.equal(old[:,2380:],new[:,2380:])
            # Both controllers use this shared feedforward prior after first entering shoot.
            z=policy.planner.prior.act(old);nz=policy.planner.prior.act(new)
            raw=policy.actor(old[:,:64],old[:,64:93],old[:,93:930],z)
            nraw=policy.actor(new[:,:64],new[:,64:93],new[:,93:930],nz)
            delta=((6*nraw).clamp(-6,6)-(6*raw).clamp(-6,6))*policy.scale
            values=dict(latent_l2=(nz-z).norm(dim=-1),latent_cosine=torch.nn.functional.cosine_similarity(z,nz),
                joint_target_rms_rad=delta.square().mean(-1).sqrt(),leg_target_rms_rad=delta[:,:12].square().mean(-1).sqrt(),
                maximum_joint_change_rad=delta.abs().amax(-1))
            for k,v in values.items():metrics[k].append(v.cpu())
    metrics={k:torch.cat(v) for k,v in metrics.items()}
    assert (metrics['latent_l2'][ages>=50]==0).all() and (metrics['joint_target_rms_rad'][ages>=50]==0).all()
    groups={'handoff':ages==0,'age_1_to_9':(ages>=1)&(ages<10),'age_10_to_24':(ages>=10)&(ages<25),
        'age_25_to_49':(ages>=25)&(ages<50),'age_50_plus':ages>=50}
    assert sha(a.checkpoint)==checkpoint_sha
    result=dict(collection_sha256=sha(a.collection/'shoot_states.pt'),checkpoint_sha256=checkpoint_sha,
        handoffs=len(starts),states=len(x),non_rma_features_exact=True,expired_history_outputs_exact=True,
        groups={name:{key:stats(v[mask]) for key,v in metrics.items()} for name,mask in groups.items()},
        expert_loaded=False,weights_modified=False,
        scope='Counterfactual input sensitivity on historical training trajectories. Modified actions were not executed; this cannot predict closed-loop direction, contact, survival or success.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
