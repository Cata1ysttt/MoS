"""Compare first shooting commands to recorded previous joint targets offline."""
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
    source=a.collection/'shoot_states.pt';data=torch.load(source,map_location='cpu',weights_only=True)
    summary=json.loads((a.collection/'summary.json').read_text())
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    x=data['x'];env=data['environment'].long();tick=data['tick'].long()
    assert x.shape[1]==2406 and set(env.tolist())==set(starts) and (x[:,-2:].argmax(-1)==1).all()
    ages=torch.empty(len(x),dtype=torch.long);anchors=torch.empty(len(x),3);tasks=torch.empty(len(x),10)
    for i,start in starts.items():
        ids=(env==i).nonzero().flatten();assert int(tick[ids[0]])==start
        first=x[ids[0]];ages[ids]=tick[ids]-start
        anchors[ids,:2]=first[933:935];anchors[ids,2]=1
        tasks[ids]=torch.cat((first[933:939],first[2400:2404]))
    digest=sha(a.checkpoint);policy=load_soccer_policy(a.checkpoint,'cuda:0')
    assert policy.refinement_scope=='initial_pass'
    groups={'handoff':ages==0,'precontact_age_1_to_9':(ages>=1)&(ages<10)&(x[:,940]<.5),
        'precontact_age_10_to_49':(ages>=10)&(ages<50)&(x[:,940]<.5),'precontact_age_50_plus':(ages>=50)&(x[:,940]<.5)}
    metrics={mode:{k:[] for k in ['joint_rms_rad','maximum_joint_change_rad']} for mode in ['recorded_executed_latent','raw_history_prior','full_history_prior','live_backward_goal']}
    with torch.inference_mode():
        for start in range(0,len(x),256):
            ids=slice(start,start+256);raw=x[ids].cuda();age=ages[ids].cuda()
            full=initialize_new_ball_history(raw,age,anchors[ids].cuda())
            full=initialize_new_ball_task_history(full,age,tasks[ids].cuda())
            # Bridge.observe returns the previous *scaled* AH action at 64:93.
            previous=raw[:,64:93]
            assert (previous.abs()<=6.00001).all()
            for mode,z in [('recorded_executed_latent',data['z'][ids].cuda()),('raw_history_prior',policy.planner.prior.act(raw)),('full_history_prior',policy.planner.prior.act(full)),('live_backward_goal',raw[:,944:1200])]:
                action=policy.actor(raw[:,:64],raw[:,64:93],raw[:,93:930],z)
                delta=((6*action).clamp(-6,6)-previous)*policy.scale
                for k,v in dict(joint_rms_rad=delta.square().mean(-1).sqrt(),maximum_joint_change_rad=delta.abs().amax(-1)).items():
                    assert torch.isfinite(v).all();metrics[mode][k].append(v.cpu())
    assert sha(a.checkpoint)==digest
    values={mode:{k:torch.cat(v) for k,v in group.items()} for mode,group in metrics.items()}
    result=dict(checkpoint_sha256=digest,collection_sha256=sha(source),collection_checkpoint_sha256=summary['checkpoint_sha256'],new_actions_executed=False,expert_loaded=False,
        scope='Recorded trajectories from the identified collection checkpoint. Changes compare AH command targets with recorded previous AH command targets, not realized joint displacement. The recorded-executed-latent branch reconstructs the collected controller command; other branches are counterfactual. Live backward goal is a diagnostic current-motion encoding, not the reset-state standing goal. No causal stability or task-performance conclusion.',
        groups={name:dict(states=int(mask.sum()),metrics={mode:{k:stats(v[mask]) for k,v in group.items()} for mode,group in values.items()}) for name,mask in groups.items()})
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
