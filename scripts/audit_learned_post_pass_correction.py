"""Compare learned and internal-target corrections on fixed historical shooting states.

These counterfactual commands are never executed. Magnitude is a diagnostic,
not a task reward, safety guarantee or performance result.
"""
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
    assert x.shape[1]==2406 and (x[:,-2:].argmax(-1)==1).all() and set(env.tolist())==set(starts)
    ages=torch.empty(len(x),dtype=torch.long);anchors=torch.empty(len(x),3);tasks=torch.empty(len(x),10)
    groups={'handoff':[],'first_close_after_history_flush':[]}
    for i,start in starts.items():
        ids=(env==i).nonzero().flatten();assert int(tick[ids[0]])==start
        first=x[ids[0]];ages[ids]=tick[ids]-start;anchors[ids,:2]=first[933:935];anchors[ids,2]=1
        tasks[ids]=torch.cat((first[933:939],first[2400:2404]));assert first[940]<.5
        groups['handoff'].append(int(ids[0]))
        close=ids[(ages[ids]>=50)&(x[ids,933:935].norm(dim=-1)<=.7)&(x[ids,940]<.5)]
        if len(close):groups['first_close_after_history_flush'].append(int(close[0]))
    digest=sha(a.checkpoint);policy=load_soccer_policy(a.checkpoint,'cuda:0')
    assert policy.post_pass_latent_adapter_version==1 and policy.refinement_scope=='initial_pass'
    result={}
    with torch.inference_mode():
        for name,indices in groups.items():
            assert indices
            values={label:{key:[] for key in ['latent_l2','joint_rms_rad','maximum_joint_change_rad','radius_error']}
                for label in ['learned','internal_target_solver']}
            for start in range(0,len(indices),64):
                ids=torch.tensor(indices[start:start+64]);raw=x[ids].cuda();age=ages[ids].cuda()
                rma=initialize_new_ball_history(raw,age,anchors[ids].cuda())
                full=initialize_new_ball_task_history(rma,age,tasks[ids].cuda())
                base=policy.planner.prior.act(full)
                desired=policy.planner.target_predictor.bounded_targets(full)
                initial=policy.planner.decode(full,base,desired)
                solved,_=policy._refine(full,initial,desired)
                learned=policy.post_pass_adapter(full,base)
                def targets(z):
                    action=policy.actor(full[:,:64],full[:,64:93],full[:,93:930],z)
                    return policy.default+(6*action).clamp(-6,6)*policy.scale
                reference=targets(base)
                for label,z in [('learned',learned),('internal_target_solver',solved)]:
                    delta=targets(z)-reference
                    metrics=dict(latent_l2=(z-base).norm(dim=-1),joint_rms_rad=delta.square().mean(-1).sqrt(),
                        maximum_joint_change_rad=delta.abs().amax(-1),radius_error=(z.norm(dim=-1)-base.norm(dim=-1)).abs())
                    assert all(torch.isfinite(v).all() for v in metrics.values())
                    for key,v in metrics.items():values[label][key].append(v.cpu())
            result[name]=dict(states=len(indices),metrics={label:{key:stats(torch.cat(v)) for key,v in metrics.items()} for label,metrics in values.items()})
    assert sha(a.checkpoint)==digest
    output=dict(checkpoint_sha256=digest,collection_sha256=sha(a.collection/'shoot_states.pt'),groups=result,
        actions_executed=False,expert_loaded=False,weights_modified=False,
        scope='Fixed historical r3 pre-contact states, manually corrected histories. The learned branch is compared with the previous failed solver on identical inputs; neither action magnitude nor historical coverage establishes on-policy task performance.')
    a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output),flush=True)

if __name__=='__main__':main()
