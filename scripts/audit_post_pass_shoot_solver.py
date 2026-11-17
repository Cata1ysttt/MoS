"""Measure internal-target fitting on fixed historical pre-contact shooting states."""
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
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    data=torch.load(a.collection/'shoot_states.pt',map_location='cpu',weights_only=True)
    sm=json.loads((a.collection/'summary.json').read_text())
    starts={i:v['tick'] for v in sm['switches'] for i in v['environments']}
    x=data['x'];env=data['environment'].long();tick=data['tick'].long()
    assert x.shape[1]==2406 and (x[:,-2:].argmax(-1)==1).all()
    ages=torch.empty(len(x),dtype=torch.long);anchors=torch.empty(len(x),3);tasks=torch.empty(len(x),10)
    groups={'handoff':[],'first_close_after_history_flush':[]}
    for i,start in starts.items():
        ids=(env==i).nonzero().flatten();assert len(ids) and int(tick[ids[0]])==start
        first=x[ids[0]];ages[ids]=tick[ids]-start
        anchors[ids,:2]=first[933:935];anchors[ids,2]=1
        tasks[ids]=torch.cat((first[933:939],first[2400:2404]))
        assert first[940]<.5
        groups['handoff'].append(int(ids[0]))
        close=ids[(ages[ids]>=50)&(x[ids,933:935].norm(dim=-1)<=.7)&(x[ids,940]<.5)]
        if len(close):groups['first_close_after_history_flush'].append(int(close[0]))
    digest=sha(a.checkpoint);header=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert header['predicted_target_refinement_scope']=='initial_pass_and_post_pass_shoot'
    # Stateless full solver on explicitly initialized inputs implements the same
    # numerical proposal as the candidate's verified post-pass gate.
    for key in ['standing_recovery_mode','new_ball_history_mode','predicted_target_refinement_scope']:header.pop(key)
    policy=load_soccer_policy(header,'cuda:0');result={}
    weights=torch.ones(29,device='cuda:0');weights[:12]=2;weights/=weights.mean()
    with torch.inference_mode():
        for name,indices in groups.items():
            values={key:[] for key in ['prior_target_rms_rad','decoded_target_rms_rad','solved_target_rms_rad',
                'prior_weighted_mse','decoded_weighted_mse','solved_weighted_mse',
                'latent_change_from_prior','joint_change_from_prior_rms_rad','maximum_joint_change_rad']}
            for offset in range(0,len(indices),64):
                ids=torch.tensor(indices[offset:offset+64]);raw=x[ids].cuda();age=ages[ids].cuda()
                rma=initialize_new_ball_history(raw,age,anchors[ids].cuda())
                full=initialize_new_ball_task_history(rma,age,tasks[ids].cuda())
                prior=policy.planner.prior.act(full);target=policy.planner.target_predictor.bounded_targets(full)
                decoded=policy.planner.decode(full,prior,target)
                def physical(z):
                    action=policy.actor(full[:,:64],full[:,64:93],full[:,93:930],z)
                    return policy.default+(6*action).clamp(-6,6)*policy.scale
                prior_targets=physical(prior);decoded_targets=physical(decoded);solved=policy(full)
                for label,predicted in [('prior',prior_targets),('decoded',decoded_targets),('solved',solved['joint_targets'])]:
                    squared=(predicted-target).square()
                    values[label+'_target_rms_rad'].append(squared.mean(-1).sqrt().cpu())
                    values[label+'_weighted_mse'].append((squared*weights).mean(-1).cpu())
                delta=solved['joint_targets']-prior_targets
                values['latent_change_from_prior'].append((solved['latent']-prior).norm(dim=-1).cpu())
                values['joint_change_from_prior_rms_rad'].append(delta.square().mean(-1).sqrt().cpu())
                values['maximum_joint_change_rad'].append(delta.abs().amax(-1).cpu())
            values={k:torch.cat(v) for k,v in values.items()}
            result[name]=dict(states=len(indices),metrics={k:stats(v) for k,v in values.items()},
                lower_objective_than_prior=int((values['solved_weighted_mse']<values['prior_weighted_mse']).sum()),
                lower_objective_than_decoder=int((values['solved_weighted_mse']<values['decoded_weighted_mse']).sum()),
                selected_states=[dict(environment=int(env[i]),tick=int(tick[i]),age=int(ages[i])) for i in indices])
    assert sha(a.checkpoint)==digest
    output=dict(collection_sha256=sha(a.collection/'shoot_states.pt'),checkpoint_sha256=digest,
        expert_loaded=False,weights_modified=False,actions_executed=False,groups=result,
        scope='Counterfactual fitting to the internal predictor, not expert truth or task reward. Historical r3 trajectories; no claim about current on-policy distribution or closed-loop success.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in group.items() if n!='selected_states'} for k,group in result.items()}),flush=True)

if __name__=='__main__':main()
