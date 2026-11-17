"""Measure projected training noise on fixed recorded states; no actions execute."""
import argparse
import json
from pathlib import Path
import torch
from audit_new_ball_history_sensitivity import sha, stats
from fiber_soccer.planner_policy import load_soccer_policy
from fiber_soccer.new_ball_rma import initialize_new_ball_history
from fiber_soccer.new_ball_task_history import initialize_new_ball_task_history
from fiber_soccer.lpp import project


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['collection','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(4)
    data=torch.load(a.collection/'shoot_states.pt',map_location='cpu',weights_only=True)
    summary=json.loads((a.collection/'summary.json').read_text())
    x=data['x'];env=data['environment'].long();tick=data['tick'].long()
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    assert x.shape[1]==2406 and set(env.tolist())==set(starts)
    groups={'handoff':[],'first_close_after_history_flush':[]};ages=torch.empty(len(x),dtype=torch.long)
    anchors=torch.empty(len(x),3);tasks=torch.empty(len(x),10)
    for i,t in starts.items():
        ids=(env==i).nonzero().flatten();assert int(tick[ids[0]])==t
        first=x[ids[0]];ages[ids]=tick[ids]-t
        anchors[ids,:2]=first[933:935];anchors[ids,2]=1
        tasks[ids]=torch.cat((first[933:939],first[2400:2404]))
        groups['handoff'].append(int(ids[0]))
        close=ids[(ages[ids]>=50)&(x[ids,933:935].norm(dim=-1)<=.7)&(x[ids,940]<.5)]
        if len(close):groups['first_close_after_history_flush'].append(int(close[0]))
    digest=sha(a.checkpoint);policy=load_soccer_policy(a.checkpoint,'cuda:0')
    assert policy.post_pass_latent_adapter_version==1
    generator=torch.Generator(device='cuda').manual_seed(817301);results={}
    with torch.inference_mode():
        for name,ids in groups.items():
            assert ids;ids=torch.tensor(ids);raw=x[ids].cuda()
            full=initialize_new_ball_history(raw,ages[ids].cuda(),anchors[ids].cuda())
            full=initialize_new_ball_task_history(full,ages[ids].cuda(),tasks[ids].cuda())
            base=policy.planner.prior.act(full);mean=policy.post_pass_adapter(full,base)
            def targets(z):
                y=full.repeat(len(z)//len(full),1)
                return (6*policy.actor(y[:,:64],y[:,64:93],y[:,93:930],z)).clamp(-6,6)*policy.scale
            clean=targets(mean);learned=clean-targets(base)
            eps=torch.randn((32,len(ids),256),device='cuda',generator=generator)
            metrics={}
            for sigma in [.2,.05,0.]:
                noisy=project(mean[None]+sigma*eps)
                commands=targets(noisy.flatten(0,1)).reshape(32,len(ids),29)
                delta=commands-clean[None]
                values=dict(latent_l2=(noisy-mean[None]).norm(dim=-1).flatten(),
                    joint_rms_rad=delta.square().mean(-1).sqrt().flatten(),
                    noise_induced_mean_command_bias_rad=delta.mean(0).square().mean(-1).sqrt(),
                    independent_draw_command_difference_rad=(commands[1:]-commands[:-1]).square().mean(-1).sqrt().flatten())
                assert all(torch.isfinite(v).all() for v in values.values())
                metrics[str(sigma)]={k:stats(v.cpu()) for k,v in values.items()}
                if sigma==0:assert delta.abs().max()<1e-5
            results[name]=dict(states=len(ids),draws_per_state=32,
                learned_joint_rms_rad=stats(learned.square().mean(-1).sqrt().cpu()),noise=metrics)
    assert sha(a.checkpoint)==digest
    output=dict(checkpoint_sha256=digest,collection_sha256=sha(a.collection/'shoot_states.pt'),
        collection_checkpoint_sha256=summary['checkpoint_sha256'],seed=817301,groups=results,
        actions_executed=False,expert_loaded=False,
        scope='Fixed-state counterfactual noise sensitivity. Independent noise draws share a physical state; their differences are not actual trajectory action rates. Lower command variance does not establish better task performance. Reducing PPO std also changes KL and anchor scales unless explicitly controlled.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:{'states':v['states'],'noise_rms_medians':{s:m['joint_rms_rad']['median'] for s,m in v['noise'].items()}} for k,v in results.items()}),flush=True)

if __name__=='__main__':main()
