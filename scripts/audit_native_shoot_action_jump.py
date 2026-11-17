"""Reconstruct native shooting commands as a reference for handoff diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from audit_new_ball_history_sensitivity import sha,stats
from fiber_soccer.planner_policy import load_soccer_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    torch.set_num_threads(4)
    summary=json.loads((a.run/'summary.json').read_text())
    assert summary['checkpoint']==str(a.checkpoint) and summary['backbone']=='ah_bfm'
    assert not summary['expert_assisted_oracle'] and summary['latent_interval']==1
    windows=torch.load(a.run/'latent_windows.pt',map_location='cpu',weights_only=True)
    diagnostics={r['tick']:r for r in torch.load(a.run/'action_diagnostics.pt',map_location='cpu',weights_only=True)}
    impacts=json.loads((a.run/'first_kick_velocity.json').read_text())['samples']
    assert len(impacts)==summary['requested']
    contact_tick=torch.tensor([impacts[str(i)]['tick'] for i in range(summary['requested'])])
    policy=load_soccer_policy(a.checkpoint,'cuda:0');digest=sha(a.checkpoint)
    values={group:{metric:[] for metric in ['previous_command_rms_rad','current_joint_error_rms_rad','maximum_previous_command_change_rad']}
            for group in ['initial_command','precontact_step_1_to_9','precontact_step_10_to_49','precontact_step_50_plus']}
    verified=0;largest_error=0.
    with torch.inference_mode():
        for tick,w in enumerate(windows):
            assert w['execution']=='hold' and len(w['states'])==len(w['masks'])==1
            state,last,history=[v.cuda() for v in w['states'][0]]
            z=w['endpoint'].cuda();scaled=(6*policy.actor(state,last,history,z)).clamp(-6,6)
            target=policy.default+scaled*policy.scale
            if tick in diagnostics:
                expected=diagnostics[tick]['bfm_targets'].cuda()
                torch.testing.assert_close(target,expected,rtol=1e-5,atol=1e-6)
                largest_error=max(largest_error,float((target-expected).abs().max()));verified+=len(target)
            delta=(scaled-last)*policy.scale
            # AH q input is joint_position minus AH default, without a scale factor.
            error=scaled*policy.scale-state[:,:29]
            active=w['masks'][0]&(tick<contact_tick)
            group='initial_command' if tick==0 else 'precontact_step_1_to_9' if tick<10 else 'precontact_step_10_to_49' if tick<50 else 'precontact_step_50_plus'
            for name,v in dict(previous_command_rms_rad=delta.square().mean(-1).sqrt(),
                    current_joint_error_rms_rad=error.square().mean(-1).sqrt(),
                    maximum_previous_command_change_rad=delta.abs().amax(-1)).items():
                assert torch.isfinite(v).all();values[group][name].append(v.cpu()[active])
    assert sha(a.checkpoint)==digest
    result=dict(checkpoint_sha256=digest,source_sha256={name:sha(a.run/name) for name in ['summary.json','latent_windows.pt','action_diagnostics.pt','first_kick_velocity.json']},
        saved_target_rows_verified=verified,maximum_target_reconstruction_error_rad=largest_error,
        groups={group:{metric:stats(torch.cat(v)) for metric,v in metrics.items()} for group,metrics in values.items()},
        new_actions_executed=False,expert_queried=False,
        scope='Existing 16-environment native shooting evaluation. Commands reconstructed from saved AH inputs and executed latents, checked against stored targets. The initial previous action is reset to zero, unlike a continuous handoff; the two initial command changes are not equivalent physical events. Later groups exclude samples at or after first valid impact. No new performance evaluation.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
