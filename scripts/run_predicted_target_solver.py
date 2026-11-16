"""Screen both native tasks with internal predicted-target latent refinement."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container


def main():
    runs=ROOT/'runs';tag='planner_predicted_target_solver_r1';run=runs/tag
    checkpoint=run/'diagnostic.pt';complete=read(run/'complete.json')
    assert complete['steps']==64 and complete['frozen_parameters_unchanged'] and complete['loading_rng_matches_source']
    assert complete['expert_targets_used'] is False and complete['checkpoint_sha256']==sha(checkpoint)
    status=runs/f'{tag}_status.json';assert not status.exists();results={}
    def state(stage,**details):
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        # Keep the already-running r2 comparison together; do not overtake its second task.
        dependency=runs/'planner_target_guided_r2_status.json'
        while read(dependency)['stage'] not in ['completed','failed']:
            pid=read(dependency)['pid']
            os.kill(pid,0)
            state('waiting_for_r2_native_screen',dependency_pid=pid)
            time.sleep(10)
        assert read(dependency)['stage']=='completed'
        for skill in ['pass','shoot']:
            state('evaluating_native',skill=skill)
            destination=runs/f'{tag}_{skill}_eval'
            entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
            command=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
                '--seed','811314','--num-envs','16','--steps','2000',
                '--projector-checkpoint',container(checkpoint),'--output',container(destination)]
            if skill=='pass':command+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
            with destination.with_suffix('.log').open('x') as log:
                subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
            summary=read(destination/'summary.json');baseline=read(runs/f'planner_history_r3_contact_{skill}_eval/summary.json')
            assert summary['completed']==summary['requested']==16 and not summary['expert_assisted_oracle']
            assert summary['initial_state_sha256']==baseline['initial_state_sha256']
            assert summary['controller']=='predicted_target_latent_solver' and summary['backbone']=='ah_bfm'
            assert summary['predicted_target_solver_steps']==64 and summary['predicted_target_solver_expert_targets_used'] is False
            assert sha(checkpoint)==complete['checkpoint_sha256']
            results[skill]=dict(candidate=summary,baseline=baseline,initial_tensors_equal=True)
        state('completed',scope='Repeated development seed, internal target closed-loop diagnostic, no real-time or generalization claim')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
