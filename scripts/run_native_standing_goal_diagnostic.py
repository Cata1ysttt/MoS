"""Separate reset-pose tracking from stationary backward-goal feedback."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container

TAG='planner_native_standing_goal_r1'

def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    assert read(runs/'planner_native_standing_r1_status.json')['stage']=='completed'
    paths=['scripts/probe_native_standing.py','src/fiber_soccer/bridge.py','src/fiber_soccer/native_bfm.py',
        'src/fiber_soccer/native_backward.py','src/fiber_soccer/captured_latent_solver.py',
        'src/fiber_soccer/shared_scene.py','AH-BFM/load_ah_bfm.py','AH-BFM/lsvla/fb_cpr_policy.py']
    sources={p:sha(ROOT/p) for p in paths};results={}
    def state(stage,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,
            runtime_source_sha256=sources,**details),indent=2)+'\n');tmp.replace(status)
    try:
        for label,mode,n,steps,seed in [('smoke','backward_stationary',2,4,816402),
            ('direct','direct_reset_pose',16,1000,816401),('latent','latent_reset_pose',16,1000,816401),
            ('backward','backward_stationary',16,1000,816401)]:
            state('queued_or_running',model=label)
            output=runs/f'{TAG}_{label}'
            cmd=[str(ROOT/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/probe_native_standing.py',
                '--headless','--device','cuda:0','--output',container(output),'--controller',mode,
                '--num-envs',str(n),'--steps',str(steps),'--seed',str(seed)]
            with output.with_suffix('.log').open('x') as log:
                subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),stdout=log,stderr=subprocess.STDOUT,check=True)
            record=read(output/'summary.json')
            assert record['requested']==n and record['steps']==steps and record['controller']==mode
            assert not record['expert_loaded'] and not record['training'] and record['all_failure_terms_retained']
            assert record['survivors']+len(record['failures'])==n
            assert sources=={p:sha(ROOT/p) for p in paths},'Diagnostic runtime changed'
            results[label]=record
        for label in ['latent','backward']:
            assert results['direct']['initial_state_sha256']==results[label]['initial_state_sha256']
            assert read(runs/f'{TAG}_direct/scene_contract.json')==read(runs/f'{TAG}_{label}/scene_contract.json')
        state('completed',paired_initial_state_and_scene_equal=True,
            scope='Basic standing capability only. Not post-shot recovery or unified-policy success.')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
