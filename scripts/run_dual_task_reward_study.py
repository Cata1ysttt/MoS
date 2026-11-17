"""Isolated GPU7 study; preserves older fixed-protocol jobs and their sources."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'
TAG='planner_dual_task_reward_r3'
status=RUNS/(TAG+'_status.json')
lock=(RUNS/'isaac_sim_gpu7.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
results={}
def state(stage,**kwargs):
    temp=status.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**kwargs),indent=2)+'\n');temp.replace(status)
def c(path):
    if path.is_relative_to(ROOT):return '/workspace/fiber-soccer/'+str(path.relative_to(ROOT))
    return '/workspace/g1-rollingball/'+str(path.relative_to('/home/jiongye/g1-rollingball'))
def run(script,args,name):
    command=['docker','exec','-w','/workspace/g1-rollingball','-e','CUDA_VISIBLE_DEVICES=7',
             '-e','PYTHONPATH=/workspace/fiber-soccer/src:/workspace/g1-rollingball/source/whole_body_tracking:/workspace/g1-rollingball/source/isaaclab:/workspace/g1-rollingball/source/isaaclab_tasks:/workspace/g1-rollingball/source/isaaclab_assets:/workspace/g1-rollingball/source/isaaclab_rl',
             'fiber-soccer-runtime','/isaac-sim/python.sh',c(ROOT/'scripts'/script),*args,'--headless','--device','cuda:0']
    state(name,command=command)
    with (RUNS/(name+'.log')).open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in protocol['source_hashes'].items())
assert not status.exists()
base=RUNS/'planner_new_ball_task_history_r1/diagnostic.pt'
zero=RUNS/'planner_dual_task_zero_r1/diagnostic.pt'
config=Path('/home/jiongye/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl')
protocol=dict(training_seed=819101,validation_seeds=[819201,819202],num_envs=128,iterations=200,
              training_steps=819200,num_eval_envs=32,eval_steps=1000,checkpoint_selection='final_iteration',
              exploration_std=.05,anchor_weight=.000625,expert_weight=.1,automatic_promotion=False,
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/train_dual_task_latent_ppo.py',ROOT/'scripts/evaluate_dual_task_sequence.py',ROOT/'src/fiber_soccer/dual_task_latent.py',ROOT/'src/fiber_soccer/pass_expert_compat.py']},
              baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),pass_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
(RUNS/(TAG+'_protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n');results['protocol']=protocol
try:
    common=['--checkpoint',c(zero),'--pass-config',c(config),'--exploration-std','.05','--anchor-weight','.000625']
    smoke=RUNS/(TAG+'_smoke')
    run('train_dual_task_latent_ppo.py',common+['--output',c(smoke),'--num-envs','32','--iterations','16','--seed','819100','--smoke-test'],TAG+'_smoke')
    results['smoke']=json.loads((smoke/'complete.json').read_text())
    trained=RUNS/TAG
    run('train_dual_task_latent_ppo.py',common+['--output',c(trained),'--num-envs','128','--iterations','200','--seed','819101'],TAG+'_train')
    results['training']=json.loads((trained/'complete.json').read_text())
    results['evaluations']={}
    for seed in protocol['validation_seeds']:
        for name,checkpoint in [('baseline',base),('trained',trained/'final.pt')]:
            dest=RUNS/f'{TAG}_{name}_{seed}'
            run('evaluate_dual_task_sequence.py',['--checkpoint',c(checkpoint),'--output',c(dest),'--pass-config',c(config),
                '--num-envs','32','--steps','1000','--seed',str(seed),'--handoff-seed',str(seed),
                '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot','--events-contract','shoot',
                '--incoming-pass-command','--continuous-after-kick','--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero'],dest.name)
            results['evaluations'][dest.name]=json.loads((dest/'summary.json').read_text())
        a=results['evaluations'][f'{TAG}_baseline_{seed}'];b=results['evaluations'][f'{TAG}_trained_{seed}']
        assert a['initial_state_sha256']==b['initial_state_sha256']
    (RUNS/(TAG+'_results.json')).write_text(json.dumps(results,indent=2)+'\n')
    state('completed')
except Exception as error:
    state('failed',error=repr(error));raise
