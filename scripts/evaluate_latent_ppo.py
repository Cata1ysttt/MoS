"""Audit a completed PPO export, then evaluate both skills on the development seed."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--evaluation-tag',help='Separate artifact name for an explicit runtime-control ablation')
p.add_argument('--post-kick-adapter-delay',type=int,default=0)
p.add_argument('--training-pid',type=int,help='Host wrapper PID, used only while waiting for completion')
a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=a.training_run.resolve();tag=a.evaluation_tag or run.name
if Path(tag).name!=tag or not tag or a.post_kick_adapter_delay<0:p.error('Invalid evaluation tag/delay')
status_path=root/'runs'/f'{tag}_evaluation_status.json'
assert not status_path.exists(),'Preserve prior evaluation'

def state(stage,**details):
    status_path.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))

def container(path):
    return '/workspace/fiber-soccer/'+str(path.relative_to(root))

try:
    while not (run/'complete.json').exists():
        state('waiting_for_training',training_pid=a.training_pid)
        status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
        if status.get('stage')=='failed':raise RuntimeError(status)
        if not a.training_pid:raise RuntimeError('Training incomplete; supply its live host PID')
        process=Path(f'/proc/{a.training_pid}')
        if not process.exists():raise RuntimeError('Training host PID is gone without completion')
        cmd=(process/'cmdline').read_bytes()
        if b'train_latent_ppo.py' not in cmd or run.name.encode() not in cmd:raise RuntimeError('Training PID does not match this run')
        time.sleep(5)
    complete=json.loads((run/'complete.json').read_text());config=json.loads((run/'config.json').read_text())
    assert not complete['smoke_test'] and not config['smoke_test']
    assert complete['shared_tensors_unchanged'] and complete['shoot_probe_outputs_unchanged'] and complete['deployable_roundtrip']
    assert config['checkpoint_selection']=='final_iteration' and config['seed']!=811314
    checkpoint=run/'final.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==complete['checkpoint_sha256']
    wrapper=str(root/'scripts/container_exec.sh')
    command=[wrapper,'/workspace/fiber-soccer/scripts/audit_skill_adapter.py',
        '--base',config['checkpoint'],'--adapted',container(checkpoint),'--device','cuda:0',
        '--data','/workspace/fiber-soccer/runs/planner_recovery_r2_sum_semantic/pass_trajectory.pt',
        '/workspace/fiber-soccer/runs/planner_recovery_r2_sum_semantic/shoot_trajectory.pt',
        '--output',container(root/'runs'/f'{tag}_preservation.json')]
    state('auditing_protected_skill',checkpoint_sha256=digest)
    with (root/'runs'/f'{tag}_preservation.log').open('x') as f:
        subprocess.run(command,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
    results={}
    for skill in ['shoot','pass']:
        name=f'{tag}_{skill}_eval';entry='probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'
        cmd=[wrapper,'/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0',
            '--evaluate','--seed','811314','--num-envs','16','--steps','2000',
            '--projector-checkpoint',container(checkpoint),'--post-kick-adapter-delay',str(a.post_kick_adapter_delay),'--output',container(root/'runs'/name)]
        if skill=='pass':cmd+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
        state('evaluating_'+skill,checkpoint_sha256=digest)
        with (root/'runs'/f'{name}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((root/'runs'/name/'summary.json').read_text())
        baseline=json.loads((root/'runs'/f'planner_history_r3_contact_{skill}_eval/summary.json').read_text())
        assert summary['completed']==summary['requested']==16
        assert summary['seed']==811314 and not summary['expert_assisted_oracle']
        assert summary['controller']=='unified_latent_planner'
        assert summary['post_kick_adapter_delay']==a.post_kick_adapter_delay
        assert summary['initial_state_sha256']==baseline['initial_state_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
        results[skill]=dict(planner=summary,baseline=baseline)
    state('completed',post_kick_adapter_delay=a.post_kick_adapter_delay,checkpoint_sha256=digest,results=results,scope='Repeated development seed; not independent validation or continuous skill switching')
except Exception as error:
    state('failed',error=repr(error));raise
