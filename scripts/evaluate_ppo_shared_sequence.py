"""Evaluate a shared-scene PPO candidate on original tasks and continuous switching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_ppo_training_scene import audit
from audit_recovery_training_scene import audit as recovery_audit
from audit_sequence_recovery import validate

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--training-pid',type=int,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=a.training_run.resolve();tag=run.name
status_path=root/'runs'/f'{tag}_shared_evaluation_status.json'
assert not status_path.exists(),'Preserve prior evaluation'

def state(stage,**details):
    status_path.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))

def container(path):
    return '/workspace/fiber-soccer/'+str(path.relative_to(root))

try:
    state('waiting_for_training_and_original_task_evaluation',training_pid=a.training_pid)
    subprocess.run([sys.executable,str(root/'scripts/evaluate_latent_ppo.py'),
        '--training-run',str(run),'--training-pid',str(a.training_pid)],check=True)
    config=json.loads((run/'config.json').read_text())
    assert config['training_scene']=='shoot' and not config['smoke_test']
    checkpoint=run/'final.pt'
    complete=json.loads((run/'complete.json').read_text())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==complete['checkpoint_sha256']
    recovery=config.get('training_task','pass')=='shoot_recover'
    contracts=(recovery_audit if recovery else audit)(run,root/'runs/planner_history_r5_control_seeded_sequence')
    (root/'runs'/f'{tag}_scene_audit.json').write_text(json.dumps(contracts,indent=2))
    results={}
    for label,policy in [('baseline',config['checkpoint']),('candidate',container(checkpoint))]:
        destination=root/'runs'/f'{tag}_sequence_{label}'
        state('evaluating_sequence_'+label)
        cmd=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
            '--headless','--device','cuda:0','--checkpoint',policy,'--output',container(destination),
            '--num-envs','16','--steps','1000','--seed','813201','--handoff-seed','813201',
            '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot',
            '--events-contract','shoot','--incoming-pass-command','--continuous-after-kick']
        if recovery:
            cmd+=['--transition-steps','0','--post-shoot-recovery-steps',str(config['recovery_delay_steps'])]
        with destination.with_suffix('.log').open('x') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        summary=json.loads((destination/'summary.json').read_text())
        assert summary['one_policy_instance'] and not summary['expert_loaded']
        assert summary['requested']==16 and summary['steps']==1000
        results[label]=dict(summary=summary,accounting=validate(summary))
    assert results['candidate']['summary']['checkpoint_sha256']==complete['checkpoint_sha256']
    assert results['baseline']['summary']['checkpoint_sha256']==config['parent_sha256']
    assert results['baseline']['summary']['initial_state_sha256']==results['candidate']['summary']['initial_state_sha256']
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==complete['checkpoint_sha256']
    state('completed',results=results,training_scene_audit=contracts,
        scope='Repeated development seeds, no expert assistance. Initial robot/ball/observation tensors match; subsequent physical states differ. Full direction/foot/speed/survival counts remain the task criterion.')
except Exception as error:
    state('failed',error=repr(error));raise
