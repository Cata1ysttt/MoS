"""Verify passive recording, then collect fresh post-pass shooting training states."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--tag',default='planner_sequence_shoot_r1')
p.add_argument('--seed',type=int,default=815201)
p.add_argument('--num-envs',type=int,default=256)
p.add_argument('--upstream-status',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
assert a.tag.replace('_','').isalnum() and a.num_envs>=4
assert a.seed not in [811314,812801,812904,812905,813201,815101,815102]
status=root/'runs'/f'{a.tag}_status.json'
assert not status.exists()
checkpoint=root/'runs/planner_history_r3_contact_train/best.pt'
expert=root/'body_yaw_49999.onnx'
digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
results={}
def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),
        seed=a.seed,num_envs=a.num_envs,checkpoint_sha256=digest,results=results,**details),indent=2))
def container(path):
    return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def execute(command,log,gpu):
    with log.open('x') as f:
        subprocess.run([str(root/'scripts/container_exec.sh')]+command,stdout=f,stderr=subprocess.STDOUT,
            env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
try:
    while True:
        upstream=json.loads(a.upstream_status.read_text())
        if upstream['stage']=='completed':break
        if upstream['stage']=='failed':raise RuntimeError(upstream)
        process=Path(f"/proc/{upstream['pid']}/cmdline")
        if not process.exists() or b'evaluate_recovery_origin.py' not in process.read_bytes():
            raise RuntimeError('Upstream evaluation is not live and has no completed result')
        state('waiting_for_paired_evaluation',upstream_pid=upstream['pid'])
        time.sleep(10)
    for label,seed,n,purpose in [('passive_smoke',815101,32,'diagnostic'),('training',a.seed,a.num_envs,'training')]:
        destination=root/'runs'/f'{a.tag}_{label}'
        state('collecting_'+label)
        command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
            '--checkpoint',container(checkpoint),'--output',container(destination),
            '--num-envs',str(n),'--steps','1000','--seed',str(seed),'--handoff-seed',str(seed),
            '--scene-base','shoot','--action-contract','shoot','--reset-contract','shoot','--events-contract','shoot',
            '--incoming-pass-command','--continuous-after-kick','--transition-steps','0',
            '--post-shoot-recovery-steps','10','--post-kick-adapter-delay','10','--recovery-adapter-shoot-only',
            '--record-shoot-states','--collection-purpose',purpose]
        execute(command,destination.with_suffix('.log'),3)
        summary=json.loads((destination/'summary.json').read_text())
        collection=json.loads((destination/'shoot_collection.json').read_text())
        assert summary['checkpoint_sha256']==digest and not summary['expert_loaded']
        assert collection['purpose']==purpose and collection['states']>0
        assert not collection['expert_loaded'] and not collection['expert_actions_executed']
        if label=='passive_smoke':
            reference=root/'runs/planner_shoot_recovery_ppo_pilot_shoot_origin_815101_baseline'
            for name in ['summary.json','protocol.json','scene_contracts.json']:
                assert json.loads((destination/name).read_text())==json.loads((reference/name).read_text()),name
            results['passive_recording']=dict(entire_summary_equal=True,protocol_equal=True,scene_equal=True)
        state('labeling_'+label)
        execute(['/workspace/fiber-soccer/scripts/label_sequence_shooting_states.py',
            '--states',container(destination/'shoot_states.pt'),'--expert',container(expert),
            '--output',container(destination/'shoot_labels.pt')],
            root/'runs'/f'{a.tag}_{label}_label.log',2)
        results[label]=json.loads((destination/'shoot_labels.json').read_text())
        assert results[label]['expert_sha256']==hashlib.sha256(expert.read_bytes()).hexdigest()
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    state('completed',scope='Fresh autonomous continuous shooting states with offline teacher supervision; diagnostic replay excluded from training. No new planner trained or promoted by this driver.')
except Exception as error:
    state('failed',error=repr(error));raise
