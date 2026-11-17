"""Collect current full-history handoffs after the active fixed study finishes."""
import json
import os
import subprocess
import time
from pathlib import Path
from run_target_guided_planner import ROOT,read,sha,container
from evaluate_predicted_target_sequence import used_seeds
from summarize_continuous_shoot_ppo import outcomes

TAG='planner_full_history_handoff_collection_r1'
SEED=817201


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    parent=runs/'planner_post_pass_valid_settle_r1_status.json'
    initial=read(parent);parent_pid=initial['pid']
    checkpoint=runs/'planner_new_ball_task_history_r1/diagnostic.pt'
    assert sha(checkpoint)=='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'
    checked=set()
    for pattern in ['*/config.json','*/summary.json','*_status.json','*_protocol.json']:checked.update(runs.glob(pattern))
    for path in checked:assert SEED not in used_seeds(read(path)),f'Seed already used: {path}'
    sources=initial['results']['protocol']['runtime_source_sha256'].copy()
    for path in ['scripts/audit_new_ball_history_sensitivity.py','scripts/audit_sequence_recovery.py','scripts/run_target_guided_planner.py','src/fiber_soccer/sequence_collection.py','scripts/queue_shoot_handoff_collection.py','scripts/audit_shoot_action_jump.py']:
        sources[path]=sha(ROOT/path)
    assert all(sha(ROOT/path)==value for path,value in sources.items())
    protocol=dict(seed=SEED,handoff_seed=SEED,steps=1000,num_envs=32,checkpoint_sha256=sha(checkpoint),
        actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),runtime_source_sha256=sources,wait_for_pid=parent_pid,
        seed_files_checked=len(checked),purpose='diagnostic',training=False,expert_loaded=False,
        scope='Collect actual full-history shooting commands after passing. No candidate comparison or promotion; all failures retained.')
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def state(stage,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),protocol=protocol,**details),indent=2)+'\n');tmp.replace(status)
    def frozen():
        assert all(sha(ROOT/path)==value for path,value in sources.items())
        assert sha(checkpoint)==protocol['checkpoint_sha256'] and sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
    try:
        while True:
            waiting=read(parent);assert waiting['pid']==parent_pid
            if waiting['stage']=='completed':break
            assert waiting['stage']!='failed','Parent study failed; inspect it before collecting'
            os.kill(parent_pid,0)  # A status file alone does not establish a live job.
            state('waiting_for_study',parent_stage=waiting['stage']);time.sleep(10)
        frozen();state('collecting');destination=runs/TAG
        command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
            '--checkpoint',container(checkpoint),'--output',container(destination),'--num-envs','32','--steps','1000',
            '--seed',str(SEED),'--handoff-seed',str(SEED),'--scene-base','shoot','--action-contract','shoot',
            '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
            '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero',
            '--record-shoot-states','--collection-purpose','diagnostic']
        with (runs/f'{TAG}.log').open('x') as log:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),stdout=log,stderr=subprocess.STDOUT,check=True)
        frozen();summary=read(destination/'summary.json');metrics=outcomes(summary)
        assert summary['checkpoint_sha256']==protocol['checkpoint_sha256'] and summary['requested']==32 and summary['steps']==1000
        assert summary['one_policy_instance'] and not summary['expert_loaded']
        state('auditing_commands',outcomes=metrics)
        with (runs/f'{TAG}_jump.log').open('x') as log:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/audit_shoot_action_jump.py',
                '--collection',container(destination),'--checkpoint',container(checkpoint),
                '--output',container(runs/f'{TAG}_jump.json')],cwd=ROOT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='7'),stdout=log,stderr=subprocess.STDOUT,check=True)
        frozen();state('completed',outcomes=metrics,collection_sha256=sha(destination/'shoot_states.pt'),
            action_jump=read(runs/f'{TAG}_jump.json'))
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
