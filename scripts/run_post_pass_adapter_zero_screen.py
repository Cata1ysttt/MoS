"""Require a zero adapter to preserve both native tasks and a full continuous rollout."""
import json
import os
import subprocess
import sys
import time
from run_target_guided_planner import ROOT,read,sha,container
from audit_initial_goal_execution import audit as recovery_audit
from audit_new_ball_history_execution import audit as history_audit

TAG='planner_post_pass_latent_adapter_zero_r1'


def adapter_audit(directory):
    summary=read(directory/'summary.json');records=read(directory/'predicted_target_solver.json')
    assert len(records)==summary['steps']
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    failures={v['environment']:v['tick'] for v in summary['first_resets']};contacts={}
    for v in summary['contact_observations']:
        if v['stage']==1:contacts.setdefault(v['environment'],v['tick'])
    expected=[0]*summary['steps']
    for i,start in starts.items():
        end=min(summary['steps'],failures.get(i,summary['steps']-1)+1,contacts.get(i,summary['steps']))
        for tick in range(start,end):expected[tick]+=1
    actual=[v['post_pass_latent_adapter_environments'] for v in records]
    assert actual==expected and sum(actual)==summary['post_pass_latent_adapter_active_control_steps']
    return dict(active_control_steps=sum(actual),live_precontact_intervals_exact=True)


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_screen_status.json';assert not status.exists()
    preceding=read(runs/'planner_post_pass_shoot_solver_study_r1_status.json');assert preceding['stage']=='completed'
    checkpoint=runs/TAG/'diagnostic.pt';prepared=read(checkpoint.parent/'complete.json')
    assert prepared['checkpoint_sha256']==sha(checkpoint) and all(prepared['checks'].values())
    paths=list(preceding['results']['protocol']['runtime_source_sha256'])+[
        'src/fiber_soccer/post_pass_latent_adapter.py','scripts/prepare_post_pass_latent_adapter.py',
        'scripts/run_post_pass_adapter_zero_screen.py']
    sources={p:sha(ROOT/p) for p in paths}
    protocol=dict(checkpoint_sha256=sha(checkpoint),runtime_source_sha256=sources,actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),
        training=False,expert_targets_used=False,automatic_promotion=False,seed=816201,num_envs=32,steps=1000,
        requirement='Zero adapter must preserve native episodes/impacts and all continuous summary fields except checkpoint, latency and explicit adapter metadata')
    (runs/f'{TAG}_screen_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol)
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        state('verifying_native_tasks')
        subprocess.run([sys.executable,str(ROOT/'scripts/run_predicted_target_solver.py'),'--tag',TAG,
            '--wait-for','planner_post_pass_shoot_solver_study_r1_status.json'],cwd=ROOT,check=True)
        native=read(runs/f'{TAG}_status.json');assert native['stage']=='completed'
        for skill,v in native['results'].items():
            assert v['behavior_preservation']['episodes_and_first_impacts_exact']
            assert v['candidate']['post_pass_latent_adapter_version']==1
            assert v['candidate']['post_pass_latent_adapter_active_control_steps']==0
        results['native']=native['results'];state('verifying_continuous_sequence')
        dest=runs/f'{TAG}_sequence'
        command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
            '--checkpoint',container(checkpoint),'--output',container(dest),'--num-envs','32','--steps','1000',
            '--seed','816201','--handoff-seed','816201','--scene-base','shoot','--action-contract','shoot',
            '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
            '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero']
        with dest.with_suffix('.log').open('x') as log:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        sm=read(dest/'summary.json');reference=runs/'planner_post_pass_shoot_solver_study_r1_816201_control'
        old=read(reference/'summary.json');excluded={'checkpoint','checkpoint_sha256','policy_latency_seconds',
            'post_pass_latent_adapter_version','post_pass_latent_adapter_active_control_steps'}
        assert {k:v for k,v in sm.items() if k not in excluded}=={k:v for k,v in old.items() if k not in excluded}
        assert sm['checkpoint_sha256']==sha(checkpoint)==protocol['checkpoint_sha256']
        assert sm['post_pass_latent_adapter_active_control_steps']>0
        task=read(dest/'protocol.json');assert task.pop('post_pass_latent_adapter_version')==1
        assert task==read(reference/'protocol.json')
        assert read(dest/'scene_contracts.json')==read(reference/'scene_contracts.json')
        results['sequence']=dict(summary=sm,reference=str(reference),summary_preserved_except_declared_fields=True,
            scene_and_task_protocol_preserved=True,recovery_audit=recovery_audit(dest),history_audit=history_audit(dest),adapter_audit=adapter_audit(dest))
        assert all(sha(ROOT/p)==h for p,h in sources.items())
        assert sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
