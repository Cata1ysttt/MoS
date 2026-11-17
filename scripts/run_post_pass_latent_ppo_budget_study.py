"""Tenfold training-budget ablation with the same adapter, rewards and optimizer."""
import json
import os
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from audit_initial_goal_execution import audit as recovery_audit
from audit_new_ball_history_execution import audit as history_audit
from run_post_pass_adapter_zero_screen import adapter_audit

TAG='planner_post_pass_latent_ppo_budget_r1'
SEEDS=[816201,816202]


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';run=runs/TAG
    assert not status.exists() and not run.exists()
    parent_path=runs/'planner_full_history_handoff_collection_r1_status.json'
    parent=read(parent_path);parent_pid=parent['pid']
    short=read(runs/'planner_post_pass_latent_ppo_r1_status.json');assert short['stage']=='completed'
    assert not short['results']['development_screen_passed']
    assert all(sha(ROOT/path)==value for path,value in parent['protocol']['runtime_source_sha256'].items())
    assert sha(ROOT/'scripts/train_post_pass_latent_ppo.py')==short['results']['protocol']['runtime_source_sha256']['scripts/train_post_pass_latent_ppo.py']
    zero=read(runs/'planner_post_pass_latent_adapter_zero_r1_screen_status.json');assert zero['stage']=='completed'
    smoke=read(runs/'planner_post_pass_latent_ppo_smoke_r1_queue_status.json');assert smoke['stage']=='completed'
    audit=read(runs/'planner_post_pass_latent_ppo_smoke_r1_audit.json')
    assert audit['checkpoint_sha256']==smoke['result']['checkpoint_sha256']
    assert audit['only_adapter_network_weights_changed'] and audit['command_and_reset_match_sequence']
    source=runs/'planner_post_pass_latent_adapter_zero_r1/diagnostic.pt'
    control=runs/'planner_new_ball_task_history_r1/diagnostic.pt'
    paths=list(parent['protocol']['runtime_source_sha256'])+[
        'scripts/train_post_pass_latent_ppo.py','scripts/audit_post_pass_latent_ppo.py',
        'scripts/run_post_pass_latent_ppo_study.py','scripts/run_post_pass_adapter_zero_screen.py',
        'scripts/run_post_pass_latent_ppo_budget_study.py']
    sources={p:sha(ROOT/p) for p in paths}
    protocol=dict(runtime_source_sha256=sources,source_sha256=sha(source),control_sha256=sha(control),
        actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),training_seed=817101,num_envs_training=256,iterations=1000,
        rollout_steps=32,training_environment_steps=8192000,training_horizon_seconds=20,
        joint_impact_bonus=10,joint_timeout_bonus=20,checkpoint_selection='final_iteration',
        evaluation_seeds=SEEDS,evaluation_seed_scope='Reused development seeds; no generalization claim',
        evaluation_num_envs=32,evaluation_steps=1000,failed_lane_control='zero',expert_targets_used=False,
        automatic_promotion=False,development_screen='Full success strictly increases, precise shot and survivors do not decrease',
        difference='Training budget only: 1000 rather than 100 iterations from the same zero adapter and training seed. Same model, reward, optimizer, exploration, physics, horizon and final-iteration selection. The failed 100-iteration weights are not resumed.',
        previous_training_iterations=100,previous_training_checkpoint_sha256=short['results']['training']['checkpoint_sha256'],
        wait_for_collection_pid=parent_pid)
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol)
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    def execute(command,log,gpu='3'):
        with log.open('x') as output:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES=gpu),check=True)
        assert all(sha(ROOT/path)==digest for path,digest in sources.items())
    try:
        while True:
            parent=read(parent_path);assert parent['pid']==parent_pid
            if parent['stage']=='completed':break
            assert parent['stage']!='failed','Collection failed; inspect it before starting training'
            os.kill(parent_pid,0)
            state('waiting_for_collection',parent_stage=parent['stage']);time.sleep(10)
        assert all(sha(ROOT/path)==digest for path,digest in sources.items())
        state('training')
        execute(['/workspace/fiber-soccer/scripts/train_post_pass_latent_ppo.py','--headless','--device','cuda:0',
            '--checkpoint',container(source),'--output',container(run),'--seed','817101','--num-envs','256',
            '--iterations','1000','--rollout-steps','32'],run.with_suffix('.log'))
        done=read(run/'complete.json');checkpoint=run/'final.pt';checksum=sha(checkpoint)
        assert done['checkpoint_sha256']==checksum and not done['smoke_test']
        assert done['iterations']==1000 and done['environment_steps']==8192000
        assert done['shared_tensors_unchanged'] and done['deployable_roundtrip']
        metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        previous_metrics=[json.loads(line) for line in (runs/'planner_post_pass_latent_ppo_r1/metrics.jsonl').read_text().splitlines()]
        assert len(previous_metrics)==100 and len(metrics)==1000
        results['first_100_iterations_exact_except_elapsed']=all(
            {k:v for k,v in old.items() if k!='elapsed'}=={k:v for k,v in new.items() if k!='elapsed'}
            for old,new in zip(previous_metrics,metrics[:100]))
        results['training']=done;state('auditing_training')
        audit_path=runs/f'{TAG}_audit.json'
        execute(['/workspace/fiber-soccer/scripts/audit_post_pass_latent_ppo.py','--source',container(source),
            '--run',container(run),'--sequence',container(runs/'planner_post_pass_latent_adapter_zero_r1_sequence'),
            '--output',container(audit_path)],runs/f'{TAG}_audit.log','7')
        results['training_audit']=read(audit_path);results['native']={}
        for skill in ['pass','shoot']:
            state('evaluating_native',skill=skill);dest=runs/f'{TAG}_{skill}_eval'
            entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
            command=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
                '--seed','811314','--num-envs','16','--steps','2000','--projector-checkpoint',container(checkpoint),'--output',container(dest)]
            if skill=='pass':command+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
            execute(command,dest.with_suffix('.log'))
            sm=read(dest/'summary.json');reference=runs/f'planner_post_pass_latent_adapter_zero_r1_{skill}_eval'
            assert sm['requested']==sm['completed']==16 and not sm['expert_assisted_oracle']
            assert sm['initial_state_sha256']==read(reference/'summary.json')['initial_state_sha256']
            assert sm['post_pass_latent_adapter_active_control_steps']==0
            for name in ['episodes.json','first_kick_velocity.json']:assert read(dest/name)==read(reference/name)
            results['native'][skill]=dict(summary=sm,episodes_and_first_impacts_exact=True)
        results['sequences']={};models=dict(control=control,candidate=checkpoint)
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,model in models.items():
                state('evaluating_sequence',seed=seed,model=label);dest=runs/f'{TAG}_{seed}_{label}'
                execute(['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(model),'--output',container(dest),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero'],dest.with_suffix('.log'))
                sm=read(dest/'summary.json');assert sm['checkpoint_sha256']==sha(model)
                assert sm['one_policy_instance'] and not sm['expert_loaded'] and sm['failed_lane_control']=='zero'
                assert sm['requested']==32 and sm['steps']==1000
                task=read(dest/'protocol.json');audits=dict(recovery=recovery_audit(dest),history=history_audit(dest))
                if label=='control':
                    old=read(runs/f'planner_post_pass_shoot_solver_study_r1_{seed}_control/summary.json')
                    assert {k:v for k,v in sm.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'}
                else:
                    assert task.pop('post_pass_latent_adapter_version')==sm['post_pass_latent_adapter_version']==1
                    audits['adapter']=adapter_audit(dest)
                pair[label]=dict(summary=sm,outcomes=outcomes(sm),execution_audits=audits)
                protocols[label]=task;contracts[label]=read(dest/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
            a=pair['control']['summary'];b=pair['candidate']['summary']
            assert a['initial_state_sha256']==b['initial_state_sha256']
            assert protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=min(v['tick'] for v in a['switches']+b['switches'])
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in a[key].items() if v['tick']<=first}=={k:v for k,v in b[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in a[key] if v['tick']<=first]==[v for v in b[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                task_protocol_equal_except_adapter=True,global_prefix_through_first_handoff_exact=True,control_reproduced=True))
        totals={label:{} for label in models}
        for entry in results['sequences'].values():
            for label,record in entry['models'].items():
                for key,value in record['outcomes'].items():
                    if isinstance(value,int):totals[label][key]=totals[label].get(key,0)+value
                    elif isinstance(value,dict):
                        group=totals[label].setdefault(key,{})
                        for k,v in value.items():group[k]=group.get(k,0)+v
        results['totals']=totals;a=totals['control'];b=totals['candidate']
        results['development_screen_passed']=b['full_success']>a['full_success'] and b['precise_shot']>=a['precise_shot'] and b['survivors']>=a['survivors']
        assert sha(checkpoint)==checksum and sha(source)==protocol['source_sha256'] and sha(control)==protocol['control_sha256']
        assert sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
