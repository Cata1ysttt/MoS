"""Screen delayed post-pass settling without changing handoff, physics or scoring."""
import json
import os
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from audit_initial_goal_execution import audit as recovery_audit
from audit_new_ball_history_execution import audit as history_audit

TAG='planner_post_pass_settle_r1'
SEEDS=[816201,816202]


def settle_audit(directory):
    sm=read(directory/'summary.json');records=read(directory/'predicted_target_solver.json')
    delay=sm['post_pass_settle_delay_steps'];assert delay==10 and len(records)==sm['steps']
    contacts={}
    for event in sm['contact_observations']:
        if event['stage']==0:contacts.setdefault(event['environment'],event['tick'])
    switches={i:e['tick'] for e in sm['switches'] for i in e['environments']}
    failures={e['environment']:e['tick'] for e in sm['first_resets']}
    expected=[0]*sm['steps'];durations={}
    for i,contact in contacts.items():
        start=contact+delay;end=min(sm['steps'],switches.get(i,sm['steps']),failures.get(i,sm['steps']-1)+1)
        durations[str(i)]=max(0,end-start)
        for tick in range(start,end):expected[tick]+=1
    actual=[r['post_pass_settle_live_environments'] for r in records]
    assert actual==expected,'Live settling activation differs from delayed passing-contact intervals'
    assert sum(actual)==sm['post_pass_settle_active_control_steps']
    assert all(r['post_pass_settle_environments']>=r['post_pass_settle_live_environments'] for r in records)
    return dict(active_control_steps=sum(actual),live_delayed_passing_intervals_exact=True,
        active_steps_per_environment=durations,first_activation_tick=next((i for i,v in enumerate(actual) if v),None),
        scope='Counts exclude failed lanes whose computed action is overridden by zero; policy routing is verified separately.')


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    previous=read(runs/'planner_post_pass_latent_ppo_r1_status.json');assert previous['stage']=='completed'
    models=dict(control=runs/'planner_new_ball_task_history_r1/diagnostic.pt',candidate=runs/TAG/'diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()};prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']==checksums['control'] and prepared['checkpoint_sha256']==checksums['candidate']
    assert all(prepared['checks'].values()) and prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source']
    paths=list(previous['results']['protocol']['runtime_source_sha256'])+[
        'src/fiber_soccer/post_pass_settle.py','scripts/prepare_post_pass_settle.py','scripts/run_post_pass_settle_study.py']
    sources={p:sha(ROOT/p) for p in paths};actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    protocol=dict(seeds=SEEDS,seed_scope='Reused development seeds; fresh confirmation required for adoption',num_envs=32,steps=1000,
        checkpoint_sha256=checksums,runtime_source_sha256=sources,actor_sha256=actor_sha,training=False,expert_targets_used=False,
        post_pass_settle_delay_steps=10,handoff_delay_steps=50,post_shoot_recovery_steps=10,
        difference='Candidate preserves ten control steps after observed passing contact, then uses the initial backward goal while still commanded to pass. Handoff, ball sampling, physics and scoring remain fixed; source full-history and post-shoot recovery are retained.',
        native_gate='Shooting episodes and first impacts exact. Passing kick success, direction <=10deg and post-kick survival do not decrease; wrong-foot rate does not increase. Passing post-contact behavior is intentionally changed.',
        development_gate='Full success strictly increases, precise shot and survivors do not decrease',
        failed_lane_control='zero',automatic_promotion=False)
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol,native={})
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    def execute(command,log):
        with log.open('x') as output:
            subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
        assert all(sha(ROOT/path)==digest for path,digest in sources.items())
    try:
        native_gate=True
        for skill in ['pass','shoot']:
            state('evaluating_native',skill=skill);dest=runs/f'{TAG}_{skill}_eval'
            entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
            command=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
                '--seed','811314','--num-envs','16','--steps','2000','--projector-checkpoint',container(models['candidate']),'--output',container(dest)]
            if skill=='pass':command+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
            execute(command,dest.with_suffix('.log'))
            sm=read(dest/'summary.json');reference=runs/f'planner_new_ball_task_history_r1_{skill}_eval';old=read(reference/'summary.json')
            assert sm['completed']==sm['requested']==16 and not sm['expert_assisted_oracle']
            assert sm['initial_state_sha256']==old['initial_state_sha256'] and sm['post_pass_settle_delay_steps']==10
            assert sm['standing_recovery_active_control_steps']==sm['new_ball_history_active_control_steps']==0
            if skill=='shoot':
                assert sm['post_pass_settle_policy_control_steps']==0
                for name in ['episodes.json','first_kick_velocity.json']:assert read(dest/name)==read(reference/name)
                passed=True
            else:
                assert sm['post_pass_settle_policy_control_steps']>0
                positive=['quality_kick_success','quality_ball_direction_success_10deg','quality_post_kick_survival_ratio']
                passed=all(sm[k]>=old[k] for k in positive) and sm['quality_wrong_foot_kick']<=old['quality_wrong_foot_kick']
                native_gate &= passed
            results['native'][skill]=dict(summary=sm,reference=old,quality_screen_passed=passed,
                first_impacts_exact=read(dest/'first_kick_velocity.json')==read(reference/'first_kick_velocity.json'))
        results['native_screen_passed']=native_gate
        if not native_gate:
            state('completed',scope='Rejected by predeclared native quality screen; no continuous rollout or promotion');return
        results['sequences']={}
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,checkpoint in models.items():
                state('evaluating_sequence',seed=seed,model=label);dest=runs/f'{TAG}_{seed}_{label}'
                execute(['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(checkpoint),'--output',container(dest),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero'],dest.with_suffix('.log'))
                sm=read(dest/'summary.json');assert sm['checkpoint_sha256']==checksums[label]==sha(checkpoint)
                assert sm['one_policy_instance'] and not sm['expert_loaded'] and sm['failed_lane_control']=='zero'
                assert sm['requested']==32 and sm['steps']==1000
                task=read(dest/'protocol.json');audits=dict(recovery=recovery_audit(dest),history=history_audit(dest))
                if label=='control':
                    old=previous['results']['sequences'][str(seed)]['models']['control']['summary']
                    assert {k:v for k,v in sm.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'}
                else:
                    assert task.pop('post_pass_settle_delay_steps')==10
                    audits['settling']=settle_audit(dest)
                pair[label]=dict(summary=sm,outcomes=outcomes(sm),execution_audits=audits)
                protocols[label]=task;contracts[label]=read(dest/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
            a=pair['control']['summary'];b=pair['candidate']['summary']
            assert a['initial_state_sha256']==b['initial_state_sha256']
            assert protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=pair['candidate']['execution_audits']['settling']['first_activation_tick'];assert first is not None
            # The intervention precedes handoff; compare only the true unchanged prefix.
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in a[key].items() if v['tick']<=first}=={k:v for k,v in b[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in a[key] if v['tick']<=first]==[v for v in b[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                task_protocol_equal_except_settling=True,global_prefix_through_first_settling_action_exact=True,control_reproduced=True))
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
        assert checksums=={k:sha(v) for k,v in models.items()} and sha(ROOT/'AH-BFM/AH-BFM.pt')==actor_sha
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
