"""Verify native preservation, then test initial backward-goal recovery after shooting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes

TAG='planner_initial_goal_recovery_study_r1'
SEEDS=[816201,816202]  # Development comparison; confirmation requires new seeds.


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    previous=read(runs/'planner_initial_pass_refinement_r1_status.json');assert previous['stage']=='completed'
    assert read(runs/'planner_initial_pass_confirmation_r1_status.json')['stage']=='completed'
    standing=read(runs/'planner_native_standing_goal_r1_status.json');assert standing['stage']=='completed'
    assert standing['results']['backward']['survivors']==standing['results']['backward']['requested']==16
    models=dict(control=runs/'planner_predicted_target_initial_pass/diagnostic.pt',
        candidate=runs/'planner_initial_goal_recovery_r1/diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()}
    prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']==checksums['control'] and prepared['checkpoint_sha256']==checksums['candidate']
    assert prepared['standing_recovery_mode']=='initial_backward_goal'
    assert len(prepared['checks'])==7 and all(prepared['checks'].values())
    assert prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source']
    paths=['scripts/probe_soccer_sequence.py','src/fiber_soccer/planner_policy.py','src/fiber_soccer/predicted_target_solver.py',
        'src/fiber_soccer/initial_goal_recovery.py','src/fiber_soccer/refinement_scope.py','src/fiber_soccer/skill_transition.py',
        'src/fiber_soccer/captured_latent_solver.py','src/fiber_soccer/native_bfm.py','src/fiber_soccer/target_guided.py',
        'src/fiber_soccer/flow_planner.py','src/fiber_soccer/evaluation.py','src/fiber_soccer/planner.py',
        'AH-BFM/load_ah_bfm.py','AH-BFM/lsvla/fb_cpr_policy.py']
    sources={p:sha(ROOT/p) for p in paths};actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    assert actor_sha==standing['results']['backward']['actor_sha256']
    protocol=dict(seeds=SEEDS,seed_scope='Reused development seeds; not fresh confirmation',num_envs=32,steps=1000,
        models={k:str(v) for k,v in models.items()},checkpoint_sha256=checksums,runtime_source_sha256=sources,
        actor_sha256=actor_sha,training=False,failed_lane_control='zero',
        difference='Only after contact first observed under shooting and subsequent pass command, use the reset-state backward goal in the same AH-BFM',
        goal_source='First observation backward feature after each physics reset; caller starts policy at reset',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed error <=30%, no failure through 20 seconds',
        checkpoint_selection='Metadata-only fixed candidate; no automatic promotion')
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');tmp.replace(status)
    try:
        state('verifying_native_tasks')
        subprocess.run([sys.executable,str(ROOT/'scripts/run_predicted_target_solver.py'),'--tag','planner_initial_goal_recovery_r1',
            '--wait-for','planner_initial_pass_confirmation_r1_status.json'],cwd=ROOT,check=True)
        native=read(runs/'planner_initial_goal_recovery_r1_status.json');assert native['stage']=='completed'
        for skill in ['pass','shoot']:
            assert native['results'][skill]['behavior_preservation']['episodes_and_first_impacts_exact']
            assert native['results'][skill]['candidate']['standing_recovery_active_control_steps']==0
        results['native']=native['results']
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,checkpoint in models.items():
                state('evaluating_sequence',seed=seed,model=label)
                destination=runs/f'{TAG}_{seed}_{label}'
                command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(checkpoint),'--output',container(destination),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero']
                with destination.with_suffix('.log').open('x') as log:
                    subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
                summary=read(destination/'summary.json')
                assert summary['checkpoint_sha256']==checksums[label]==sha(checkpoint)
                assert summary['one_policy_instance'] and not summary['expert_loaded'] and summary['failed_lane_control']=='zero'
                assert summary['predicted_target_refinement_scope']=='initial_pass'
                task_protocol=read(destination/'protocol.json')
                if label=='control':
                    old=previous['results']['sequences'][str(seed)]['models']['candidate']['summary']
                    assert {k:v for k,v in summary.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'}
                    assert 'standing_recovery_mode' not in summary and 'standing_recovery_mode' not in task_protocol
                else:
                    assert summary['standing_recovery_mode']=='initial_backward_goal'
                    assert summary['standing_recovery_active_control_steps']>0
                    assert task_protocol.pop('standing_recovery_mode')=='initial_backward_goal'
                    assert (destination/'initial_backward_goal.pt').exists()
                pair[label]=dict(summary=summary,outcomes=outcomes(summary))
                protocols[label]=task_protocol;contracts[label]=read(destination/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert sources=={p:sha(ROOT/p) for p in paths}
            a=pair['control']['summary'];b=pair['candidate']['summary']
            assert a['initial_state_sha256']==b['initial_state_sha256'] and protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=min(v['tick'] for v in a['recovery_switches']+b['recovery_switches'])
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in a[key].items() if v['tick']<=first}=={k:v for k,v in b[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in a[key] if v['tick']<=first]==[v for v in b[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                protocol_equal_except_recovery_mode=True,global_prefix_through_first_recovery_exact=True))
        totals={label:{} for label in models}
        for entry in results['sequences'].values():
            for label,record in entry['models'].items():
                for key,value in record['outcomes'].items():
                    if isinstance(value,int):totals[label][key]=totals[label].get(key,0)+value
                    elif isinstance(value,dict):
                        group=totals[label].setdefault(key,{})
                        for k,v in value.items():group[k]=group.get(k,0)+v
        results['totals']=totals
        assert checksums=={k:sha(v) for k,v in models.items()} and sha(ROOT/'AH-BFM/AH-BFM.pt')==actor_sha
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n');state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
