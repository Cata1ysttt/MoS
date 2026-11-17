"""Confirm initial backward-goal recovery against r3 and its exact parent."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from evaluate_predicted_target_sequence import used_seeds

SEEDS=[816501,816502,816503,816504]
TAG='planner_initial_goal_confirmation_r1'
BASE_SHA='5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133'
CONTROL_SHA='9832d86c0d9efa6e028262ff347d7512082c2a2ef976d2294a2be8bdf484f8ba'
CANDIDATE_SHA='7c98136ec5848fbf9ca67f7132f471548d0f4c098266c94df4b67cb993361fdc'


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';protocol_path=runs/f'{TAG}_protocol.json'
    assert not status.exists() and not protocol_path.exists()
    development=read(runs/'planner_initial_goal_recovery_study_r1_status.json')
    assert development['stage']=='completed', 'Finish development study before confirmation'
    for skill in ['pass','shoot']:
        assert development['results']['native'][skill]['behavior_preservation']['episodes_and_first_impacts_exact']
    models=dict(r3=runs/'planner_history_r3_contact_train/best.pt',
        control=runs/'planner_predicted_target_initial_pass/diagnostic.pt',
        candidate=runs/'planner_initial_goal_recovery_r1/diagnostic.pt')
    checksums={key:sha(path) for key,path in models.items()}
    assert checksums['r3']==BASE_SHA and checksums['control']==CONTROL_SHA and checksums['candidate']==CANDIDATE_SHA
    prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']==CONTROL_SHA and prepared['checkpoint_sha256']==CANDIDATE_SHA
    assert prepared['backend']=='cuda_graph' and prepared['steps']==16
    assert prepared['refinement_scope']=='initial_pass' and prepared['standing_recovery_mode']=='initial_backward_goal'
    assert prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source']
    assert len(prepared['checks'])==7 and all(prepared['checks'].values())
    checked=set()
    for pattern in ['*/config.json','*/summary.json','*_status.json','*_protocol.json']:checked.update(runs.glob(pattern))
    for path in checked:assert not used_seeds(read(path)).intersection(SEEDS),f'Previously used seeds: {path}'
    paths=['scripts/probe_soccer_sequence.py','src/fiber_soccer/planner_policy.py','src/fiber_soccer/predicted_target_solver.py',
        'src/fiber_soccer/initial_goal_recovery.py','src/fiber_soccer/skill_transition.py','src/fiber_soccer/bridge.py',
        'src/fiber_soccer/planner.py','src/fiber_soccer/native_backward.py','src/fiber_soccer/shared_scene.py','src/fiber_soccer/refinement_scope.py','src/fiber_soccer/captured_latent_solver.py','src/fiber_soccer/native_bfm.py','src/fiber_soccer/target_guided.py','src/fiber_soccer/flow_planner.py','AH-BFM/load_ah_bfm.py','AH-BFM/lsvla/fb_cpr_policy.py']
    sources={path:sha(ROOT/path) for path in paths}
    actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    assert actor_sha=='9d8b9ae1b775c4b57034a49a051e527c8470456b51093ff9f7d6467dceb895d9'
    protocol=dict(actor_sha256=actor_sha,seed_scope='Fresh confirmation; frozen candidate selected before these trials',seeds=SEEDS,num_envs=32,steps=1000,models={k:str(v) for k,v in models.items()},
        checkpoint_sha256=checksums,runtime_source_sha256=sources,driver_sha256=sha(Path(__file__)),
        checked_seed_files=len(checked),training=False,failed_lane_control='zero',
        purpose='Compare frozen shared policies through pass, shoot and recovery; preserve all failures; no automatic promotion',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed relative error <=0.30, no failure through 20 seconds',
        model_differences='r3 can differ throughout; control/candidate differ only after recovery starts. Global prefix through first recovery must match.')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        temp=status.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),seeds=SEEDS,results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        for seed in SEEDS:
            pair={};contracts={};task_protocols={}
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
                assert summary['one_policy_instance'] and not summary['expert_loaded'] and summary['requested']==32
                assert summary['failed_lane_control']=='zero' and summary['steps']==1000
                if label!='r3':
                    assert summary['shoot_controller']=='predicted_target_latent_solver'
                    assert summary['predicted_target_solver_steps']==16
                    assert summary['predicted_target_solver_backend']=='cuda_graph' and not summary['predicted_target_solver_expert_targets_used']
                    assert summary['predicted_target_refinement_scope']=='initial_pass'
                if label=='candidate':
                    assert summary['standing_recovery_mode']=='initial_backward_goal'
                    assert summary['standing_recovery_active_control_steps']>=0
                else:assert 'standing_recovery_mode' not in summary
                pair[label]=dict(summary=summary,outcomes=outcomes(summary))
                task_protocol=read(destination/'protocol.json')
                expected_controller='unified_planner' if label=='r3' else 'predicted_target_latent_solver'
                assert task_protocol.pop('shoot_controller')==expected_controller
                if label!='r3':
                    assert task_protocol.pop('predicted_target_solver_steps')==16
                    assert task_protocol.pop('predicted_target_solver_backend')=='cuda_graph'
                    assert task_protocol.pop('predicted_target_solver_expert_targets_used') is False
                    assert task_protocol.pop('predicted_target_refinement_scope')=='initial_pass'
                if label=='candidate':assert task_protocol.pop('standing_recovery_mode')=='initial_backward_goal'
                task_protocols[label]=task_protocol;contracts[label]=read(destination/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert sources=={path:sha(ROOT/path) for path in paths},'Runtime code changed during comparison'
            for label in ['control','candidate']:
                assert pair[label]['summary']['initial_state_sha256']==pair['r3']['summary']['initial_state_sha256']
                assert task_protocols[label]==task_protocols['r3'] and contracts[label]==contracts['r3']
            before=pair['control']['summary'];after=pair['candidate']['summary']
            first=min((v['tick'] for v in before['recovery_switches']+after['recovery_switches']),default=1000)
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in before[key].items() if v['tick']<=first}=={k:v for k,v in after[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in before[key] if v['tick']<=first]==[v for v in after[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,
                control_candidate_global_prefix_through_first_recovery_exact=True,
                task_protocol_equal_except_declared_controller=True,scene_contracts_equal=True,
                robot_history_handoff_and_failure_accounting_verified=True))
        totals={label:{} for label in models}
        for entry in results['sequences'].values():
            for label,record in entry['models'].items():
                for key,value in record['outcomes'].items():
                    if isinstance(value,int):totals[label][key]=totals[label].get(key,0)+value
                    elif isinstance(value,dict):
                        group=totals[label].setdefault(key,{})
                        for k,v in value.items():group[k]=group.get(k,0)+v
        results['totals']=totals
        assert checksums=={k:sha(v) for k,v in models.items()}
        assert sha(ROOT/'AH-BFM/AH-BFM.pt')==actor_sha
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n')
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
