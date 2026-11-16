"""Frozen r3 versus 16/64-step internal-target solvers on two fresh sequences."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes

SEEDS=[816201,816202]
TAG='planner_predicted_target_sequence_r1'
BASE_SHA='5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133'
FAST_SHA='a327bf36b283e11fe5f768ce6f907dc707c027056bebfc4013aec23dfceedb77'
SLOW_SHA='bd6bb12dc8b51c418a50c942a0b546777744020378886c022f1c0aa5ede76544'
SOURCE_SHA='3f4e4e6ff866802cb27f73881beaf1662d2200c58d74fd28e7f3071e96e9a9d1'


def used_seeds(value):
    if isinstance(value,list):return set().union(*(used_seeds(v) for v in value)) if value else set()
    if not isinstance(value,dict):return set()
    result=set()
    for key,item in value.items():
        if key in ['seed','handoff_seed'] and isinstance(item,int):result.add(item)
        if key=='seeds' and isinstance(item,list):result.update(v for v in item if isinstance(v,int))
        if isinstance(item,(dict,list)):result.update(used_seeds(item))
    return result


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';protocol_path=runs/f'{TAG}_protocol.json'
    assert not status.exists() and not protocol_path.exists()
    assert read(runs/'planner_predicted_target_solver_graph16_status.json')['stage']=='completed'
    smoke=read(runs/'planner_predicted_target_solver_sequence_smoke/summary.json')
    assert smoke['requested']==2 and smoke['steps']==4 and smoke['one_policy_instance'] and not smoke['expert_loaded']
    assert smoke['shoot_controller']=='predicted_target_latent_solver' and smoke['predicted_target_solver_backend']=='cuda_graph'
    models=dict(r3=runs/'planner_history_r3_contact_train/best.pt',
        solver16=runs/'planner_predicted_target_solver_graph16/diagnostic.pt',
        solver64=runs/'planner_predicted_target_solver_graph64_verified/diagnostic.pt')
    checksums={key:sha(path) for key,path in models.items()}
    assert checksums['r3']==BASE_SHA and checksums['solver16']==FAST_SHA and checksums['solver64']==SLOW_SHA
    for key in ['solver16','solver64']:
        prepared=read(models[key].parent/'complete.json')
        assert prepared['source_sha256']==SOURCE_SHA and prepared['checkpoint_sha256']==checksums[key]
        assert prepared['backend']=='cuda_graph' and prepared['steps']==int(key.removeprefix('solver'))
        assert prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source']
    assert read(models['solver64'].parent/'complete.json')['reference_solver_outputs_exact']
    checked=set()
    for pattern in ['*/config.json','*/summary.json','*_status.json','*_protocol.json']:checked.update(runs.glob(pattern))
    for path in checked:assert not used_seeds(read(path)).intersection(SEEDS),f'Previously used seeds: {path}'
    paths=['scripts/probe_soccer_sequence.py','src/fiber_soccer/planner_policy.py','src/fiber_soccer/predicted_target_solver.py',
        'src/fiber_soccer/captured_latent_solver.py','src/fiber_soccer/native_bfm.py','src/fiber_soccer/target_guided.py','src/fiber_soccer/flow_planner.py']
    sources={path:sha(ROOT/path) for path in paths}
    protocol=dict(seeds=SEEDS,num_envs=32,steps=1000,models={k:str(v) for k,v in models.items()},
        checkpoint_sha256=checksums,runtime_source_sha256=sources,driver_sha256=sha(Path(__file__)),
        checked_seed_files=len(checked),training=False,failed_lane_control='zero',
        purpose='Compare frozen shared policies through pass, shoot and recovery; preserve all failures; no automatic promotion',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed relative error <=0.30, no failure through 20 seconds',
        model_differences='Both skills can change; no shared pre-handoff action/prefix assumption')
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
                    assert summary['predicted_target_solver_steps']==int(label.removeprefix('solver'))
                    assert summary['predicted_target_solver_backend']=='cuda_graph' and not summary['predicted_target_solver_expert_targets_used']
                pair[label]=dict(summary=summary,outcomes=outcomes(summary))
                task_protocol=read(destination/'protocol.json')
                expected_controller='unified_planner' if label=='r3' else 'predicted_target_latent_solver'
                assert task_protocol.pop('shoot_controller')==expected_controller
                if label!='r3':
                    assert task_protocol.pop('predicted_target_solver_steps')==int(label.removeprefix('solver'))
                    assert task_protocol.pop('predicted_target_solver_backend')=='cuda_graph'
                    assert task_protocol.pop('predicted_target_solver_expert_targets_used') is False
                task_protocols[label]=task_protocol;contracts[label]=read(destination/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert sources=={path:sha(ROOT/path) for path in paths},'Runtime code changed during comparison'
            for label in ['solver16','solver64']:
                assert pair[label]['summary']['initial_state_sha256']==pair['r3']['summary']['initial_state_sha256']
                assert task_protocols[label]==task_protocols['r3'] and contracts[label]==contracts['r3']
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,
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
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n')
        state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
