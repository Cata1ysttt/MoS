"""Diagnostic only: compare unified policy with raw/full-history shooting expert takeover."""
import json
import os
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from audit_initial_goal_execution import audit as recovery_audit

TAG='planner_shoot_expert_history_study_r1'
SEEDS=[816201,816202]
EXPERT_SHA='38c72a52330a9bdfda663e6312d3d451b7b55dec19d720ee818b12eddf41c43c'


def expert_audit(directory,mode):
    sm=read(directory/'summary.json');records=read(directory/'shoot_expert_diagnostic.json')
    assert len(records)==sm['steps']
    starts={i:v['tick'] for v in sm['switches'] for i in v['environments']}
    failures={v['environment']:v['tick'] for v in sm['first_resets']}
    recovery={i:v['tick'] for v in sm['recovery_switches'] for i in v['environments']}
    result={}
    for frames,key in [(50,'new_ball_history'),(10,'new_ball_task_history')]:
        expected=[0]*sm['steps']
        if mode!='raw':
            for i,start in starts.items():
                end=min(start+frames,sm['steps'],failures.get(i,sm['steps']-1)+1,recovery.get(i,sm['steps']))
                for tick in range(start,end):expected[tick]+=1
        assert [r[key+'_environments'] for r in records]==expected
        assert sum(expected)==sm['expert_'+key+'_active_control_steps']
        result[key+'_active_control_steps']=sum(expected)
    result['activation_matches_live_new_ball_intervals']=True
    return result


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    confirmation=read(runs/'planner_new_ball_task_confirmation_r1_status.json');assert confirmation['stage']=='completed'
    previous=read(runs/'planner_new_ball_task_history_study_r1_status.json');assert previous['stage']=='completed'
    smoke=read(runs/'planner_shoot_expert_history_routing_r2.json');assert smoke['expert_sha256']==EXPERT_SHA
    assert smoke['native_raw_actions_exact'] and smoke['full_history_actions_equal_manual_onnx']
    models=dict(unified=runs/'planner_new_ball_task_history_r1/diagnostic.pt',
        expert_raw=runs/'planner_initial_goal_recovery_r1/diagnostic.pt',expert_full=runs/'planner_initial_goal_recovery_r1/diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()}
    assert checksums['unified']=='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'
    assert checksums['expert_raw']==checksums['expert_full']=='7c98136ec5848fbf9ca67f7132f471548d0f4c098266c94df4b67cb993361fdc'
    paths=list(confirmation['results']['protocol']['runtime_source_sha256'])
    paths+=['scripts/run_shoot_expert_history_study.py','src/fiber_soccer/shoot_expert_diagnostic.py']
    sources={p:sha(ROOT/p) for p in paths};expert_sha=sha(ROOT/'body_yaw_49999.onnx');assert expert_sha==EXPERT_SHA
    protocol=dict(seeds=SEEDS,seed_scope='Reused development seeds, explicit expert diagnostic; not autonomous policy confirmation',
        num_envs=32,steps=1000,models={k:str(v) for k,v in models.items()},checkpoint_sha256=checksums,
        runtime_source_sha256=sources,expert_sha256=expert_sha,actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),
        training=False,failed_lane_control='zero',automatic_promotion=False,
        difference='Same initial passing and backward-goal recovery. Shooting uses unified AH-BFM, original ONNX with raw histories, or original ONNX with full new-ball histories. Expert raw environment action order and action contract preserved.',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed error <=30%, no failure through 20 seconds')
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,checkpoint in models.items():
                state('evaluating_sequence',seed=seed,model=label);dest=runs/f'{TAG}_{seed}_{label}'
                command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(checkpoint),'--output',container(dest),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero']
                if label!='unified':
                    mode='raw' if label=='expert_raw' else 'initialize_shoot_ball_task'
                    command+=['--shoot-expert-diagnostic','--shoot-expert-history-mode',mode]
                with dest.with_suffix('.log').open('x') as log:
                    subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
                sm=read(dest/'summary.json');assert sm['checkpoint_sha256']==checksums[label]==sha(checkpoint)
                assert sm['requested']==32 and sm['steps']==1000 and sm['failed_lane_control']=='zero'
                task=read(dest/'protocol.json');audits=dict(recovery=recovery_audit(dest))
                if label=='unified':
                    old=previous['results']['sequences'][str(seed)]['models']['candidate']['summary']
                    assert {k:v for k,v in sm.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'}
                    assert not sm['expert_loaded'] and sm['one_policy_instance']
                    assert task.pop('new_ball_history_mode')=='initialize_shoot_ball_task'
                else:
                    assert sm['expert_loaded'] and not sm['one_policy_instance'] and sm['shoot_controller']=='original_expert'
                    assert sm['shoot_expert_sha256']==expert_sha and sm['expert_base_inputs_match_environment']
                    assert sm['shoot_expert_history_mode']==task.pop('shoot_expert_history_mode')==mode
                    assert task.pop('shoot_expert_sha256')==expert_sha
                    task.pop('expert_comparison_scope')
                    audits['expert_history']=expert_audit(dest,mode)
                task.pop('shoot_controller')
                pair[label]=dict(summary=sm,outcomes=outcomes(sm),execution_audits=audits)
                protocols[label]=task;contracts[label]=read(dest/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert all(sha(ROOT/p)==digest for p,digest in sources.items())
            base=pair['unified']['summary']
            for label in ['expert_raw','expert_full']:
                other=pair[label]['summary'];assert base['initial_state_sha256']==other['initial_state_sha256']
                assert protocols['unified']==protocols[label] and contracts['unified']==contracts[label]
                first=min((v['tick'] for v in base['switches']+other['switches']),default=1000)
                for key in ['pass_impacts','shoot_impacts']:
                    assert {k:v for k,v in base[key].items() if v['tick']<=first}=={k:v for k,v in other[key].items() if v['tick']<=first}
                for key in ['switches','first_resets','contact_observations','recovery_switches']:
                    assert [v for v in base[key] if v['tick']<=first]==[v for v in other[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                task_protocol_equal_except_declared_shoot_controller=True,global_prefix_through_first_handoff_exact=True,
                unified_control_reproduced=True))
        totals={k:{} for k in models}
        for e in results['sequences'].values():
            for label,r in e['models'].items():
                for key,value in r['outcomes'].items():
                    if isinstance(value,int):totals[label][key]=totals[label].get(key,0)+value
                    elif isinstance(value,dict):
                        group=totals[label].setdefault(key,{})
                        for k,v in value.items():group[k]=group.get(k,0)+v
        results['totals']=totals
        assert sha(ROOT/'body_yaw_49999.onnx')==expert_sha and sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n');state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
