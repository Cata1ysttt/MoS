"""Confirm full new-ball task history against the preserved recovery controller."""
import json
import os
from pathlib import Path
import subprocess
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from evaluate_predicted_target_sequence import used_seeds
from audit_new_ball_history_execution import audit as history_audit
from audit_initial_goal_execution import audit as recovery_audit

SEEDS=[816601,816602,816603,816604]
TAG='planner_new_ball_task_confirmation_r1'
CONTROL_SHA='7c98136ec5848fbf9ca67f7132f471548d0f4c098266c94df4b67cb993361fdc'
CANDIDATE_SHA='0d98300f2b299da3af5d229f8d8534a6670e9ef692c76895cde98ae232136514'


def useful_development(totals):
    control=totals['control'];candidate=totals['candidate']
    return (candidate['shots']>control['shots'] and candidate['precise_shot']>=control['precise_shot']
        and candidate['full_success']>control['full_success'] and candidate['survivors']>=control['survivors'])


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';protocol_path=runs/f'{TAG}_protocol.json'
    assert not status.exists() and not protocol_path.exists()
    development=read(runs/'planner_new_ball_task_history_study_r1_status.json')
    assert development['stage']=='completed','Finish the three-arm development comparison first'
    assert useful_development(development['results']['totals']),'Development did not meet the declared screening rule'
    for skill in ['pass','shoot']:
        r=development['results']['native'][skill]
        assert r['behavior_preservation']['episodes_and_first_impacts_exact']
        assert r['candidate']['new_ball_task_history_active_control_steps']==0
    models=dict(control=runs/'planner_initial_goal_recovery_r1/diagnostic.pt',
        candidate=runs/'planner_new_ball_task_history_r1/diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()}
    assert checksums==dict(control=CONTROL_SHA,candidate=CANDIDATE_SHA)
    prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']==CONTROL_SHA and prepared['checkpoint_sha256']==CANDIDATE_SHA
    assert prepared['new_ball_history_mode']=='initialize_shoot_ball_task'
    assert prepared['standing_recovery_mode']=='initial_backward_goal' and prepared['refinement_scope']=='initial_pass'
    assert prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source']
    assert all(prepared['checks'].values())
    checked=set()
    for pattern in ['*/config.json','*/summary.json','*_status.json','*_protocol.json']:checked.update(runs.glob(pattern))
    for p in checked:assert not used_seeds(read(p)).intersection(SEEDS),f'Previously used seeds: {p}'
    sources=development['results']['protocol']['runtime_source_sha256'].copy()
    assert all(sha(ROOT/p)==digest for p,digest in sources.items()),'Development runtime changed'
    sources['scripts/confirm_new_ball_task_history.py']=sha(Path(__file__))
    sources['scripts/audit_initial_goal_execution.py']=sha(ROOT/'scripts/audit_initial_goal_execution.py')
    actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    assert actor_sha==development['results']['protocol']['actor_sha256']
    protocol=dict(seeds=SEEDS,seed_scope='Fresh confirmation after fixed development screening; no automatic promotion',
        num_envs=32,steps=1000,models={k:str(v) for k,v in models.items()},checkpoint_sha256=checksums,
        runtime_source_sha256=sources,actor_sha256=actor_sha,checked_seed_files=len(checked),training=False,
        failed_lane_control='zero',success='Both kicks direction <=10 degrees, preferred foot, forward speed error <=30%, no failure through 20 seconds',
        model_differences='Only full new-ball RMA and ball/task input initialization; both retain the same initial backward-goal recovery. Physical state and robot/AH history preserved.',
        development_gate='More shots and full successes, no decrease in precise shots or survivors versus original recovery; screening only')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n');results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');tmp.replace(status)
    try:
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,checkpoint in models.items():
                state('evaluating_sequence',seed=seed,model=label)
                dest=runs/f'{TAG}_{seed}_{label}'
                command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(checkpoint),'--output',container(dest),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero']
                with dest.with_suffix('.log').open('x') as log:
                    subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
                sm=read(dest/'summary.json')
                assert sm['checkpoint_sha256']==checksums[label]==sha(checkpoint)
                assert sm['requested']==32 and sm['steps']==1000 and sm['one_policy_instance'] and not sm['expert_loaded']
                assert sm['failed_lane_control']=='zero' and sm['standing_recovery_mode']=='initial_backward_goal'
                assert sm['predicted_target_refinement_scope']=='initial_pass'
                task_protocol=read(dest/'protocol.json');audits=dict(recovery=recovery_audit(dest))
                if label=='candidate':
                    assert task_protocol.pop('new_ball_history_mode')==sm['new_ball_history_mode']=='initialize_shoot_ball_task'
                    audits['history']=history_audit(dest)
                else:assert 'new_ball_history_mode' not in sm and 'new_ball_history_mode' not in task_protocol
                (runs/f'{TAG}_{seed}_{label}_execution_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
                pair[label]=dict(summary=sm,outcomes=outcomes(sm),execution_audits=audits)
                protocols[label]=task_protocol;contracts[label]=read(dest/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert all(sha(ROOT/p)==digest for p,digest in sources.items()),'Frozen runtime changed'
            before=pair['control']['summary'];after=pair['candidate']['summary']
            assert before['initial_state_sha256']==after['initial_state_sha256']
            assert protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=min((v['tick'] for v in before['switches']+after['switches']),default=1000)
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in before[key].items() if v['tick']<=first}=={k:v for k,v in after[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in before[key] if v['tick']<=first]==[v for v in after[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                task_protocol_equal_except_history_mode=True,global_prefix_through_first_handoff_exact=True))
        totals={label:{} for label in models}
        for e in results['sequences'].values():
            for label,r in e['models'].items():
                for key,value in r['outcomes'].items():
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
