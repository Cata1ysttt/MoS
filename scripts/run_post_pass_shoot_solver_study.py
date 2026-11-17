"""Paired development trial of post-pass shooting refinement with corrected input history."""
import argparse
import json
import os
import subprocess
import sys
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes
from audit_initial_goal_execution import audit as recovery_audit
from audit_new_ball_history_execution import audit as history_audit

TAG='planner_post_pass_shoot_solver_study_r1'
SCOPE='initial_pass_and_post_pass_shoot'
SEEDS=[816201,816202]


def refinement_audit(directory):
    sm=read(directory/'summary.json');records=read(directory/'predicted_target_solver.json')
    assert len(records)==sm['steps']
    starts={i:v['tick'] for v in sm['switches'] for i in v['environments']}
    failures={v['environment']:v['tick'] for v in sm['first_resets']}
    contacts={}
    for event in sm['contact_observations']:
        if event['stage']==1:contacts.setdefault(event['environment'],event['tick'])
    expected=[0]*sm['steps']
    for i,start in starts.items():
        end=min(sm['steps'],failures.get(i,sm['steps']-1)+1,contacts.get(i,sm['steps']))
        for tick in range(start,end):expected[tick]+=1
    actual=[r['post_pass_shoot_refined_environments'] for r in records]
    assert actual==expected,'Post-pass refinement differs from live pre-contact shooting intervals'
    return dict(active_control_steps=sum(actual),live_precontact_intervals_exact=True,
        scope='Aggregate execution counts; actual per-row solver routing is verified in checkpoint preparation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',help='Reuse completed simulations after a terminal driver failure')
    args=parser.parse_args()
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json'
    if args.resume:
        stopped=read(status);assert stopped['stage']=='failed'
        try:os.kill(stopped['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Previous driver is still live')
    else:assert not status.exists()
    previous=read(runs/'planner_shoot_expert_history_study_r1_status.json');assert previous['stage']=='completed'
    models=dict(control=runs/'planner_new_ball_task_history_r1/diagnostic.pt',candidate=runs/'planner_post_pass_shoot_solver_r1/diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()};prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']==checksums['control'] and prepared['checkpoint_sha256']==checksums['candidate']
    assert prepared['frozen_parameters_unchanged'] and prepared['loading_rng_matches_source'] and all(prepared['checks'].values())
    paths=list(previous['results']['protocol']['runtime_source_sha256'])+['scripts/run_post_pass_shoot_solver_study.py','scripts/prepare_post_pass_shoot_solver.py']
    sources={p:sha(ROOT/p) for p in paths};actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    protocol=dict(seeds=SEEDS,seed_scope='Reused development seeds; fresh confirmation required after any positive screen',
        num_envs=32,steps=1000,models={k:str(v) for k,v in models.items()},checkpoint_sha256=checksums,
        runtime_source_sha256=sources,actor_sha256=actor_sha,training=False,expert_targets_used=False,
        failed_lane_control='zero',automatic_promotion=False,
        difference='Both retain full new-ball histories and initial backward-goal recovery. Candidate additionally refines internal predicted targets through AH-BFM after observed passing contact, only during shooting before contact. Native single-skill behavior must remain exact.',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed error <=30%, no failure through 20 seconds',
        development_screen='Candidate full_success > control, precise_shot >= control, survivors >= control; no automatic confirmation launch')
    protocol_path=runs/f'{TAG}_protocol.json'
    if args.resume:
        original=read(protocol_path)
        assert {k:v for k,v in original.items() if k!='runtime_source_sha256'}=={k:v for k,v in protocol.items() if k!='runtime_source_sha256'}
        changed={p:dict(original=original['runtime_source_sha256'][p],current=digest) for p,digest in sources.items() if original['runtime_source_sha256'][p]!=digest}
        assert set(changed)<= {'scripts/audit_initial_goal_execution.py','scripts/run_post_pass_shoot_solver_study.py'},'Simulation runtime changed'
        archive=runs/f'{TAG}_failure_{int(stopped["updated"])}.json'
        assert not archive.exists();archive.write_text(json.dumps(stopped,indent=2)+'\n')
        protocol['resume_audit']=dict(terminal_failure=str(archive),changed_non_simulation_sources=changed,
            simulation_runtime_unchanged=True,reason='Recovery audit now checks the declared refinement scope; driver reuses complete simulations')
        (runs/f'{TAG}_resumed_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    else:protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        state('verifying_native_tasks')
        if not args.resume:
            subprocess.run([sys.executable,str(ROOT/'scripts/run_predicted_target_solver.py'),'--tag','planner_post_pass_shoot_solver_r1',
                '--wait-for','planner_shoot_expert_history_study_r1_status.json'],cwd=ROOT,check=True)
        native=read(runs/'planner_post_pass_shoot_solver_r1_status.json');assert native['stage']=='completed'
        for skill in ['pass','shoot']:
            assert native['results'][skill]['behavior_preservation']['episodes_and_first_impacts_exact']
            assert native['results'][skill]['candidate']['new_ball_task_history_active_control_steps']==0
        results['native']=native['results']
        for seed in SEEDS:
            pair={};protocols={};contracts={}
            for label,checkpoint in models.items():
                state('evaluating_sequence',seed=seed,model=label);dest=runs/f'{TAG}_{seed}_{label}'
                command=['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py','--headless','--device','cuda:0',
                    '--checkpoint',container(checkpoint),'--output',container(dest),'--num-envs','32','--steps','1000',
                    '--seed',str(seed),'--handoff-seed',str(seed),'--scene-base','shoot','--action-contract','shoot',
                    '--reset-contract','shoot','--events-contract','shoot','--incoming-pass-command','--continuous-after-kick',
                    '--transition-steps','0','--post-shoot-recovery-steps','10','--failed-lane-control','zero']
                if not (args.resume and (dest/'summary.json').exists()):
                    assert not dest.exists(),'Inspect incomplete simulation before resuming'
                    with dest.with_suffix('.log').open('x') as log:
                        subprocess.run([str(ROOT/'scripts/container_exec.sh'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
                sm=read(dest/'summary.json');assert sm['checkpoint_sha256']==checksums[label]==sha(checkpoint)
                assert sm['requested']==32 and sm['steps']==1000 and sm['failed_lane_control']=='zero'
                assert sm['one_policy_instance'] and not sm['expert_loaded']
                task=read(dest/'protocol.json');audits=dict(recovery=recovery_audit(dest),history=history_audit(dest))
                scope=task.pop('predicted_target_refinement_scope')
                assert scope==sm['predicted_target_refinement_scope']==('initial_pass' if label=='control' else SCOPE)
                if label=='control':
                    old=previous['results']['sequences'][str(seed)]['models']['unified']['summary']
                    assert {k:v for k,v in sm.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'}
                else:audits['shoot_refinement']=refinement_audit(dest)
                pair[label]=dict(summary=sm,outcomes=outcomes(sm),execution_audits=audits)
                protocols[label]=task;contracts[label]=read(dest/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert all(sha(ROOT/p)==digest for p,digest in sources.items())
            a=pair['control']['summary'];b=pair['candidate']['summary']
            assert a['initial_state_sha256']==b['initial_state_sha256']
            assert protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=min(v['tick'] for v in a['switches']+b['switches'])
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in a[key].items() if v['tick']<=first}=={k:v for k,v in b[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations','recovery_switches']:
                assert [v for v in a[key] if v['tick']<=first]==[v for v in b[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,scene_contracts_equal=True,
                task_protocol_equal_except_solver_scope=True,global_prefix_through_first_handoff_exact=True,control_reproduced=True))
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
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n');state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
