"""Preserve native behavior, then test scoped refinement against all-phase control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_target_guided_planner import ROOT,read,sha,container
from summarize_continuous_shoot_ppo import outcomes

TAG='planner_initial_pass_refinement_r1'
SEEDS=[816201,816202]  # Reused development seeds: this is a counterfactual, not fresh confirmation.


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json'
    assert not status.exists()
    previous=read(runs/'planner_predicted_target_sequence_r1_status.json');assert previous['stage']=='completed'
    models=dict(control=runs/'planner_predicted_target_solver_graph16/diagnostic.pt',
        candidate=runs/'planner_predicted_target_initial_pass/diagnostic.pt')
    checksums={k:sha(v) for k,v in models.items()}
    assert checksums['control']=='a327bf36b283e11fe5f768ce6f907dc707c027056bebfc4013aec23dfceedb77'
    assert checksums['candidate']=='9832d86c0d9efa6e028262ff347d7512082c2a2ef976d2294a2be8bdf484f8ba'
    prepared=read(models['candidate'].parent/'complete.json')
    assert prepared['source_sha256']=='3f4e4e6ff866802cb27f73881beaf1662d2200c58d74fd28e7f3071e96e9a9d1'
    assert prepared['checkpoint_sha256']==checksums['candidate'] and prepared['steps']==16 and prepared['backend']=='cuda_graph'
    assert prepared['refinement_scope']=='initial_pass' and prepared['reference_solver_outputs_exact']
    assert all(prepared['scope_checks'].values()) and len(prepared['scope_checks'])==5
    paths=['scripts/probe_soccer_sequence.py','src/fiber_soccer/planner_policy.py','src/fiber_soccer/predicted_target_solver.py',
        'src/fiber_soccer/captured_latent_solver.py','src/fiber_soccer/refinement_scope.py','src/fiber_soccer/native_bfm.py',
        'src/fiber_soccer/target_guided.py','src/fiber_soccer/flow_planner.py','src/fiber_soccer/evaluation.py',
        'AH-BFM/load_ah_bfm.py','AH-BFM/lsvla/fb_cpr_policy.py']
    sources={path:sha(ROOT/path) for path in paths}
    actor_sha=sha(ROOT/'AH-BFM/AH-BFM.pt')
    assert actor_sha=='9d8b9ae1b775c4b57034a49a051e527c8470456b51093ff9f7d6467dceb895d9'
    protocol=dict(seeds=SEEDS,seed_scope='Reused development seeds from preceding frozen comparison',
        models={k:str(v) for k,v in models.items()},checkpoint_sha256=checksums,runtime_source_sha256=sources,actor_sha256=actor_sha,
        num_envs=32,steps=1000,training=False,failed_lane_control='zero',
        difference='Candidate refines initial passing, then uses the same checkpoint shared r3 prior after first shooting command, including recovery',
        success='Both kicks direction <=10 degrees, preferred foot, forward speed error <=30%, survive 20 seconds',
        checkpoint_selection='Fixed metadata-only candidate, no trained weights or automatic promotion')
    (runs/f'{TAG}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    results=dict(protocol=protocol,sequences={})
    def state(stage,**details):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n');temp.replace(status)
    try:
        state('verifying_native_tasks')
        subprocess.run([sys.executable,str(ROOT/'scripts/run_predicted_target_solver.py'),'--tag','planner_predicted_target_initial_pass',
            '--wait-for','planner_predicted_target_solver_graph16_status.json'],check=True,cwd=ROOT)
        native=read(runs/'planner_predicted_target_initial_pass_status.json');assert native['stage']=='completed'
        for skill in ['pass','shoot']:assert native['results'][skill]['behavior_preservation']['episodes_and_first_impacts_exact']
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
                assert summary['predicted_target_solver_steps']==16 and summary['predicted_target_solver_backend']=='cuda_graph'
                expected_scope='all' if label=='control' else 'initial_pass'
                assert summary.get('predicted_target_refinement_scope','all')==expected_scope
                if label=='control':
                    old=previous['results']['sequences'][str(seed)]['models']['solver16']['summary']
                    assert {k:v for k,v in summary.items() if k!='policy_latency_seconds'}=={k:v for k,v in old.items() if k!='policy_latency_seconds'},'Control changed beyond timing'
                pair[label]=dict(summary=summary,outcomes=outcomes(summary))
                protocols[label]=read(destination/'protocol.json');assert protocols[label].pop('predicted_target_refinement_scope','all')==expected_scope
                contracts[label]=read(destination/'scene_contracts.json')
                results['sequences'][str(seed)]=dict(models=pair,audit_pending=True)
                assert sources=={path:sha(ROOT/path) for path in paths}
            a=pair['control']['summary'];b=pair['candidate']['summary']
            assert a['initial_state_sha256']==b['initial_state_sha256'] and protocols['control']==protocols['candidate'] and contracts['control']==contracts['candidate']
            first=min((v['tick'] for summary in [a,b] for v in summary['switches']),default=1000)
            for key in ['pass_impacts','shoot_impacts']:
                assert {k:v for k,v in a[key].items() if v['tick']<=first}=={k:v for k,v in b[key].items() if v['tick']<=first}
            for key in ['switches','first_resets','contact_observations']:
                assert [v for v in a[key] if v['tick']<=first]==[v for v in b[key] if v['tick']<=first]
            results['sequences'][str(seed)]=dict(models=pair,audit=dict(initial_tensors_equal=True,
                protocol_equal_except_scope=True,scene_contracts_equal=True,global_prefix_before_first_handoff_exact=True))
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
        (runs/f'{TAG}_results.json').write_text(json.dumps(results,indent=2)+'\n');state('completed')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
