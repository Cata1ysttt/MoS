"""Queue fresh native-task confirmation of the unified history policy against r3."""
import json
import os
import subprocess
import time
from run_target_guided_planner import ROOT, read, sha, container

TAG='planner_unified_native_confirmation_r1'
SEEDS=[818101,818102]


def main():
    runs=ROOT/'runs';status=runs/f'{TAG}_status.json';assert not status.exists()
    parent_path=runs/'planner_post_pass_latent_ppo_low_noise_r1_status.json'
    parent=read(parent_path);parent_pid=parent['pid']
    models={'r3':runs/'planner_history_r3_contact_train/best.pt',
            'history':runs/'planner_new_ball_task_history_r1/diagnostic.pt'}
    sources=dict(parent['results']['protocol']['runtime_source_sha256'])
    sources['scripts/confirm_unified_native_skills.py']=sha(ROOT/'scripts/confirm_unified_native_skills.py')
    protocol=dict(seeds=SEEDS,num_envs=32,steps=2000,models={k:dict(path=str(v),sha256=sha(v)) for k,v in models.items()},
        runtime_source_sha256=sources,actor_sha256=sha(ROOT/'AH-BFM/AH-BFM.pt'),wait_for_pid=parent_pid,
        scope='Fresh native-task confirmation. Each model uses one checkpoint for both skills. First completed episode per environment. Not a continuous-switching test.',
        automatic_promotion=False)
    results={}
    def state(stage,**extra):
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),protocol=protocol,results=results,**extra),indent=2)+'\n');temporary.replace(status)
    def verify():
        assert all(sha(ROOT/p)==h for p,h in sources.items()),'Frozen source changed'
        assert all(sha(v)==protocol['models'][k]['sha256'] for k,v in models.items())
        assert sha(ROOT/'AH-BFM/AH-BFM.pt')==protocol['actor_sha256']
    try:
        verify()
        while True:
            parent=read(parent_path);assert parent['pid']==parent_pid
            if parent['stage']=='completed':break
            assert parent['stage']!='failed','Inspect prior study failure before proceeding'
            os.kill(parent_pid,0);state('waiting_for_low_noise_study',parent_stage=parent['stage']);time.sleep(10)
        verify()
        for seed in SEEDS:
            results[str(seed)]={}
            for skill in ['pass','shoot']:
                pair={};results[str(seed)][skill]=pair
                for label,checkpoint in models.items():
                    state('evaluating',seed=seed,skill=skill,model=label)
                    dest=runs/f'{TAG}_{seed}_{skill}_{label}'
                    entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
                    cmd=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
                         '--seed',str(seed),'--num-envs','32','--steps','2000','--projector-checkpoint',container(checkpoint),'--output',container(dest)]
                    if skill=='pass':cmd+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
                    with dest.with_suffix('.log').open('x') as stream:
                        subprocess.run([str(ROOT/'scripts/container_exec.sh'),*cmd],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES='3'),check=True)
                    verify();sm=read(dest/'summary.json')
                    assert sm['requested']==sm['completed']==32 and not sm['expert_assisted_oracle']
                    assert sm['seed']==seed and sm['checkpoint']==container(checkpoint)
                    episodes=read(dest/'episodes.json');assert len(episodes)==32
                    assert len({row['environment'] for row in episodes})==32
                    if label=='history':
                        assert sm['new_ball_history_active_control_steps']==0 and sm['new_ball_task_history_active_control_steps']==0
                        assert sm['standing_recovery_active_control_steps']==0
                    counts={}
                    for key in ['quality_kick_success','quality_ball_direction_success_10deg','quality_wrong_foot_kick']:
                        value=sm[key]*32;assert abs(value-round(value))<1e-5
                        counts[key]=round(value)
                    pair[label]=dict(summary=sm,counts=counts,summary_sha256=sha(dest/'summary.json'),episodes_sha256=sha(dest/'episodes.json'),first_impact_sha256=sha(dest/'first_kick_velocity.json'))
                assert pair['r3']['summary']['initial_state_sha256']==pair['history']['summary']['initial_state_sha256']
        totals={skill:{label:{'requested':64,**{key:sum(results[str(seed)][skill][label]['counts'][key] for seed in SEEDS) for key in results[str(SEEDS[0])][skill][label]['counts']}} for label in models} for skill in ['pass','shoot']}
        verify();state('completed',totals=totals)
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
