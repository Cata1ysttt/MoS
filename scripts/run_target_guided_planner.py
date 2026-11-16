"""Train target-guided latents, verify export, then screen both native tasks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def container(path):return '/workspace/fiber-soccer/'+str(path.relative_to(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default='planner_target_guided_r1')
    p.add_argument('--distillation',action='store_true')
    a=p.parse_args();assert a.tag.replace('_','').isalnum()
    runs=ROOT/'runs';run=runs/a.tag;status=runs/f'{a.tag}_status.json'
    assert not run.exists() and not status.exists()
    base=runs/'planner_history_r3_contact_train/best.pt'
    predictor=runs/'planner_target_predictability_r1/final.pt'
    assert sha(base)=='5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133'
    assert sha(predictor)=='d71d4f26456f771d36375e183d72623cbbea119b6c2d83431c383ce052b5e99b'
    smoke_name='planner_target_guided_distillation_smoke' if a.distillation else 'planner_target_guided_smoke'
    smoke=read(runs/smoke_name/'complete.json')
    if a.distillation:assert smoke['training_only_labels_verified'] and smoke['warm_start_matches_source']
    assert smoke['smoke_test'] and smoke['deployable_roundtrip'] and smoke['loading_rng_matches_r3']
    assert smoke['prior_and_predictor_unchanged'] and smoke['actor_frozen'] and smoke['zero_initialization_matches_r3']
    results={}
    def state(stage,**details):
        temporary=status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),results=results,**details),indent=2)+'\n')
        temporary.replace(status)
    def execute(command,log,gpu):
        with log.open('x') as f:
            subprocess.run([str(ROOT/'scripts/container_exec.sh')]+command,stdout=f,stderr=subprocess.STDOUT,
                           env=dict(os.environ,MOS_CUDA_VISIBLE_DEVICES=str(gpu)),check=True)
    try:
        state('training')
        data=[runs/f'planner_recovery_r{i}_sum_semantic/{skill}_trajectory.pt' for i in [2,3] for skill in ['pass','shoot']]
        extra=[]
        if a.distillation:
            extra=['--warm-start',container(runs/'planner_target_guided_r1/final.pt'),
                '--latent-labels',container(runs/'planner_target_guided_labels_r1'),'--seed','816021']
        execute(['/workspace/fiber-soccer/scripts/train_target_guided_planner.py',
            '--base',container(base),'--predictor',container(predictor),'--data',*[container(v) for v in data],
            '--output',container(run),'--updates','20000',*extra],run.with_suffix('.log'),2)
        done=read(run/'complete.json');assert done['updates']==20000 and not done['smoke_test']
        assert done['prior_and_predictor_unchanged'] and done['actor_frozen'] and done['deployable_roundtrip']
        assert done['loading_rng_matches_r3'] and done['zero_initialization_matches_r3']
        checksum=sha(run/'final.pt');assert checksum==done['checkpoint_sha256']
        results['training']=done;results['native']={}
        for skill in ['pass','shoot']:
            state('evaluating_native',skill=skill,checkpoint_sha256=checksum)
            destination=runs/f'{a.tag}_{skill}_eval'
            entry='expert_baseline.py' if skill=='pass' else 'probe_shoot_expert.py'
            command=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate',
                '--seed','811314','--num-envs','16','--steps','2000',
                '--projector-checkpoint',container(run/'final.pt'),'--output',container(destination)]
            if skill=='pass':command+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
            execute(command,destination.with_suffix('.log'),3)
            summary=read(destination/'summary.json');baseline=read(runs/f'planner_history_r3_contact_{skill}_eval/summary.json')
            assert summary['completed']==summary['requested']==16 and not summary['expert_assisted_oracle']
            assert summary['initial_state_sha256']==baseline['initial_state_sha256']
            assert summary['controller']=='unified_latent_planner' and summary['backbone']=='ah_bfm'
            assert sha(run/'final.pt')==checksum
            results['native'][skill]=dict(candidate=summary,baseline=baseline,
                initial_tensors_equal=True,scope='Repeated development seed, not fresh generalization evaluation.')
        state('completed',checkpoint_sha256=checksum,
              scope='Fixed final checkpoint, offline mapping and both native task screens. No automatic promotion; continuous execution remains to be evaluated.')
    except Exception as error:
        state('failed',error=repr(error));raise


if __name__=='__main__':main()
