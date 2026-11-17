"""Small paired native-task regression on GPU2, independent of continuous scoring."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs';TAG='planner_dual_task_reward_r3'
status=RUNS/(TAG+'_native_status.json');assert not status.exists()
lock=(RUNS/'isaac_sim_gpu2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
results={}
def save(stage,**extra):
 temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),results=results,**extra),indent=2)+'\n');temp.replace(status)
def c(p):return '/workspace/fiber-soccer/'+str(p.relative_to(ROOT))
try:
 save('waiting_for_expert_benchmark')
 while not (RUNS/'pass_expert_655000_fixed_scene_r1/trace.pt').exists():time.sleep(10)
 for model in ['baseline','trained']:
  checkpoint=RUNS/'planner_new_ball_task_history_r1/diagnostic.pt' if model=='baseline' else RUNS/TAG/'final.pt'
  while not checkpoint.exists() or (model=='trained' and not (RUNS/TAG/'complete.json').exists()):
   train=json.loads((RUNS/(TAG+'_status.json')).read_text());assert train['stage']!='failed'
   save('waiting_for_final_checkpoint');time.sleep(10)
  if model=='trained':assert (RUNS/TAG/'complete.json').exists()
  for skill in ['pass','shoot']:
   name=f'{TAG}_native_{model}_{skill}';dest=RUNS/name
   cmd=['docker','exec','-w','/workspace/g1-rollingball','-e','CUDA_VISIBLE_DEVICES=2',
        '-e','PYTHONPATH=/workspace/fiber-soccer/src:/workspace/g1-rollingball/source/whole_body_tracking:/workspace/g1-rollingball/source/isaaclab:/workspace/g1-rollingball/source/isaaclab_tasks:/workspace/g1-rollingball/source/isaaclab_assets:/workspace/g1-rollingball/source/isaaclab_rl',
        'fiber-soccer-runtime','/isaac-sim/python.sh',c(ROOT/'scripts/evaluate_dual_task_native.py'),
        '--skill',skill,'--projector-checkpoint',c(checkpoint),'--output',c(dest),'--num-envs','16','--steps','2000','--seed','819401','--evaluate','--headless','--device','cuda:0']
   if skill=='pass':cmd+=['--checkpoint',c(ROOT/'assets/passing_stage2_20261116/model_655000.pt'),
    '--env-config','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl']
   save(name,command=cmd)
   with (RUNS/(name+'.log')).open('x') as stream:subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True)
   result=json.loads((dest/'summary.json').read_text());assert result['completed']==16 and not result['expert_assisted_oracle']
   results[model+'_'+skill]=result
 for skill in ['pass','shoot']:assert results['baseline_'+skill]['initial_state_sha256']==results['trained_'+skill]['initial_state_sha256']
 save('completed')
except Exception as error:save('failed',error=repr(error));raise
