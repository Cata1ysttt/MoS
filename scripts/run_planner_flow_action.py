"""Train/evaluate experimental backward-residual unified planner."""
import json,subprocess,time,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs';prefix=[str(root/'scripts/container_exec.sh')];status=runs/'planner_flow_action_status.json'
def c(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def state(stage,**more):status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**more),indent=2))
def execute(name,args):
 state(name,status='running')
 with (runs/(name+'.log')).open('x') as f:subprocess.run(prefix+args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 data=runs/'planner_semantic_wide_v1';assert (data/'complete.json').exists()
 out=runs/'planner_flow_action_train'
 execute(out.name,['/workspace/fiber-soccer/scripts/train_flow_planner.py','--data',c(data/'pass_trajectory.pt'),c(data/'shoot_trajectory.pt'),'--output',c(out),'--updates','10000','--resume',c(runs/'planner_flow_wide_train/best.pt'),'--action-weight','100','--leg-weight','2'])
 assert (out/'complete.json').exists()
 result={}
 for skill in ['shoot','pass']:
  name='planner_flow_action_'+skill+'_eval';entry='probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'
  args=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate','--projector-checkpoint',c(out/'best.pt'),'--seed','811314','--num-envs','16','--steps','2000','--output',c(runs/name)]
  if skill=='pass':args+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
  execute(name,args);d=json.loads((runs/name/'summary.json').read_text());ref=json.loads((runs/f'pass_shoot_{skill}_expert_eval_v1/summary.json').read_text())
  assert d['completed']==16 and not d['expert_assisted_oracle'] and d['initial_state_sha256']==ref['initial_state_sha256']
  result[skill]=d;(runs/'planner_flow_action_results.json').write_text(json.dumps(result,indent=2))
 state('completed',status='completed',results=result)
except Exception as e:
 state('failed',status='failed',error=repr(e));raise
