"""Wait for live collectors, then train and evaluate one unified checkpoint."""
import json,subprocess,time,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs';prefix=[str(root/'scripts/container_exec.sh')]
status=runs/'planner_v2_status.json'
def state(stage,**more):status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**more),indent=2))
def c(path):return '/workspace/fiber-soccer/'+str(path.relative_to(root))
def execute(name,args):
 state(name,status='running')
 with (runs/(name+'.log')).open('x') as f:subprocess.run(prefix+args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 names=['planner_v2_pass_s452','planner_v2_shoot_s451']
 while True:
  pending=[name for name in names if not (runs/name/'complete.json').exists()]
  if not pending:break
  process=subprocess.check_output(['docker','top','fiber-soccer-runtime','-eo','pid,args'],text=True)
  missing=[name for name in pending if '/runs/'+name not in process]
  if missing:raise RuntimeError('Collectors stopped before completion: '+str(missing))
  state('collection',status='running',pending=pending);time.sleep(30)
 out=runs/'planner_v2_train'
 execute(out.name,['/workspace/fiber-soccer/scripts/train_latent_planner.py','--data',*[c(runs/n/'trajectory.pt') for n in names],'--output',c(out),'--updates','6000','--action-supervision','latent_teacher'])
 assert (out/'complete.json').exists()
 result={}
 for skill in ['shoot','pass']:
  name='planner_v2_'+skill+'_eval';entry='probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'
  args=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate','--projector-checkpoint',c(out/'best.pt'),'--seed','811314','--num-envs','16','--steps','2000','--output',c(runs/name)]
  if skill=='pass':args+=['--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
  execute(name,args);data=json.loads((runs/name/'summary.json').read_text())
  reference=json.loads((runs/f'pass_shoot_{skill}_expert_eval_v1/summary.json').read_text())
  assert data['completed']==16 and not data['expert_assisted_oracle']
  assert data['initial_state_sha256']==reference['initial_state_sha256']
  assert data['controller']=='unified_latent_planner'
  result[skill]=data;(runs/'planner_v2_results.json').write_text(json.dumps(result,indent=2))
 state('completed',status='completed',results=result)
except Exception as e:
 state('failed',status='failed',error=repr(e));raise
