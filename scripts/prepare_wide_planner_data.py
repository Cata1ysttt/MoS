"""Wait for verified live wide collectors and build semantic training data."""
import json,subprocess,time,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs';names=['planner_wide_pass_s102','planner_wide_shoot_s101'];status=runs/'planner_wide_data_status.json'
def state(stage,**extra):status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra),indent=2))
try:
 while True:
  pending=[n for n in names if not (runs/n/'complete.json').exists()]
  if not pending:break
  process=subprocess.check_output(['docker','top','fiber-soccer-runtime','-eo','pid,args'],text=True)
  missing=[n for n in pending if '/runs/'+n not in process]
  if missing:raise RuntimeError('Collector no longer live: '+str(missing))
  state('collection',status='running',pending=pending);time.sleep(30)
 state('semantic_labels',status='running')
 with (runs/'planner_semantic_wide_v1.log').open('x') as f:
  subprocess.run([str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/prepare_semantic_planner_data.py','--data',*['/workspace/fiber-soccer/runs/'+n+'/trajectory.pt' for n in names],'--output','/workspace/fiber-soccer/runs/planner_semantic_wide_v1'],stdout=f,stderr=subprocess.STDOUT,check=True)
 state('completed',status='completed')
except Exception as e:state('failed',status='failed',error=repr(e));raise
