"""Run balanced pilot and paired autonomous evaluations without overwriting runs."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs';status=runs/'pass_shoot_pilot_status.json'
base=[str(root/'scripts/container_exec.sh')]
checkpoint='/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'
model='/workspace/fiber-soccer/runs/pass_shoot_joint_pilot_v1/latest.pt'
def run(name,args):
 status.write_text(json.dumps(dict(stage=name,status='running'),indent=2))
 with (runs/(name+'.log')).open('x') as f:subprocess.run(base+args,stdout=f,stderr=subprocess.STDOUT,check=True)
try:
 assert (runs/'shoot_expert_probe_v1/teacher_records.pt').exists()
 run('pass_shoot_joint_pilot_v1',['/workspace/fiber-soccer/scripts/train_pass_shoot_pilot.py','--pass-data','/workspace/fiber-soccer/runs/expert_dense_v1/expert_dense.pt','--shoot-data','/workspace/fiber-soccer/runs/shoot_expert_probe_v1/teacher_records.pt','--output','/workspace/fiber-soccer/runs/pass_shoot_joint_pilot_v1','--updates','1000'])
 for skill in ['shoot','pass']:
  for controlled in [False,True]:
   name=f'pass_shoot_{skill}_{"joint" if controlled else "expert"}_eval_v1'
   args=['/workspace/fiber-soccer/scripts/'+('probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'),'--headless','--device','cuda:0','--evaluate','--steps','2000','--num-envs','16','--seed','811314','--output','/workspace/fiber-soccer/runs/'+name]
   if skill=='pass':args+=['--checkpoint',checkpoint]
   if controlled:args+=['--projector-checkpoint',model]
   run(name,args)
 status.write_text(json.dumps(dict(status='completed'),indent=2))
except Exception as e:
 status.write_text(json.dumps(dict(status='failed',error=repr(e)),indent=2));raise
