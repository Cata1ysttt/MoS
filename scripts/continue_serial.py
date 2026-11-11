"""Train bounded segments then evaluate, preserving optimizer and avoiding concurrent simulators."""
import subprocess,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
expert='/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'
def container(p):return '/workspace/fiber-soccer/'+str(p.relative_to(root))
base=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint',expert]
# Wait for the specifically launched pre-resume evaluator; absence plus result is required.
while True:
 p=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True)
 if p.returncode==1:
  if not (root/'runs/eval_192k_v1/summary.json').exists():raise RuntimeError('Pre-resume evaluation did not produce results')
  break
 if p.returncode not in (0,1):raise RuntimeError('Cannot verify simulation process state')
 time.sleep(15)
resume=root/'runs/lpp_long_v1/latest.pt'
for stage in range(1,5):
 out=root/'runs'/f'lpp_serial_{stage}'
 if out.exists():raise RuntimeError(f'Refuse overwrite {out}')
 with out.with_suffix('.log').open('w') as f:
  result=subprocess.run(base+['--num-envs','64','--steps','200000','--train-lpp','--resume-projector',container(resume),'--output',container(out)],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode or not (out/'parameter_audit.json').exists():
  (root/'runs/serial_status.json').write_text(json.dumps(dict(stage=stage,status='training_failed',exit_code=result.returncode)));break
 resume=out/'latest.pt';evaluation=root/'runs'/f'eval_serial_{stage}'
 with evaluation.with_suffix('.log').open('w') as f:
  result=subprocess.run(base+['--num-envs','16','--steps','2000','--evaluate','--projector-checkpoint',container(resume),'--output',container(evaluation)],stdout=f,stderr=subprocess.STDOUT)
 (root/'runs/serial_status.json').write_text(json.dumps(dict(stage=stage,status='evaluated' if result.returncode==0 else 'evaluation_failed',checkpoint=str(resume),exit_code=result.returncode)))
 if result.returncode:break
 subprocess.run(['python',str(root/'scripts/summarize_evaluations.py')],check=True)
