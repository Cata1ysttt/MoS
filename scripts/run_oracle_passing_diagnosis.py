"""Matched expert / autonomous / expert-assisted diagnostic; no policy parity claim."""
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'
base=[str(ROOT/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt','--evaluate','--num-envs','16','--steps','2000','--seed','810161']
status=RUNS/'oracle_passing_diagnosis_status.json'
try:
 probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True)
 if probe.returncode!=1:raise RuntimeError('Simulator not confirmed idle')
 for name,extra in [('oracle_diag_expert16',[]),('oracle_diag_student16',['--projector-checkpoint','/workspace/fiber-soccer/runs/passing_50hz_long_s1/latest.pt']),('oracle_diag_assisted16',['--oracle-latent-steps','64'])]:
  output=RUNS/name
  if output.exists():raise RuntimeError(f'Refuse overwrite {output}')
  status.write_text(json.dumps(dict(stage=name,status='running')))
  with output.with_suffix('.log').open('x') as log:
   subprocess.run(base+extra+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT,check=True)
  summary=json.loads((output/'summary.json').read_text())
  if summary['completed']!=16:raise RuntimeError('Incomplete evaluation')
 status.write_text(json.dumps(dict(stage='diagnosis',status='completed')))
except Exception as error:
 status.write_text(json.dumps(dict(stage='diagnosis',status='failed',error=repr(error))))
 raise
