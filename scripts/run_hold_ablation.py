"""Queue a hold-vs-SLERP branch after the serial control completes."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
while True:
 probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True)
 if probe.returncode not in (0,1):raise RuntimeError('Cannot verify simulator state')
 status=json.loads((root/'runs/serial_status.json').read_text())
 if probe.returncode==1 and status.get('stage')==4 and status.get('status')=='evaluated':break
 if status.get('status','').endswith('failed'):raise RuntimeError(status)
 time.sleep(15)
base=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
for name,extra in [('lpp_hold_v1',['--train-lpp','--num-envs','64','--steps','200000','--latent-execution','hold','--resume-projector','/workspace/fiber-soccer/runs/lpp_serial_3/latest.pt']),('eval_hold_v1',['--evaluate','--num-envs','16','--steps','2000','--projector-checkpoint','/workspace/fiber-soccer/runs/lpp_hold_v1/latest.pt'])]:
 if (root/'runs'/name).exists():raise RuntimeError(f'Refuse overwrite {name}')
 with (root/'runs'/f'{name}.log').open('w') as log:
  subprocess.run(base+extra+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT,check=True)
