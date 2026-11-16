"""Paired PD diagnosis on teacher and existing UFO projector, no retraining."""
import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parents[1];runs=r/'runs'
base=[str(r/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt','--evaluate','--num-envs','16','--steps','2000','--seed','810221']
def status(**kw):(runs/'ufo_pd_status.json').write_text(json.dumps(kw,indent=2))
try:
 probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True);assert probe.returncode==1
 results={}
 for actor in ['expert','ufo']:
  for native in [False,True]:
   name=f'ufo_pd_{actor}_{"native" if native else "original"}16';out=runs/name
   if out.exists():raise RuntimeError(f'Refuse overwrite {out}')
   args=base+(['--ufo-native-pd'] if native else [])
   if actor=='ufo':args+=['--projector-checkpoint','/workspace/fiber-soccer/runs/ufo_online_v1/latest.pt']
   status(stage=name,status='running')
   with (runs/f'{name}.log').open('x') as f:subprocess.run(args+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=f,stderr=subprocess.STDOUT,check=True)
   results[name]=json.loads((out/'summary.json').read_text());assert results[name]['completed']==16
 status(stage='completed',status='completed',results=results)
except Exception as e:
 status(status='failed',error=repr(e));raise
