"""Gated UFO smoke, teacher warm-start, autonomous projection and evaluation."""
import json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'runs'
base=[str(ROOT/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
def run(name,args):
 if (RUNS/name).exists():raise RuntimeError(f'Output exists {name}')
 (RUNS/'ufo_pilot_status.json').write_text(json.dumps(dict(stage=name,status='running')))
 with (RUNS/f'{name}.log').open('x') as f:subprocess.run(base+args+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=f,stderr=subprocess.STDOUT,check=True)
 if '--ufo-train' in args:
  audit=json.loads((RUNS/name/'parameter_audit.json').read_text());assert audit['actor_frozen'] and audit['encoder_frozen']
 else:
  summary=json.loads((RUNS/name/'summary.json').read_text());assert summary['completed']==64
try:
 probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True);assert probe.returncode==1
 subprocess.run(['docker','exec','fiber-soccer-runtime','/isaac-sim/python.sh','-c','import onnxruntime'],check=True)
 run('ufo_bridge_smoke_v3',['--ufo-train','--ufo-teacher-collect','--num-envs','8','--steps','400','--optimization-passes','1'])
 run('ufo_teacher_warmstart_v1',['--ufo-train','--ufo-teacher-collect','--num-envs','64','--steps','64000','--seed','810211'])
 run('ufo_online_v1',['--ufo-train','--num-envs','64','--steps','201600','--seed','810212','--resume-projector','/workspace/fiber-soccer/runs/ufo_teacher_warmstart_v1/latest.pt'])
 run('eval_ufo_online_v1',['--evaluate','--num-envs','64','--steps','2000','--seed','810127','--projector-checkpoint','/workspace/fiber-soccer/runs/ufo_online_v1/latest.pt'])
 (RUNS/'ufo_pilot_status.json').write_text(json.dumps(dict(status='completed',stage='pilot_evaluated',note='Exploratory pilot on unchanged expert dynamics; not a matched-training-budget algorithm comparison.')))
except Exception as e:
 (RUNS/'ufo_pilot_status.json').write_text(json.dumps(dict(status='failed',error=repr(e))));raise
