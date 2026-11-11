"""Evaluate immutable training checkpoints sequentially, never overlapping jobs."""
import subprocess,time,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=root/'runs/lpp_long_v1'
expert='/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'
while True:
 for ck in sorted(run.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[-1])):
  out=root/'runs'/('eval_long_'+ck.stem.split('_')[-1])
  status=out.with_suffix('.watch.json')
  if status.exists():continue
  # A running older trainer may write snapshots directly: require stable size.
  first=ck.stat().st_size;time.sleep(5)
  if ck.stat().st_size!=first:continue
  cmd=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--num-envs','16','--steps','2000','--evaluate','--checkpoint',expert,'--projector-checkpoint','/workspace/fiber-soccer/'+str(ck.relative_to(root)),'--output','/workspace/fiber-soccer/'+str(out.relative_to(root))]
  with out.with_suffix('.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  status.write_text(json.dumps(dict(checkpoint=str(ck),exit_code=r.returncode)))
 if (run/'parameter_audit.json').exists():break
 time.sleep(30)
