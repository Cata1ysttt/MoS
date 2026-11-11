"""More optimization passes, paired parent/budget with serial4."""
import subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py','--headless','--device','cuda:0','--checkpoint','/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
for name,extra in [('lpp_updates20_v1',['--train-lpp','--num-envs','64','--steps','200000','--latent-execution','slerp','--optimization-passes','20','--resume-projector','/workspace/fiber-soccer/runs/lpp_serial_3/latest.pt']),('eval_updates20_v1',['--evaluate','--num-envs','16','--steps','2000','--projector-checkpoint','/workspace/fiber-soccer/runs/lpp_updates20_v1/latest.pt'])]:
 if (root/'runs'/name).exists():raise RuntimeError(f'Refuse overwrite {name}')
 with (root/'runs'/f'{name}.log').open('w') as log:
  subprocess.run(base+extra+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT,check=True)
