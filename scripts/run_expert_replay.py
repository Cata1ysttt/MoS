"""After the active oracle, evaluate warm start and train with expert replay."""
import json,subprocess,time,os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
oracle_pid=1817124
while Path(f'/proc/{oracle_pid}').exists():
    time.sleep(15)
oracle=root/'runs/oracle_closed_loop_v1/summary.json'
if not oracle.exists():raise RuntimeError('Oracle exited without an evaluation summary')
probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True)
if probe.returncode!=1:raise RuntimeError('Simulator runtime is not idle')
base=[str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/expert_baseline.py',
      '--headless','--device','cuda:0','--checkpoint',
      '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
stages=[
 ('expert_oracle_pair_v1',['--evaluate','--num-envs','4','--steps','2000','--seed','810125']),
 ('eval_expert_warmstart_v1',['--evaluate','--num-envs','16','--steps','2000',
   '--projector-checkpoint','/workspace/fiber-soccer/runs/expert_warmstart_v1/best.pt']),
 ('lpp_expert_replay_v1',['--train-lpp','--num-envs','64','--steps','200000','--seed','810126',
   '--resume-projector','/workspace/fiber-soccer/runs/expert_warmstart_v1/best.pt',
   '--expert-replay','/workspace/fiber-soccer/runs/expert_dense_v1']),
 ('eval_expert_replay_v1',['--evaluate','--num-envs','16','--steps','2000',
   '--projector-checkpoint','/workspace/fiber-soccer/runs/lpp_expert_replay_v1/latest.pt'])]
for name,extra in stages:
    output=root/'runs'/name
    if output.exists():raise RuntimeError(f'Refuse overwrite {output}')
    status=root/'runs/expert_replay_driver_status.json'
    status.write_text(json.dumps(dict(stage=name,status='running')))
    with output.with_suffix('.log').open('x') as log:
        result=subprocess.run(base+extra+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT)
    status.write_text(json.dumps(dict(stage=name,status='completed' if result.returncode==0 else 'failed',exit_code=result.returncode)))
    if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    required='parameter_audit.json' if '--train-lpp' in extra else 'summary.json'
    if not (output/required).exists():raise RuntimeError(f'{name} missing {required}')
