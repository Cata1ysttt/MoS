"""Serial 50Hz continuation with intermediate evaluations and untouched paired validation."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
probe = subprocess.run(['docker', 'exec', 'fiber-soccer-runtime', 'pgrep', '-f',
                        '[p]ython3.*expert_baseline.py'], capture_output=True)
if probe.returncode != 1:
    raise RuntimeError('Cannot establish that the simulator is idle')
base = [str(root / 'scripts/container_exec.sh'), '/workspace/fiber-soccer/scripts/expert_baseline.py',
        '--headless', '--device', 'cuda:0', '--checkpoint',
        '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
stages = []
parent='frequency_50hz_64k_v2'
for segment in range(1,4):
    name=f'passing_50hz_long_s{segment}'
    stages.extend([
        (name, ['--train-lpp','--num-envs','64','--steps','200000','--seed',str(810140+segment),
                '--latent-interval','1','--resume-projector',f'/workspace/fiber-soccer/runs/{parent}/latest.pt',
                '--expert-replay','/workspace/fiber-soccer/runs/expert_dense_v1']),
        (f'eval_{name}', ['--evaluate','--num-envs','64','--steps','2000','--seed','810127',
                         '--projector-checkpoint',f'/workspace/fiber-soccer/runs/{name}/latest.pt'])])
    parent=name
status_file = root / 'runs/passing_optimization_status.json'
for name, extra in stages:
    output = root / 'runs' / name
    if output.exists():
        raise RuntimeError(f'Refuse overwrite {output}')
    status_file.write_text(json.dumps(dict(stage=name, status='running')))
    with output.with_suffix('.log').open('x') as log:
        result = subprocess.run(base + extra + ['--output', f'/workspace/fiber-soccer/runs/{name}'],
                                stdout=log, stderr=subprocess.STDOUT)
    status_file.write_text(json.dumps(dict(stage=name, status='completed' if result.returncode == 0 else 'failed',
                                          exit_code=result.returncode)))
    if result.returncode:
        raise RuntimeError(f'{name} failed')
    artifact = 'parameter_audit.json' if '--train-lpp' in extra else 'summary.json'
    if not (output / artifact).exists():
        raise RuntimeError(f'{name} lacks required {artifact}')
    if artifact == 'summary.json':
        summary = json.loads((output / artifact).read_text())
        if summary['completed'] != summary['requested']:
            raise RuntimeError(f'{name} has incomplete episodes')

# Select on development data only, before either held-out evaluation runs.
candidates=['frequency_50hz_64k_v2']+[f'passing_50hz_long_s{i}' for i in range(1,4)]
def development(name):
    path=root/'runs'/('eval_'+name)
    summary=json.loads((path/'summary.json').read_text())
    return (summary['quality_ball_direction_success_10deg'], -summary['terminated'])
best=max(candidates,key=development)
(root/'runs/passing_selected.json').write_text(json.dumps(dict(checkpoint=best,selection='Development first-ball direction, then fewer early terminations',heldout_seed=910241)))
for name,extra in [('passing_expert_heldout128',[]),('passing_selected_heldout128',['--projector-checkpoint',f'/workspace/fiber-soccer/runs/{best}/latest.pt'])]:
    output=root/'runs'/name
    if output.exists():raise RuntimeError(f'Refuse overwrite {output}')
    status_file.write_text(json.dumps(dict(stage=name,status='running')))
    with output.with_suffix('.log').open('x') as log:
        result=subprocess.run(base+['--evaluate','--num-envs','128','--steps','2000','--seed','910241']+extra+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        status_file.write_text(json.dumps(dict(stage=name,status='failed',exit_code=result.returncode)))
        raise RuntimeError(f'{name} failed')
    summary=json.loads((output/'summary.json').read_text())
    if summary['completed']!=128:raise RuntimeError('Incomplete held-out evaluation')
subprocess.run(['/home/jiongye/P-BFM/.venv-deploy/bin/python',str(root/'scripts/compare_paired.py'),'--expert',str(root/'runs/passing_expert_heldout128'),'--candidate',str(root/'runs/passing_selected_heldout128'),'--output',str(root/'runs/passing_heldout_comparison.json')],check=True)
status_file.write_text(json.dumps(dict(stage='heldout_comparison',status='completed',selected=best)))
