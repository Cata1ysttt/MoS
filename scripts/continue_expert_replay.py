"""Evaluate a fresh scenario seed, then continue the promising replay branch."""
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
stages = [
    ('eval_replay_v1_new64', ['--evaluate', '--num-envs', '64', '--steps', '2000', '--seed', '810127',
                            '--projector-checkpoint', '/workspace/fiber-soccer/runs/lpp_expert_replay_v1/latest.pt']),
    ('expert_replay_new64', ['--evaluate', '--num-envs', '64', '--steps', '2000', '--seed', '810127']),
    ('lpp_expert_replay_v2', ['--train-lpp', '--num-envs', '64', '--steps', '200000', '--seed', '810128',
                            '--resume-projector', '/workspace/fiber-soccer/runs/lpp_expert_replay_v1/latest.pt',
                            '--expert-replay', '/workspace/fiber-soccer/runs/expert_dense_v1']),
    ('eval_expert_replay_v2', ['--evaluate', '--num-envs', '16', '--steps', '2000', '--seed', '810123',
                             '--projector-checkpoint', '/workspace/fiber-soccer/runs/lpp_expert_replay_v2/latest.pt']),
    ('eval_replay_v2_new64', ['--evaluate', '--num-envs', '64', '--steps', '2000', '--seed', '810127',
                            '--projector-checkpoint', '/workspace/fiber-soccer/runs/lpp_expert_replay_v2/latest.pt']),
]
status_file = root / 'runs/replay_continuation_status.json'
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
