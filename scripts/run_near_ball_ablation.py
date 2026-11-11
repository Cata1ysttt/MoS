"""Queue a paired development ablation after the existing held-out driver finishes."""
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
STATUS = RUNS / 'near_ball_ablation_status.json'
BASE = [str(ROOT / 'scripts/container_exec.sh'), '/workspace/fiber-soccer/scripts/expert_baseline.py',
        '--headless', '--device', 'cuda:0', '--checkpoint',
        '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']

def status(stage, **kwargs):
    temp = STATUS.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(stage=stage, **kwargs), indent=2))
    temp.replace(STATUS)

def run(name, args, artifact):
    output = RUNS / name
    if output.exists():
        raise RuntimeError(f'Refuse overwrite {output}')
    status(name, status='running')
    with output.with_suffix('.log').open('x') as log:
        subprocess.run(BASE + args + ['--output', f'/workspace/fiber-soccer/runs/{name}'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output / artifact).read_text())
    if artifact == 'summary.json' and result['completed'] != 64:
        raise RuntimeError('Incomplete development evaluation')
    if artifact == 'parameter_audit.json' and not result['bfm_gradients_absent']:
        raise RuntimeError('Frozen actor audit failed')
    return result

def main():
    with (RUNS / 'near_ball_ablation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status('waiting_for_existing_driver', status='queued')
        while True:
            prior = json.loads((RUNS / 'passing_optimization_status.json').read_text())
            if prior.get('status') == 'failed':
                raise RuntimeError(f'Previous experiment failed: {prior}')
            if prior.get('stage') == 'heldout_comparison' and prior.get('status') == 'completed':
                break
            time.sleep(30)
        probe = subprocess.run(['docker', 'exec', 'fiber-soccer-runtime', 'pgrep', '-f',
                                '[p]ython3.*expert_baseline.py'], capture_output=True)
        if probe.returncode != 1:
            raise RuntimeError('Cannot establish simulator idle')
        selected = json.loads((RUNS / 'passing_selected.json').read_text())['checkpoint']
        if Path(selected).name != selected:
            raise ValueError('Invalid checkpoint name')
        common = ['--train-lpp', '--num-envs', '64', '--seed', '810151', '--latent-interval', '1',
                  '--resume-projector', f'/workspace/fiber-soccer/runs/{selected}/latest.pt',
                  '--expert-replay', '/workspace/fiber-soccer/runs/expert_dense_v1', '--near-ball-radius', '0.6']
        run('near_ball_weight4_smoke', common + ['--steps', '3200', '--near-ball-weight', '4'], 'parameter_audit.json')
        results = {}
        for weight in (1, 4):
            name = f'passing_near_ball_w{weight}_64k'
            run(name, common + ['--steps', '64000', '--near-ball-weight', str(weight)], 'parameter_audit.json')
            results[str(weight)] = run('eval_' + name, ['--evaluate', '--num-envs', '64', '--steps', '2000',
                '--seed', '810127', '--projector-checkpoint', f'/workspace/fiber-soccer/runs/{name}/latest.pt'], 'summary.json')
        subprocess.run(['/home/jiongye/P-BFM/.venv-deploy/bin/python', str(ROOT / 'scripts/compare_paired.py'),
            '--expert', str(RUNS / 'eval_passing_near_ball_w1_64k'),
            '--candidate', str(RUNS / 'eval_passing_near_ball_w4_64k'),
            '--output', str(RUNS / 'near_ball_ablation_paired.json')], check=True)
        status('ablation_completed', status='completed', parent=selected, results=results,
               note='Development only; comparator named expert in paired file is weight1 control, not teacher.')

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('ablation_error', status='failed', error=repr(error))
        raise
