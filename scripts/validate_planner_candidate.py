"""Paired fresh-seed evaluation of one frozen planner checkpoint and both experts."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--baseline-checkpoint',type=Path,help='Optional frozen planner baseline on the same physical initial states')
parser.add_argument('--seed', type=int, default=812801)
parser.add_argument('--num-envs', type=int, default=32)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
runs = root / 'runs'
checkpoint = args.checkpoint.resolve()
assert (checkpoint.parent / 'complete.json').exists(), 'Training must be complete'
config = json.loads((checkpoint.parent / 'config.json').read_text())
assert args.seed != 811314 and args.seed not in {d['seed'] for d in config['data']}
assert args.tag.replace('_', '').isalnum()
checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
baseline=args.baseline_checkpoint.resolve() if args.baseline_checkpoint else None
baseline_hash=None;baseline_seeds=[]
if baseline:
    assert (baseline.parent/'complete.json').exists(),'Baseline training must be complete'
    baseline_config=json.loads((baseline.parent/'config.json').read_text())
    baseline_seeds=[d['seed'] for d in baseline_config['data']]
    assert args.seed not in baseline_seeds,'Validation seed appears in baseline training'
    baseline_hash=hashlib.sha256(baseline.read_bytes()).hexdigest()
status_path = runs / (args.tag + '_status.json')
expert = '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'


def container(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(root))


def state(stage, **details):
    status_path.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), updated=time.time(), **details), indent=2))


protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, seed=args.seed,
                num_envs=args.num_envs, training_seeds=[d['seed'] for d in config['data']],
                protocol='First completed episode per environment; fresh seed; paired initial state hashes')
if baseline:protocol.update(baseline_checkpoint=str(baseline),baseline_checkpoint_sha256=baseline_hash,baseline_training_seeds=baseline_seeds)
with (runs / (args.tag + '_protocol.json')).open('x') as stream:
    json.dump(protocol, stream, indent=2)
results = {}
try:
    for skill in ['shoot', 'pass']:
        results[skill] = {}
        for controller in ['planner']+(['baseline'] if baseline else [])+['expert']:
            name = args.tag + '_' + skill + '_' + controller
            entry = 'probe_shoot_expert.py' if skill == 'shoot' else 'expert_baseline.py'
            command = [str(root / 'scripts/container_exec.sh'), '/workspace/fiber-soccer/scripts/' + entry,
                       '--headless', '--device', 'cuda:0', '--evaluate', '--seed', str(args.seed),
                       '--num-envs', str(args.num_envs), '--steps', '2000', '--output', container(runs / name)]
            if skill == 'pass':
                command += ['--checkpoint', expert]
            if controller in ('planner','baseline'):
                command += ['--projector-checkpoint', container(checkpoint if controller=='planner' else baseline)]
            state(name, status='running', checkpoint_sha256=checkpoint_hash)
            with (runs / (name + '.log')).open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((runs / name / 'summary.json').read_text())
            assert result['completed'] == result['requested'] == args.num_envs and not result['expert_assisted_oracle']
            assert result['seed']==args.seed
            assert result['controller'] == ('unified_latent_planner' if controller in ('planner','baseline') else 'expert')
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
            if baseline:assert hashlib.sha256(baseline.read_bytes()).hexdigest()==baseline_hash
            results[skill][controller] = result
            (runs / (args.tag + '_results.json')).write_text(json.dumps(results, indent=2))
        assert len({value['initial_state_sha256'] for value in results[skill].values()})==1
    state('completed', status='completed', protocol=protocol, results=results)
except Exception as error:
    state('failed', status='failed', error=repr(error))
    raise
