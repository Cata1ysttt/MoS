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
status_path = runs / (args.tag + '_status.json')
expert = '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'


def container(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(root))


def state(stage, **details):
    status_path.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), updated=time.time(), **details), indent=2))


protocol = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, seed=args.seed,
                num_envs=args.num_envs, training_seeds=[d['seed'] for d in config['data']],
                protocol='First completed episode per environment; fresh seed; paired initial state hashes')
with (runs / (args.tag + '_protocol.json')).open('x') as stream:
    json.dump(protocol, stream, indent=2)
results = {}
try:
    for skill in ['shoot', 'pass']:
        results[skill] = {}
        for controller in ['planner', 'expert']:
            name = args.tag + '_' + skill + '_' + controller
            entry = 'probe_shoot_expert.py' if skill == 'shoot' else 'expert_baseline.py'
            command = [str(root / 'scripts/container_exec.sh'), '/workspace/fiber-soccer/scripts/' + entry,
                       '--headless', '--device', 'cuda:0', '--evaluate', '--seed', str(args.seed),
                       '--num-envs', str(args.num_envs), '--steps', '2000', '--output', container(runs / name)]
            if skill == 'pass':
                command += ['--checkpoint', expert]
            if controller == 'planner':
                command += ['--projector-checkpoint', container(checkpoint)]
            state(name, status='running', checkpoint_sha256=checkpoint_hash)
            with (runs / (name + '.log')).open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            result = json.loads((runs / name / 'summary.json').read_text())
            assert result['completed'] == args.num_envs and not result['expert_assisted_oracle']
            assert result['controller'] == ('unified_latent_planner' if controller == 'planner' else 'expert')
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
            results[skill][controller] = result
            (runs / (args.tag + '_results.json')).write_text(json.dumps(results, indent=2))
        assert results[skill]['planner']['initial_state_sha256'] == results[skill]['expert']['initial_state_sha256']
    state('completed', status='completed', protocol=protocol, results=results)
except Exception as error:
    state('failed', status='failed', error=repr(error))
    raise
