"""Collect current flow-policy states, relabel, and evaluate one shared checkpoint."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root / 'runs'
status = runs / 'planner_flow_dagger_r1_status.json'
prefix = [str(root / 'scripts/container_exec.sh')]
expert = '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'


def c(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(root))


def state(stage, **more):
    status.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), updated=time.time(), **more), indent=2))


def execute(name, args):
    state(name, status='running')
    with (runs / (name + '.log')).open('x') as stream:
        subprocess.run(prefix + args, stdout=stream, stderr=subprocess.STDOUT, check=True)


try:
    student = runs / 'planner_flow_action_train/best.pt'
    assert student.exists()
    collections = []
    for skill, seed in [('shoot', 812201), ('pass', 812202)]:
        name = 'planner_flow_dagger_r1_' + skill + '_collect'
        entry = 'probe_shoot_expert.py' if skill == 'shoot' else 'expert_baseline.py'
        source = 'joint_v2_r2_pass_train' if skill == 'shoot' else 'passing_50hz_long_s1'
        args = ['/workspace/fiber-soccer/scripts/' + entry, '--headless', '--device', 'cuda:0',
                '--planner-collect', '--planner-observation-schema', 'soccer_state_v2_950',
                '--projector-checkpoint', c(runs / source / 'latest.pt'),
                '--planner-student', c(student), '--planner-teacher-ratio', '.25',
                '--oracle-latent-steps', '64', '--seed', str(seed), '--num-envs', '64',
                '--steps', '1000', '--output', c(runs / name)]
        if skill == 'pass':
            args += ['--checkpoint', expert]
        execute(name, args)
        assert (runs / name / 'complete.json').exists()
        collections.append(runs / name / 'trajectory.pt')
    semantic = runs / 'planner_flow_dagger_r1_semantic'
    execute(semantic.name, ['/workspace/fiber-soccer/scripts/prepare_semantic_planner_data.py',
                            '--data', *map(c, collections), '--output', c(semantic)])
    assert (semantic / 'complete.json').exists()
    data = [folder / (skill + '_trajectory.pt')
            for folder in [runs / 'planner_semantic_wide_v1', semantic]
            for skill in ['pass', 'shoot']]
    out = runs / 'planner_flow_dagger_r1_train'
    execute(out.name, ['/workspace/fiber-soccer/scripts/train_flow_planner.py',
                       '--data', *map(c, data), '--output', c(out), '--updates', '20000',
                       '--resume', c(student), '--action-weight', '100', '--leg-weight', '2'])
    assert (out / 'complete.json').exists()
    results = {}
    for skill in ['shoot', 'pass']:
        name = 'planner_flow_dagger_r1_' + skill + '_eval'
        entry = 'probe_shoot_expert.py' if skill == 'shoot' else 'expert_baseline.py'
        args = ['/workspace/fiber-soccer/scripts/' + entry, '--headless', '--device', 'cuda:0',
                '--evaluate', '--projector-checkpoint', c(out / 'best.pt'), '--seed', '811314',
                '--num-envs', '16', '--steps', '2000', '--output', c(runs / name)]
        if skill == 'pass':
            args += ['--checkpoint', expert]
        execute(name, args)
        result = json.loads((runs / name / 'summary.json').read_text())
        reference = json.loads((runs / f'pass_shoot_{skill}_expert_eval_v1/summary.json').read_text())
        assert result['completed'] == 16 and not result['expert_assisted_oracle']
        assert result['initial_state_sha256'] == reference['initial_state_sha256']
        results[skill] = result
        (runs / 'planner_flow_dagger_r1_results.json').write_text(json.dumps(results, indent=2))
    state('completed', status='completed', results=results)
except Exception as error:
    state('failed', status='failed', error=repr(error))
    raise
