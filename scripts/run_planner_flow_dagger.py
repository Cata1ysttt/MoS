"""Collect current flow-policy states, relabel, and evaluate one shared checkpoint."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root / 'runs'
status = runs / 'planner_flow_dagger_r1_status.json'
prefix = [str(root / 'scripts/container_exec.sh')]
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Reuse completed stages; preserve failed attempt logs')
parser.add_argument('--label-error-scale', type=float, default=0., help='Optional confidence-weighted ablation on the same collected data')
parser.add_argument('--contact-sample-fraction', type=float, default=0., help='Training-only contact-window sample fraction')
parser.add_argument('--pass-approach-sample-fraction', type=float, default=0., help='Training-only passing approach sampling outside contact windows')
parser.add_argument('--action-target', choices=['decoded_label','expert'], default='decoded_label')
parser.add_argument('--flow-weight', type=float, default=1., help='Flow objective weight; zero keeps only decoded-action alignment')
parser.add_argument('--train-resume', type=Path, help='Training initializer, independent of the already-collected student data')
parser.add_argument('--run-name', help='Explicit unique experiment name for another continuation')
parser.add_argument('--extra-semantic', type=Path, nargs='*', default=[], help='Additional completed recovery datasets, preserving the original replay data')
parser.add_argument('--extra-trajectory', type=Path, nargs='*', default=[], help='Individual completed semantic trajectories for task-specific replay ablations')
parser.add_argument('--only-extra-semantic', action='store_true', help='Controlled training on the explicitly provided datasets only')
parser.add_argument('--skill-adapter',choices=['pass','shoot'],help='Freeze shared flow and learn only a skill-gated endpoint adapter')
parser.add_argument('--adapter-space',choices=['endpoint','latent'],default='endpoint')
parser.add_argument('--adapter-width',type=int,default=128)
parser.add_argument('--adapter-observation-input',action='store_true',help='Give the skill adapter direct access to normalized state and task history')
parser.add_argument('--architecture', choices=['flow','direct'], default='flow')
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--training-gpu', type=int, help='Physical GPU override for native training only; simulation keeps the existing container default')
parser.add_argument('--task-history', action='store_true', help='Train the v4 observation-history extension')
parser.add_argument('--wait-for-recovery', help='Wait for this collect_planner_recovery run tag before training')
parser.add_argument('--wait-for-baseline', action='store_true', help='Queue the weighted ablation after baseline training and evaluation')
options = parser.parse_args()
if options.adapter_space=='latent' and (not options.skill_adapter or options.flow_weight!=0):
    parser.error('Latent-output adapters require a skill adapter and --flow-weight 0')
if not math.isfinite(options.label_error_scale) or options.label_error_scale < 0:
    parser.error('Label error scale must be finite and nonnegative')
if not math.isfinite(options.contact_sample_fraction) or not 0 <= options.contact_sample_fraction <= .75:
    parser.error('Contact sample fraction must be in [0, .75]')
run_name = 'planner_flow_dagger_r1' if options.label_error_scale == 0 else 'planner_flow_dagger_r1_weighted'
if options.contact_sample_fraction > 0:
    run_name += '_contact'
if not math.isfinite(options.pass_approach_sample_fraction) or not 0 <= options.pass_approach_sample_fraction <= .75-options.contact_sample_fraction:
    parser.error('Approach and contact fractions must be nonnegative and total at most .75')
if not math.isfinite(options.flow_weight) or options.flow_weight < 0:
    parser.error('Flow weight must be finite and nonnegative')
if options.run_name:
    if not options.run_name.replace('_','').isalnum():
        parser.error('Run name must contain only letters, digits and underscores')
    run_name = options.run_name
elif options.adapter_space!='endpoint' or options.action_target!='decoded_label' or options.pass_approach_sample_fraction or options.train_resume or options.flow_weight != 1 or options.skill_adapter or options.adapter_observation_input or options.adapter_width!=128 or options.extra_semantic or options.extra_trajectory or options.task_history or options.only_extra_semantic or options.architecture!='flow' or options.horizon!=5:
    parser.error('Changed continuation settings require an explicit --run-name')
status = runs / (run_name + '_status.json')
lock = (runs / (run_name + '.lock')).open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
expert = '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'


def c(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(root))


def state(stage, **more):
    status.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), updated=time.time(), **more), indent=2))


def execute(name, args):
    output = runs / name
    marker = output / ('summary.json' if '--evaluate' in args else 'complete.json')
    if options.resume and marker.exists():
        json.loads(marker.read_text())
        state(name, status='reused_completed')
        return
    if output.exists():
        raise RuntimeError(f'Incomplete output requires explicit recovery: {output}')
    log = runs / (name + '.log')
    if options.resume:
        attempt = 1
        while log.exists():
            log = runs / f'{name}.attempt{attempt}.log'
            attempt += 1
    state(name, status='running', log=str(log))
    environment = os.environ.copy()
    if options.training_gpu is not None and Path(args[0]).name == 'train_flow_planner.py':
        environment['MOS_CUDA_VISIBLE_DEVICES'] = str(options.training_gpu)
    with log.open('x') as stream:
        subprocess.run(prefix + args, stdout=stream, stderr=subprocess.STDOUT, check=True, env=environment)


try:
    if options.wait_for_recovery:
        if not options.wait_for_recovery.replace('_','').isalnum():
            raise ValueError('Invalid recovery tag')
        while True:
            recovery = json.loads((runs / (options.wait_for_recovery + '_status.json')).read_text())
            if recovery['status'] == 'completed':
                break
            if recovery['status'] == 'failed':
                raise RuntimeError('Recovery collection failed: ' + recovery.get('error', 'unknown'))
            command = Path(f"/proc/{recovery['pid']}/cmdline")
            if not command.exists() or b'collect_planner_recovery.py' not in command.read_bytes():
                raise RuntimeError('Recovery collector is no longer live')
            state('waiting_for_recovery', status='running', collector_pid=recovery['pid'])
            time.sleep(30)
    if options.wait_for_baseline:
        if options.label_error_scale == 0:
            raise ValueError('Only a weighted ablation can wait for the baseline')
        while True:
            baseline = json.loads((runs / 'planner_flow_dagger_r1_status.json').read_text())
            if baseline['status'] == 'completed':
                break
            if baseline['status'] == 'failed':
                raise RuntimeError('Baseline failed: ' + baseline.get('error', 'unknown error'))
            command = Path(f"/proc/{baseline['pid']}/cmdline")
            if not command.exists() or b'run_planner_flow_dagger.py' not in command.read_bytes():
                raise RuntimeError('Baseline driver is no longer live')
            state('waiting_for_baseline', status='running', baseline_pid=baseline['pid'])
            time.sleep(30)
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
    extra = [folder.resolve() for folder in options.extra_semantic]
    for folder in extra:
        assert (folder / 'complete.json').exists(), 'Recovery labeling must finish before training'
    individual = [path.resolve() for path in options.extra_trajectory]
    for path in individual:
        assert path.is_file() and (path.parent/'complete.json').exists(), 'Individual trajectory labeling must finish before training'
    if options.only_extra_semantic and not (extra or individual):
        raise ValueError('Dataset-only control requires explicit extra datasets')
    folders = extra if options.only_extra_semantic else [runs / 'planner_semantic_wide_v1', semantic, *extra]
    data = [folder / (skill + '_trajectory.pt')
            for folder in folders
            for skill in ['pass', 'shoot']] + individual
    assert len(set(data)) == len(data), 'Duplicate semantic trajectories would silently change replay weights'
    initializer = options.train_resume.resolve() if options.train_resume else student
    assert initializer.exists()
    out = runs / (run_name + '_train')
    execute(out.name, ['/workspace/fiber-soccer/scripts/train_flow_planner.py',
                       '--data', *map(c, data), '--output', c(out), '--updates', '20000',
                       '--resume', c(initializer), '--architecture', options.architecture, '--horizon', str(options.horizon), '--flow-weight', str(options.flow_weight), '--action-weight', '100', '--leg-weight', '2', '--action-target', options.action_target,
                       '--label-error-scale', str(options.label_error_scale),
                       '--contact-sample-fraction', str(options.contact_sample_fraction),
                       '--pass-approach-sample-fraction', str(options.pass_approach_sample_fraction)] + (['--task-history'] if options.task_history else []) + (['--skill-adapter',options.skill_adapter,'--adapter-width',str(options.adapter_width),'--adapter-space',options.adapter_space] if options.skill_adapter else []) + (['--adapter-observation-input'] if options.adapter_observation_input else []))
    assert (out / 'complete.json').exists()
    results = {}
    for skill in ['shoot', 'pass']:
        name = run_name + '_' + skill + '_eval'
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
        (runs / (run_name + '_results.json')).write_text(json.dumps(results, indent=2))
    state('completed', status='completed', results=results)
except Exception as error:
    state('failed', status='failed', error=repr(error))
    raise
