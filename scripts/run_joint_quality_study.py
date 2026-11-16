"""Run matched-budget joint-reward arms, native checks and three-model evaluation.

The fixed protocol compares final checkpoints only. Failures retain their original
sequence denominator; no candidate is automatically promoted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from summarize_continuous_shoot_ppo import outcomes

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = '2f23219b55fd8789cabd6be238228e74cf957ea3c4d44c769b9a2afd63d04474'
SEEDS = [815701, 815702]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def container(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(ROOT))


def check_pair(control, treatment):
    assert read(control/'initialization.json') == read(treatment/'initialization.json')
    assert read(control/'scene_contracts.json') == read(treatment/'scene_contracts.json')
    a, b = read(control/'config.json'), read(treatment/'config.json')
    assert (a['joint_impact_bonus'], a['joint_timeout_bonus']) == (0, 0)
    assert (b['joint_impact_bonus'], b['joint_timeout_bonus']) == (10, 20)
    for key in a:
        if key not in {'output', 'joint_impact_bonus', 'joint_timeout_bonus', 'resume_audit'}:
            assert a[key] == b[key], key
    for key in a['resume_audit']:
        if key not in {'joint_impact_bonus', 'joint_timeout_bonus'}:
            assert a['resume_audit'][key] == b['resume_audit'][key], key
    return dict(initialization_exact=True, scenes_exact=True,
                configurations_differ_only_in_output_and_joint_bonuses=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='planner_joint_quality_r1')
    args = parser.parse_args()
    assert args.tag.replace('_', '').isalnum()
    runs = ROOT/'runs'
    status_path = runs/f'{args.tag}_status.json'
    result_path = runs/f'{args.tag}_results.json'
    assert not status_path.exists() and not result_path.exists()
    source = runs/'planner_continuous_shoot_ppo_r2'
    base = runs/'planner_history_r3_contact_train/best.pt'
    assert sha(source/'final.pt') == SOURCE_SHA
    arms = {arm: runs/f'{args.tag}_{arm}' for arm in ['control', 'treatment']}
    assert all(not path.exists() for path in arms.values())
    # Check actual prior run records, not documentation reserving these seeds.
    for pattern in ['*/config.json', '*/summary.json', '*_status.json']:
        for path in runs.glob(pattern):
            record = read(path)
            recorded = {record.get('seed'), record.get('handoff_seed')}
            recorded.update(record.get('seeds', []))
            assert not recorded.intersection(SEEDS), f'Evaluation seed already used: {path}'
    results = dict(source_checkpoint_sha256=SOURCE_SHA, seeds=SEEDS, training={}, native={}, sequences={})

    def state(stage, **details):
        data = dict(stage=stage, pid=os.getpid(), updated=time.time(), results=results, **details)
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2)+'\n')
        temporary.replace(status_path)

    def execute(command, destination, gpu):
        with destination.open('x') as log:
            subprocess.run([str(ROOT/'scripts/container_exec.sh')]+command,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, MOS_CUDA_VISIBLE_DEVICES=str(gpu)))

    try:
        smokes = {arm: runs/f'planner_joint_quality_{arm}_smoke' for arm in arms}
        results['smoke_pair_audit'] = check_pair(smokes['control'], smokes['treatment'])
        for arm, path in smokes.items():
            done = read(path/'complete.json')
            audit = read(runs/f'{path.name}_audit.json')
            assert done['smoke_test'] and done['deployable_roundtrip']
            assert done['joint_successful_timeouts'] > 0
            assert sha(path/'final.pt') == done['checkpoint_sha256'] == audit['checkpoint_sha256']
            assert audit['joint_quality_audit']['joint_reward_sums_exact']
            assert audit['resume_step_audit']['new_critic_memory_steps_exact']
        for arm, run in arms.items():
            state('training', arm=arm)
            impact, timeout = (0, 0) if arm == 'control' else (10, 20)
            execute(['/workspace/fiber-soccer/scripts/train_continuous_shoot_ppo.py',
                     '--checkpoint', container(base), '--resume-run', container(source),
                     '--output', container(run), '--joint-quality-study',
                     '--joint-impact-bonus', str(impact), '--joint-timeout-bonus', str(timeout),
                     '--iterations', '100', '--num-envs', '256', '--rollout-steps', '32',
                     '--epochs', '4', '--minibatches', '4', '--seed', '815401',
                     '--headless', '--device', 'cuda:0'], run.with_suffix('.log'), 3)
            done = read(run/'complete.json')
            assert not done['smoke_test'] and done['iterations'] == 100
            assert done['total_iterations'] == 400 and done['environment_steps'] == 819200
            assert done['total_environment_steps'] == 3276800
            assert done['shared_tensors_unchanged'] and done['deployable_roundtrip']
            assert done['checkpoint_sha256'] == sha(run/'final.pt')
            assert done['resume_audit']['checkpoint_sha256'] == SOURCE_SHA
            state('auditing_training', arm=arm)
            audit_path = runs/f'{run.name}_audit.json'
            execute(['/workspace/fiber-soccer/scripts/audit_continuous_shoot_ppo.py',
                     '--base', container(base), '--run', container(run), '--data',
                     container(runs/'planner_recovery_r2_sum_semantic/pass_trajectory.pt'),
                     container(runs/'planner_recovery_r2_sum_semantic/shoot_trajectory.pt'),
                     '--sequence-reference', container(runs/'planner_post_pass_gate_zero_control_source'),
                     '--output', container(audit_path), '--device', 'cuda:0'], audit_path.with_suffix('.log'), 2)
            results['training'][arm] = dict(complete=done, audit=read(audit_path))
        results['training_pair_audit'] = check_pair(arms['control'], arms['treatment'])
        models = dict(reference=source/'final.pt', **{arm: run/'final.pt' for arm, run in arms.items()})
        hashes = {label: sha(model) for label, model in models.items()}
        for arm, run in arms.items():
            results['native'][arm] = {}
            for skill in ['pass', 'shoot']:
                state('evaluating_native', arm=arm, skill=skill)
                destination = runs/f'{run.name}_{skill}_eval'
                entry = 'expert_baseline.py' if skill == 'pass' else 'probe_shoot_expert.py'
                command = ['/workspace/fiber-soccer/scripts/'+entry, '--headless', '--device', 'cuda:0',
                           '--evaluate', '--seed', '811314', '--num-envs', '16', '--steps', '2000',
                           '--projector-checkpoint', container(models[arm]), '--output', container(destination)]
                if skill == 'pass':
                    command += ['--checkpoint', '/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt']
                execute(command, destination.with_suffix('.log'), 3)
                summary = read(destination/'summary.json')
                native_base = runs/f'planner_history_r3_contact_{skill}_eval'
                assert summary['completed'] == summary['requested'] == 16
                assert not summary['expert_assisted_oracle']
                assert summary['initial_state_sha256'] == read(native_base/'summary.json')['initial_state_sha256']
                for name in ['episodes.json', 'first_kick_velocity.json']:
                    assert read(destination/name) == read(native_base/name), (arm, skill, name)
                results['native'][arm][skill] = dict(summary=summary, episodes_and_first_impacts_exact=True)
        for seed in SEEDS:
            summaries, protocols, contracts = {}, {}, {}
            for label, model in models.items():
                state('evaluating_sequence', seed=seed, model=label)
                destination = runs/f'{args.tag}_{seed}_{label}'
                execute(['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
                         '--headless', '--device', 'cuda:0', '--checkpoint', container(model),
                         '--output', container(destination), '--num-envs', '32', '--steps', '1000',
                         '--seed', str(seed), '--handoff-seed', str(seed), '--scene-base', 'shoot',
                         '--action-contract', 'shoot', '--reset-contract', 'shoot', '--events-contract', 'shoot',
                         '--incoming-pass-command', '--continuous-after-kick', '--transition-steps', '0',
                         '--post-shoot-recovery-steps', '10', '--failed-lane-control', 'zero'], destination.with_suffix('.log'), 3)
                summary = read(destination/'summary.json')
                assert summary['one_policy_instance'] and not summary['expert_loaded']
                assert summary['requested'] == 32 and summary['failed_lane_control'] == 'zero'
                assert summary['checkpoint_sha256'] == hashes[label] == sha(model)
                summaries[label] = dict(summary=summary, outcomes=outcomes(summary))
                protocols[label] = read(destination/'protocol.json')
                assert protocols[label]['shoot_adapter_after_pass_only'] is True
                contracts[label] = read(destination/'scene_contracts.json')
            reference = summaries['reference']['summary']
            first = min(event['tick'] for entry in summaries.values() for event in entry['summary']['switches'])
            for label in arms:
                candidate = summaries[label]['summary']
                assert candidate['initial_state_sha256'] == reference['initial_state_sha256']
                assert protocols[label] == protocols['reference'] and contracts[label] == contracts['reference']
                for key in ['pass_impacts', 'shoot_impacts']:
                    assert {k:v for k,v in candidate[key].items() if v['tick']<=first} == {k:v for k,v in reference[key].items() if v['tick']<=first}, (label,key)
                for key in ['switches', 'first_resets', 'contact_observations']:
                    assert [v for v in candidate[key] if v['tick']<=first] == [v for v in reference[key] if v['tick']<=first], (label,key)
            results['sequences'][str(seed)] = dict(models=summaries, audit=dict(initial_tensors_equal=True,
                protocol_and_scene_equal=True, observed_prefix_equal=True, first_handoff_tick=first))
        totals = {label:{} for label in models}
        for pair in results['sequences'].values():
            for label, entry in pair['models'].items():
                for key, value in entry['outcomes'].items():
                    if isinstance(value, int):
                        totals[label][key] = totals[label].get(key, 0)+value
        results['totals'] = totals
        results['checkpoint_hashes'] = hashes
        assert all(sha(path) == hashes[label] for label, path in models.items())
        results['scope'] = 'Fixed final checkpoints, matched 100-round arms from 300-round source; no automatic promotion.'
        result_path.write_text(json.dumps(results, indent=2)+'\n')
        state('completed', result_path=str(result_path))
    except Exception as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
