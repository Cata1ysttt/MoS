"""Evaluate frozen r3/control/joint checkpoints on four preselected fresh seeds.

No training or checkpoint selection. All 32 first episodes per seed remain in the
score denominator, including failures. This does not automatically promote a model.
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
SEEDS = [815801, 815802, 815803, 815804]
EXPECTED = {
    'r3': '5cdc03e0d5d5c44a20626d03c4a4088fab0d6f2ab5b229b66466c18bdade2133',
    'control': '09eaaf69670241624a5dede285532125d507a649f833245c487fe424f0370f9d',
    'treatment': 'e590d391cbdd806958c8852299b0561d1b5637c35832b2e26a0c437ab2e073f5',
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def container(path):
    return '/workspace/fiber-soccer/' + str(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='planner_joint_quality_confirmation_r1')
    args = parser.parse_args()
    assert args.tag.replace('_', '').isalnum()
    runs = ROOT/'runs'
    status = runs/f'{args.tag}_status.json'
    result_path = runs/f'{args.tag}_results.json'
    protocol_path = runs/f'{args.tag}_protocol.json'
    assert all(not path.exists() for path in [status, result_path, protocol_path])
    assert read(runs/'planner_joint_quality_r1_status.json')['stage'] == 'completed'
    models = dict(r3=runs/'planner_history_r3_contact_train/best.pt',
                  control=runs/'planner_joint_quality_r1_control/final.pt',
                  treatment=runs/'planner_joint_quality_r1_treatment/final.pt')
    assert {k: sha(v) for k, v in models.items()} == EXPECTED
    for arm in ['control', 'treatment']:
        audit = read(runs/f'planner_joint_quality_r1_{arm}_audit.json')
        assert audit['checkpoint_sha256'] == EXPECTED[arm]
        assert audit['shared_tensors_exact'] and audit['standalone_outputs_match_base']
    for pattern in ['*/config.json', '*/summary.json', '*_status.json']:
        for path in runs.glob(pattern):
            data = read(path)
            used = {data.get('seed'), data.get('handoff_seed')}
            used.update(data.get('seeds', []))
            assert not used.intersection(SEEDS), f'Previously used seed: {path}'
    protocol = dict(seeds=SEEDS, models={k:str(v) for k,v in models.items()},
        checkpoint_sha256=EXPECTED, driver_sha256=sha(Path(__file__)),
        num_envs=32, steps=1000, failed_lane_control='zero', training=False,
        purpose='Confirm frozen candidates against default r3 on four additional fresh seeds; no automatic promotion.')
    protocol_path.write_text(json.dumps(protocol, indent=2)+'\n')
    results = dict(protocol=protocol, sequences={})

    def state(stage, **details):
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(stage=stage, pid=os.getpid(), updated=time.time(),
                                            seeds=SEEDS, results=results, **details), indent=2)+'\n')
        temporary.replace(status)

    try:
        for seed in SEEDS:
            pair, protocols, contracts = {}, {}, {}
            for label, checkpoint in models.items():
                state('evaluating_sequence', seed=seed, model=label)
                destination = runs/f'{args.tag}_{seed}_{label}'
                command = ['/workspace/fiber-soccer/scripts/probe_soccer_sequence.py',
                    '--headless', '--device', 'cuda:0', '--checkpoint', container(checkpoint),
                    '--output', container(destination), '--num-envs', '32', '--steps', '1000',
                    '--seed', str(seed), '--handoff-seed', str(seed), '--scene-base', 'shoot',
                    '--action-contract', 'shoot', '--reset-contract', 'shoot', '--events-contract', 'shoot',
                    '--incoming-pass-command', '--continuous-after-kick', '--transition-steps', '0',
                    '--post-shoot-recovery-steps', '10', '--failed-lane-control', 'zero']
                with destination.with_suffix('.log').open('x') as log:
                    subprocess.run([str(ROOT/'scripts/container_exec.sh')]+command,
                        stdout=log, stderr=subprocess.STDOUT, check=True,
                        env=dict(os.environ, MOS_CUDA_VISIBLE_DEVICES='3'))
                summary = read(destination/'summary.json')
                assert summary['checkpoint_sha256'] == EXPECTED[label] == sha(checkpoint)
                assert summary['one_policy_instance'] and not summary['expert_loaded']
                assert summary['requested'] == 32 and summary['failed_lane_control'] == 'zero'
                pair[label] = dict(summary=summary, outcomes=outcomes(summary))
                protocols[label] = read(destination/'protocol.json')
                gate = protocols[label].pop('shoot_adapter_after_pass_only', False)
                assert gate == (label != 'r3')
                contracts[label] = read(destination/'scene_contracts.json')
            reference = pair['r3']['summary']
            first = min((event['tick'] for entry in pair.values() for event in entry['summary']['switches']), default=1000)
            for label in ['control', 'treatment']:
                candidate = pair[label]['summary']
                assert candidate['initial_state_sha256'] == reference['initial_state_sha256']
                assert protocols[label] == protocols['r3'] and contracts[label] == contracts['r3']
                for key in ['pass_impacts', 'shoot_impacts']:
                    assert {k:v for k,v in candidate[key].items() if v['tick']<=first} == {k:v for k,v in reference[key].items() if v['tick']<=first}, (label,key)
                for key in ['switches', 'first_resets', 'contact_observations']:
                    assert [v for v in candidate[key] if v['tick']<=first] == [v for v in reference[key] if v['tick']<=first], (label,key)
            results['sequences'][str(seed)] = dict(models=pair, audit=dict(initial_tensors_equal=True,
                protocol_and_scene_equal_except_declared_gate=True, observed_prefix_equal=True, first_handoff_tick=first))
        totals = {label:{} for label in models}
        for entry in results['sequences'].values():
            for label, record in entry['models'].items():
                for key,value in record['outcomes'].items():
                    if isinstance(value,int):totals[label][key] = totals[label].get(key,0)+value
                failure = totals[label].setdefault('failures', {})
                for key,value in record['outcomes']['failures'].items():failure[key] = failure.get(key,0)+value
        results['totals'] = totals
        assert {k:sha(v) for k,v in models.items()} == EXPECTED
        result_path.write_text(json.dumps(results, indent=2)+'\n')
        state('completed', result_path=str(result_path))
    except Exception as error:
        state('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
