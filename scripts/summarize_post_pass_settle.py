"""Audit and summarize the fixed two-seed post-pass settling comparison."""
import hashlib
import json
from pathlib import Path
from statistics import median
from summarize_continuous_shoot_ppo import outcomes

ROOT = Path(__file__).resolve().parents[1]
TAG = 'planner_post_pass_settle_r1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    return dict(count=len(values), median=median(values), minimum=min(values), maximum=max(values)) if values else dict(count=0)


def main():
    state = read(ROOT / 'runs' / f'{TAG}_status.json')
    assert state['stage'] == 'completed', 'Wait for the fixed experiment to complete'
    results = state['results']; protocol = results['protocol']
    assert results['native_screen_passed']
    assert all(sha(ROOT / name) == digest for name, digest in protocol['runtime_source_sha256'].items())
    assert sha(ROOT / 'AH-BFM/AH-BFM.pt') == protocol['actor_sha256']
    compact = dict(stage='completed', protocol=protocol, native=results['native'], sequences={},
                   totals=results['totals'], development_screen_passed=results['development_screen_passed'],
                   frozen_hashes_verified=True, automatic_promotion=False)
    groups = {label: [] for label in ['control', 'candidate']}
    for seed in protocol['seeds']:
        pair = results['sequences'][str(seed)]
        assert all(pair['audit'].values())
        entry = dict(audit=pair['audit'], models={})
        for label, record in pair['models'].items():
            summary = read(ROOT / 'runs' / f'{TAG}_{seed}_{label}' / 'summary.json')
            assert summary == record['summary'] and outcomes(summary) == record['outcomes']
            checkpoint = ROOT / str(summary['checkpoint']).removeprefix('/workspace/fiber-soccer/')
            assert sha(checkpoint) == protocol['checkpoint_sha256'][label]
            groups[label].append(summary)
            entry['models'][label] = dict(outcomes=record['outcomes'], execution_audits=record['execution_audits'])
        compact['sequences'][str(seed)] = entry
    analysis = dict(scope='Descriptive metrics on all observed impacts and handoffs in the fixed development comparison. Overlapping quality failures are not summed. After the first intervention, global random streams can diverge; per-environment changes are not causal estimates.', models={})
    for label, summaries in groups.items():
        shots = [row for s in summaries for row in s['shoot_impacts'].values()]
        slow = [row for row in shots if row['forward_speed_mps'] < .7 * row['target_speed_mps']]
        motion = {key: [value for s in summaries for event in s['switches'] for value in event[key]]
                  for key in ['root_speed_xy', 'root_angular_speed', 'root_tilt_rad']}
        shot_stats = {key: distribution([row[key] for row in shots])
                      for key in ['speed_mps', 'forward_speed_mps', 'direction_error_deg', 'target_speed_mps']}
        # Exhaustive disjoint buckets, unlike the overlapping headline errors.
        buckets = {}
        for row in shots:
            failures = []
            if not row['preferred_foot']: failures.append('foot')
            if row['direction_error_deg'] > 10: failures.append('direction')
            if row['relative_forward_speed_error'] > .3: failures.append('speed')
            key = '+'.join(failures) or 'precise'
            buckets[key] = buckets.get(key, 0) + 1
        assert sum(buckets.values()) == len(shots)
        analysis['models'][label] = dict(shot_metrics=shot_stats, disjoint_shot_quality=buckets,
            slow_shots=len(slow), slow_also_low_speed_magnitude=sum(row['speed_mps'] < .7 * row['target_speed_mps'] for row in slow),
            handoff_motion={key: distribution(value) for key, value in motion.items()})
    for name, payload in [('results', compact), ('analysis', analysis)]:
        destination = ROOT / 'reports' / f'planner_post_pass_settle_{name}.json'
        assert not destination.exists()
        destination.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(totals=compact['totals'], development_screen_passed=compact['development_screen_passed'], analysis=analysis), indent=2))


if __name__ == '__main__':
    main()
