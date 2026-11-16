"""Verify an incoming-ball sequence keeps the source shooting physical contract."""
import argparse
import json
from pathlib import Path

from audit_sequence_recovery import validate


def audit(run, pass_reference):
    contracts = json.loads((run / 'scene_contracts.json').read_text())
    shared, source = contracts['shared'], contracts['original_shoot']
    passing = json.loads((pass_reference / 'scene_contracts.json').read_text())['shared']
    protocol = json.loads((run / 'protocol.json').read_text())
    result = json.loads((run / 'summary.json').read_text())
    assert protocol['seed'] == result['seed']
    assert protocol['num_envs'] == result['requested']
    assert protocol['steps'] == result['steps']
    assert protocol['incoming_pass_command'] and protocol['continuous_after_kick']
    assert result['incoming_pass_command'] and result['continuous_after_kick']
    assert protocol['scene'] == result['scene_base'] == 'shoot'
    assert protocol['initial_skill'] == result['initial_skill'] == 'pass'
    assert result['one_policy_instance'] and not result['expert_loaded']
    assert result['shoot_controller'] == 'unified_planner'
    for field in ['sim_dt', 'decimation', 'robot', 'actions', 'events', 'ball_dynamics']:
        assert shared[field] == source[field], f'Changed source physical contract: {field}'
    for field, ignored in [('scene', {'num_envs'}), ('sim', {'device'})]:
        assert {k:v for k,v in shared[field].items() if k not in ignored} == {k:v for k,v in source[field].items() if k not in ignored}, field
    assert shared['terminations']['event_kick_elapsed'] is None
    assert protocol['removed_normal_termination'] == source['terminations']['event_kick_elapsed']
    for key in set(shared['terminations']) | set(source['terminations']):
        if key != 'event_kick_elapsed':
            assert shared['terminations'][key] == source['terminations'][key], key
    duration = result['steps'] * shared['sim_dt'] * shared['decimation']
    assert shared['episode_length_s'] == protocol['episode_length_s'] > duration
    assert protocol['original_episode_length_s'] == source['episode_length_s']
    # These fields intentionally keep the shooting initial pose/prelaunch reset.
    reset_fields = {'pose_range', 'velocity_range', 'joint_position_range',
                    'randomize_time_on_reset', 'event_reset_to_reference_pose',
                    'prelaunch_training_probability'}
    reset_fields.update(k for k in source['command'] if k.startswith('reset_joint_'))
    assert set(shared['command']) == set(passing['command'])
    changed = []
    for key, value in shared['command'].items():
        expected = source['command'][key] if key in reset_fields else passing['command'][key]
        assert value == expected, f'Unexpected command change: {key}'
        if value != passing['command'][key]:
            changed.append(key)
    return dict(run=str(run), pass_reference=str(pass_reference),
                source_physical_contract_preserved=True, failure_terms_preserved=True,
                passing_command_preserved_except_source_reset_fields=sorted(changed),
                source_timeout_s=source['episode_length_s'], sequence_timeout_s=shared['episode_length_s'],
                accounting=validate(result),
                scope='Configuration and outcome-accounting audit only; different scene settings are not paired physical initial states or evidence of task success.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--pass-reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(a.run, a.pass_reference)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
