"""Check PPO source-shooting physics against a completed continuous scene run."""
import argparse
import json
from pathlib import Path


def audit(run, sequence):
    snapshots=json.loads((run/'scene_contracts.json').read_text())
    train=json.loads((run/'physical_contract.json').read_text())
    config=json.loads((run/'config.json').read_text())
    passing,shooting=snapshots['passing'],snapshots['shooting']
    reference=json.loads((sequence/'scene_contracts.json').read_text())['shared']
    protocol=json.loads((sequence/'protocol.json').read_text())
    assert config['training_scene']=='shoot'
    assert protocol['scene']=='shoot' and protocol['incoming_pass_command']
    for key in ['sim_dt','decimation','robot','actions','events','ball_dynamics']:
        assert train[key]==shooting[key]==reference[key],key
    for key,ignored in [('scene',{'num_envs'}),('sim',{'device'})]:
        values=[{k:v for k,v in c[key].items() if k not in ignored}
                for c in [train,shooting,reference]]
        assert values[0]==values[1]==values[2],key
    assert train['rewards']==passing['rewards'],'Passing reward changed'
    assert set(train['terminations'])==set(shooting['terminations'])==set(reference['terminations'])
    assert train['terminations']['event_kick_elapsed'] is None
    for key,value in shooting['terminations'].items():
        if key!='event_kick_elapsed':
            assert value==train['terminations'][key]==reference['terminations'][key],key
    reset={'pose_range','velocity_range','joint_position_range','randomize_time_on_reset',
           'event_reset_to_reference_pose','prelaunch_training_probability'}
    reset.update(key for key in shooting['command'] if key.startswith('reset_joint_'))
    assert set(train['command'])==set(passing['command'])==set(reference['command'])
    for key,value in train['command'].items():
        expected=shooting['command'][key] if key in reset else passing['command'][key]
        assert value==expected,key
        if key not in {'ball_resample_after_kick_enabled','ball_reset_after_kick_enabled'}:
            assert value==reference['command'][key],key
    expected_timeout=6*train['sim_dt']*train['decimation'] if config['smoke_test'] else passing['episode_length_s']
    assert train['episode_length_s']==expected_timeout
    return dict(training_run=str(run),sequence_reference=str(sequence),
        source_physics_and_sequence_physics_match=True,failure_terms_preserved=True,
        full_passing_command_with_source_reset=True,passing_rewards_preserved=True,
        timeout_s=expected_timeout,smoke_test=config['smoke_test'],
        scope='Configuration audit, not task success. Training retains passing ball resampling; sequence testing instead schedules a static second ball. Histories and physical states are not paired.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--sequence-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    result=audit(a.run,a.sequence_reference)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
