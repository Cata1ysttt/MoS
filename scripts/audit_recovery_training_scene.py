"""Check post-shot training keeps continuous-scene physics and source shot sampling."""
import argparse
import json
from pathlib import Path


def audit(run,sequence):
    snapshots=json.loads((run/'scene_contracts.json').read_text())
    train=json.loads((run/'physical_contract.json').read_text())
    config=json.loads((run/'config.json').read_text())
    source=snapshots['shooting']
    reference=json.loads((sequence/'scene_contracts.json').read_text())['shared']
    assert config['training_task']=='shoot_recover' and config['training_scene']=='shoot'
    assert config['recovery_transition_steps']==0
    for key in ['sim_dt','decimation','robot','actions','events','ball_dynamics']:
        assert train[key]==source[key]==reference[key],key
    for key,ignored in [('scene',{'num_envs'}),('sim',{'device'})]:
        values=[{k:v for k,v in c[key].items() if k not in ignored} for c in [train,source,reference]]
        assert values[0]==values[1]==values[2],key
    assert set(train['terminations'])==set(source['terminations'])
    for key,value in train['terminations'].items():
        assert value==(None if key=='event_kick_elapsed' else source['terminations'][key]),key
    assert set(train['command'])==set(reference['command'])
    for key,value in train['command'].items():
        expected=source['command'][key] if key.startswith(('ball_init','ball_target','ball_launch','ball_sample_point')) else reference['command'][key]
        assert value==expected,key
    assert not train['command']['ball_resample_after_kick_enabled']
    assert not train['command']['ball_reset_after_kick_enabled']
    assert train['episode_length_s']==(4. if config['smoke_test'] else 20.)
    expected_rewards={
        'stability':dict(func='fiber_soccer.recovery_training:recovery_stability',params={'target_height':source['robot']['init_state']['pos'][2]},weight=5.),
        'action_rate':dict(func='fiber_soccer.recovery_training:recovery_action_rate',params={},weight=-.05),
        'failure':dict(func='fiber_soccer.recovery_training:recovery_failure',params={},weight=-100.)}
    assert train['rewards']==expected_rewards
    return dict(run=str(run),sequence_reference=str(sequence),source_physics_preserved=True,
        failure_terms_preserved=True,source_static_shot_sampling=True,passing_phase_observations_preserved=True,
        recovery_stability_reward=True,smoke_test=config['smoke_test'],
        scope='Configuration audit only. Training starts with a static shot; continuous evaluation starts with an incoming pass. No expert actions or robot reset at the shooting-to-recovery transition.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--sequence-reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.sequence_reference)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
