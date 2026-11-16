import pytest
from fiber_soccer.continuous_resume import (
    CONTINUOUS_MATCHED_OPTIONS, CONTINUOUS_OBJECTIVE, validate_continuous_resume,
)


def manifests():
    current={key:1 for key in CONTINUOUS_MATCHED_OPTIONS}
    current.update(seed=815401,num_envs=256,rollout_steps=32,smoke_test=False)
    config=dict(current,**CONTINUOUS_OBJECTIVE,parent_sha256='r3')
    complete=dict(iterations=100,environment_steps=819200,smoke_test=False,
        shared_tensors_unchanged=True,deployable_roundtrip=True,shooting_entries=20,
        active_policy_steps=1000,handoff_history_checks=20)
    return config,complete,current


def test_continuation_counts_all_prior_iterations_without_restoring_simulator():
    config,complete,current=manifests()
    assert validate_continuous_resume(config,complete,current,'r3')==dict(iterations=100,environment_steps=819200)
    complete.update(total_iterations=300,total_environment_steps=2457600)
    assert validate_continuous_resume(config,complete,current,'r3')['iterations']==300
    complete['total_environment_steps']-=1
    with pytest.raises(ValueError,match='accounting'):validate_continuous_resume(config,complete,current,'r3')


@pytest.mark.parametrize('key',['gamma','pass_delay','timeout_bootstrap','training'])
def test_continuation_rejects_a_changed_objective(key):
    config,complete,current=manifests();config[key]='changed'
    with pytest.raises(ValueError,match='objective'):validate_continuous_resume(config,complete,current,'r3')


@pytest.mark.parametrize('key',['learning_rate','exploration_std','num_envs','seed','smoke_test'])
def test_continuation_cannot_silently_change_experiment_options(key):
    config,complete,current=manifests();current[key]='changed'
    with pytest.raises(ValueError,match=key):validate_continuous_resume(config,complete,current,'r3')


def test_continuation_requires_verified_source_and_the_same_frozen_base():
    config,complete,current=manifests()
    with pytest.raises(ValueError,match='base'):validate_continuous_resume(config,complete,current,'other')
    complete['handoff_history_checks']=19
    with pytest.raises(ValueError,match='handoff'):validate_continuous_resume(config,complete,current,'r3')
