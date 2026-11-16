import copy
import pytest
import torch
from fiber_soccer.sequence_adaptation import split_sequence_labels,EnvironmentSampler


def fixture():
    x=torch.zeros(4,2406);x[:,-1]=1
    return dict(metadata=dict(purpose='training',schema='soccer_state_v4_task_history_2406',
        format='flat_sequence_shoot_states_v1',seed=815201,num_envs=4,checkpoint_sha256='base',expert_sha256='expert',
        expert_loaded=False,expert_actions_executed=False,offline_expert_loaded=True,
        physical_target_mapping='raw_action_clamp_10_then_affine_no_physical_clip',raw_action_clip=10,joint_order=list(range(29))),
        x=x,environment=torch.arange(4),target=torch.zeros(4,29),expert_raw_action=torch.zeros(4,29),
        action_scale=torch.ones(4,29),action_offset=torch.zeros(4,29))


def test_training_split_rejects_diagnostic_or_uncorrected_labels_and_preserves_lane_boundary():
    data=fixture();train,valid=split_sequence_labels(data,'base','expert')
    assert train.tolist()==[0,1,2] and valid.tolist()==[3]
    for key,value,match in [('purpose','diagnostic','Diagnostic'),('seed',815101,'Reserved'),
            ('physical_target_mapping','raw_action_affine_no_clip','raw-action clamp')]:
        bad=copy.deepcopy(data);bad['metadata'][key]=value
        with pytest.raises(ValueError,match=match):split_sequence_labels(bad,'base','expert')
    bad=copy.deepcopy(data);bad['target'][0,0]=1
    with pytest.raises(ValueError,match='action contract'):split_sequence_labels(bad,'base','expert')


def test_environment_balancing_includes_short_failure_lanes_without_validation_leakage():
    env=torch.tensor([0]*100+[1]+[2]*20+[3]*100)
    sampler=EnvironmentSampler(env,(env<3).nonzero().flatten())
    ids=sampler.sample(3000,torch.Generator().manual_seed(987))
    counts=torch.bincount(env[ids],minlength=4)
    assert counts[3]==0
    assert (counts[:3]>800).all() and (counts[:3]<1200).all()
