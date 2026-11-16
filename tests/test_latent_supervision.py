import copy
import pytest
import torch
from fiber_soccer.latent_supervision import confidence,distillation_loss,tensor_sha,validate_labels


def sample():
    x=torch.arange(12,dtype=torch.float32).reshape(3,4)
    contact=torch.tensor([False,True,False])
    metadata=dict(purpose='train_inverse_from_predicted_targets_v1',
        split='first_three_quarters_of_each_source_environment_set',expert_targets_used=False,validation_states_used=False)
    z=torch.zeros(3,256);z[:,0]=16
    record=dict(metadata=metadata,skill='pass',x_sha256=tensor_sha(x),contact=contact,z=z,
        desired_targets=torch.zeros(3,29),initial_mapping_mse=torch.ones(3),final_mapping_mse=torch.zeros(3))
    return record,x,contact,metadata


def test_valid_and_reordered_states():
    record,x,contact,metadata=sample()
    validate_labels(record,x,contact,'pass',metadata)
    with pytest.raises(AssertionError):validate_labels(record,x.flip(0),contact,'pass',metadata)


@pytest.mark.parametrize('corruption',['split','validation','expert','skill','contact','norm','nan','negative'])
def test_reject_bad_labels(corruption):
    record,x,contact,metadata=sample();record=copy.deepcopy(record)
    if corruption=='split':record['metadata']['split']='held_out'
    if corruption=='validation':record['metadata']['validation_states_used']=True
    if corruption=='expert':record['metadata']['expert_targets_used']=True
    if corruption=='skill':record['skill']='shoot'
    if corruption=='contact':record['contact'][0]=True
    if corruption=='norm':record['z'][0]*=2
    if corruption=='nan':record['desired_targets'][0,0]=float('nan')
    if corruption=='negative':record['final_mapping_mse'][0]=-1
    with pytest.raises(AssertionError):validate_labels(record,x,contact,'pass',metadata)


def test_quality_weight_and_equal_skill_loss():
    residual=torch.tensor([0.,.001,.1])
    assert torch.all(confidence(residual)[:-1]>confidence(residual)[1:])
    with pytest.raises(ValueError):confidence(torch.tensor([-1.]))
    latent=torch.tensor([1.,3.,5.,7.],requires_grad=True)
    physical=torch.zeros(4)
    weight=torch.tensor([1.,1.,.01,.01])
    loss=distillation_loss(latent,physical,weight,2)
    torch.testing.assert_close(loss,torch.tensor(4.))
    loss.backward()
    torch.testing.assert_close(latent.grad,torch.full((4,),.25))
    # A poorly solved outlier has a smaller contribution within its skill.
    assert distillation_loss(torch.tensor([1.,100.]),torch.zeros(2),torch.tensor([1.,.001]),2)<1.1
