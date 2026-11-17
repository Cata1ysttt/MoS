import pytest
import torch
from fiber_soccer.refinement_scope import RefinementScope


def test_initial_pass_disables_refinement_through_shoot_and_recovery():
    gate=RefinementScope('initial_pass')
    assert torch.equal(gate(torch.tensor([0,1,0])),torch.tensor([True,False,True]))
    assert torch.equal(gate(torch.tensor([1,0,0])),torch.tensor([False,False,True]))
    assert torch.equal(gate(torch.tensor([0,0,0])),torch.tensor([False,False,True]))
    gate.reset(torch.tensor([False,True,False]))
    assert torch.equal(gate(torch.tensor([0,0,0])),torch.tensor([False,True,True]))
    gate.reset();assert gate(torch.zeros(2,dtype=torch.long)).all()
    assert not gate.state_dict(),'Episode memory must not be serialized as model parameters'


def test_default_scope_and_invalid_batch():
    gate=RefinementScope()
    assert gate(torch.tensor([0,1])).all()
    assert gate(torch.tensor([1])).all()
    with pytest.raises(ValueError):RefinementScope('unknown')
    with pytest.raises(ValueError):gate(torch.tensor([2]))
    scoped=RefinementScope('initial_pass');scoped(torch.tensor([0,1]))
    with pytest.raises(ValueError):scoped(torch.tensor([0]))


def test_post_pass_scope_preserves_native_shoot_and_stops_at_contact():
    gate=RefinementScope('initial_pass_and_post_pass_shoot')
    # A native shooting lane, a passing lane, and a pass without contact.
    skills=torch.tensor([1,0,0]);contact=torch.tensor([False,False,False])
    with torch.inference_mode():
        assert gate(skills,contact).tolist()==[False,True,True]
        contact[1]=True
        assert gate(skills,contact).tolist()==[False,True,True]
        skills[:]=1
        # Carrying the old ball contact must not activate shooting refinement.
        assert gate(skills,contact).tolist()==[False,False,False]
        contact[:]=False
        assert gate(skills,contact).tolist()==[False,True,False]
        contact[:]=True
        assert not gate(skills,contact).any()
        skills[:]=0
        assert not gate(skills,contact).any()  # Shooting recovery is unchanged.
    gate.reset(torch.tensor([False,True,False]))
    contact[:]=False
    assert gate(skills,contact).tolist()==[False,True,False]
    skills[:]=1
    assert not gate(skills,contact).any()  # Reset clears the completed pass.
    assert not gate.state_dict()
    with pytest.raises(ValueError,match='contact'):gate(skills)
    with pytest.raises(ValueError,match='contact'):gate(skills,torch.zeros(3))
    gate.reset()
    assert not gate(torch.ones(1,dtype=torch.long),torch.zeros(1,dtype=torch.bool)).any()
