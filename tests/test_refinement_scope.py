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
