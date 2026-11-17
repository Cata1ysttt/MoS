import pytest
import torch
from fiber_soccer.initial_goal_recovery import InitialBackwardGoal


def obs(n=2):
    x=torch.zeros(n,2406);x[:,944]=16;x[:,-2]=1
    return x


def test_goal_capture_contact_origin_and_new_ball_clear():
    gate=InitialBackwardGoal();x=obs();before=x.clone()
    goal,active=gate(x);assert not active.any() and torch.equal(goal,x[:,944:1200])
    assert torch.equal(x,before)
    x[:,940]=1;assert not gate(x)[1].any()  # Ordinary pass contact.
    x[:,-2:]=torch.tensor([0.,1.]);assert not gate(x)[1].any()
    x[:,940]=0;gate(x)  # New shooting ball clears old contact origin.
    x[:,940]=1;assert not gate(x)[1].any()  # Keep the shooting action intact.
    x[:,-2:]=torch.tensor([1.,0.]);assert gate(x)[1].all()
    x[:,940]=0;assert not gate(x)[1].any()
    assert not gate.state_dict()


def test_goal_validation_batch_and_partial_reset():
    gate=InitialBackwardGoal();x=obs()
    with torch.inference_mode():gate(x)
    with pytest.raises(ValueError,match='batch'):gate(obs(1))
    gate.reset(torch.tensor([True,False]))
    x[0,944:1200]=0
    with pytest.raises(ValueError,match='radius-16'):gate(x)
    x[0,945]=16;gate(x)
    assert gate.goal[0,1]==16 and gate.goal[1,0]==16
    gate.reset();gate(obs(1))
    with pytest.raises(ValueError):gate(torch.zeros(1,1206))
