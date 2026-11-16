import pytest
import torch
from fiber_soccer.planner_policy import recovery_controller_speed


def test_override_only_selected_controller_commands_without_changing_scoring():
    task=torch.tensor([2.2,6.,6.])
    original=task.clone()
    recovering=torch.tensor([False,True,False])
    assert recovery_controller_speed(task,recovering) is None
    control=recovery_controller_speed(task,recovering,2.5)
    assert torch.equal(control,torch.tensor([2.2,2.5,6.]))
    assert torch.equal(task,original)
    control.zero_()
    assert torch.equal(task,original)


def test_reject_invalid_recovery_command():
    task=torch.tensor([6.]);mask=torch.tensor([True])
    for speed in [float('nan'),float('inf'),0.,-1.]:
        with pytest.raises(ValueError):recovery_controller_speed(task,mask,speed)
    with pytest.raises(ValueError):recovery_controller_speed(task,torch.tensor([1]),2.5)
    with pytest.raises(ValueError):recovery_controller_speed(task,torch.tensor([True,False]),2.5)
