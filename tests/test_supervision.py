import pytest
import torch
from fiber_soccer.supervision import proximity_weights


def test_proximity_translation_height_and_default():
    robot = torch.tensor([[0., 0., 1.], [10., 2., 1.]])
    ball = torch.tensor([[.3, .4, .1], [11., 2., .1]])
    torch.testing.assert_close(proximity_weights(robot, ball), torch.ones(2))
    expected = torch.tensor([4., 1.])
    torch.testing.assert_close(proximity_weights(robot, ball, multiplier=4), expected)
    offset = torch.tensor([100., -5., 20.])
    torch.testing.assert_close(proximity_weights(robot+offset, ball+offset, multiplier=4), expected)


def test_weighted_mask_normalization_and_microbatch_gradient():
    # A reset frame must not influence the objective even when heavily weighted.
    pred = torch.tensor([[[1.]], [[2.]], [[999.]]], requires_grad=True)
    mask = torch.tensor([[1.], [1.], [0.]]) * torch.tensor([[4.], [1.], [4.]])
    denominator = mask.sum()
    full = (pred.square().sum(-1)*mask).sum()/denominator
    assert full.item() == pytest.approx(1.6)
    full.backward(); expected = pred.grad.clone(); pred.grad.zero_()
    for i in range(3):
        ((pred[i:i+1].square().sum(-1)*mask[i:i+1]).sum()/denominator).backward()
    torch.testing.assert_close(pred.grad, expected)
    assert pred.grad[2].item() == 0


@pytest.mark.parametrize('kwargs', [{'radius': 0}, {'radius': float('nan')}, {'multiplier': .5}, {'multiplier': float('inf')}])
def test_invalid_weights(kwargs):
    with pytest.raises(ValueError):
        proximity_weights(torch.zeros(1,3), torch.zeros(1,3), **kwargs)
