import torch
from fiber_soccer.lpp import Projector
from fiber_soccer.multiskill import WarmStartedJointProjector, condition, load_projector


def test_warm_start_preserves_pass_and_checkpoint_roundtrip():
    torch.manual_seed(10)
    base = Projector(1200).eval()
    joint = WarmStartedJointProjector().eval()
    joint.base.load_state_dict(base.state_dict())
    x = torch.randn(3, 1200)
    with torch.no_grad():
        expected = base(x)
        torch.testing.assert_close(joint(condition(x, 'pass')), expected, rtol=0, atol=0)
    saved = dict(projector=joint.state_dict(), obs_dim=1202, projector_arch='warm_shared_v2')
    restored = load_projector(saved, 'cpu').eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(condition(x, 'pass')), expected, rtol=0, atol=0)


def test_skill_condition_and_shoot_gradient_reach_shared_trunk():
    model = WarmStartedJointProjector()
    x = condition(torch.randn(2, 1180), 'shoot')
    assert x.shape == (2, 1202)
    assert x[:, 1180:1201].count_nonzero() == 0
    assert torch.all(x[:, -1] == 1)
    model(x)[:, 0].sum().backward()
    assert model.skill_offset.grad[1].abs().sum() > 0
    assert model.skill_offset.grad[0].abs().sum() == 0
    assert next(model.base.parameters()).grad.abs().sum() > 0
