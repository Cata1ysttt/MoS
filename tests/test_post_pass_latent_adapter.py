import pytest
import torch
from fiber_soccer.post_pass_latent_adapter import LatentResidual


def test_zero_start_preserves_latent_rng_and_learns_without_changing_normalization():
    x=torch.randn(4,2406);base=torch.randn(4,256);base=16*base/base.norm(dim=-1,keepdim=True)
    mean=torch.randn(2406);std=torch.ones(2406);state=torch.random.get_rng_state()
    layer=LatentResidual(mean,std,16)
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.equal(layer(x,base),base)
    desired=base.roll(1,-1);optimizer=torch.optim.Adam(layer.parameters(),lr=.01)
    loss=(layer(x,base)-desired).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
    changed=layer(x,base)
    assert not torch.equal(changed,base)
    assert torch.allclose(changed.norm(dim=-1),base.norm(dim=-1),atol=2e-6,rtol=0)
    assert torch.equal(layer.mean,mean) and torch.equal(layer.std,std)
    assert mean.grad is None and std.grad is None


def test_cancelling_correction_and_invalid_shapes():
    layer=LatentResidual(torch.zeros(2406),torch.ones(2406),8)
    x=torch.zeros(2,2406);base=torch.zeros(2,256);base[:,0]=16
    with torch.no_grad():layer.net[-1].bias[0]=-16
    assert torch.equal(layer(x,base),base)
    with pytest.raises(ValueError):layer(x[:,:64],base)
    with pytest.raises(ValueError):LatentResidual(torch.zeros(12),torch.ones(12))
    with pytest.raises(ValueError):LatentResidual(torch.zeros(2406),torch.ones(2406),True)
