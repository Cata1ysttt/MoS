import torch
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.planner import DirectLatentPlanner, SCHEMA_V4, load_planner
from fiber_soccer.lpp import project


def test_direct_latent_context_transfer_prior_and_checkpoint_roundtrip():
    torch.manual_seed(713)
    source = FlowLatentPlanner(horizon=5,width=32,schema=SCHEMA_V4,command_conditioning=True)
    source.obs_mean.copy_(torch.randn(2406)*.1)
    source.obs_std.copy_(torch.rand(2406)+.2)
    direct = DirectLatentPlanner.from_context(source,horizon=1)
    x = torch.randn(4,2406)
    x[:,-2:] = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    torch.testing.assert_close(direct.encode(x),source.encode(x))
    torch.testing.assert_close(direct(x),project(x[:,944:1200])[:,None])
    # A learned residual remains differentiable and norm-constrained.
    torch.nn.init.normal_(direct.head[-1].weight,std=.02)
    z = direct(x)
    torch.testing.assert_close(z.norm(dim=-1),torch.full((4,1),16.))
    z[...,0].sum().backward()
    assert direct.head[-1].weight.grad.abs().sum() > 0
    assert direct.condition[0].weight.grad.abs().sum() > 0
    ck = dict(planner=direct.state_dict(),planner_arch='direct_latent_v1',
              observation_schema=SCHEMA_V4,horizon=1,width=32,command_conditioning=True)
    restored = load_planner(ck,'cpu')
    torch.testing.assert_close(restored(x),z)
    for i in range(4):
        torch.testing.assert_close(restored.act(x[i:i+1]),z[i:i+1,0],atol=1e-5,rtol=1e-5)
