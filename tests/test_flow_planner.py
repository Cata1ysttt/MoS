import torch
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.planner import SCHEMA_V3,load_planner
from fiber_soccer.lpp import project


def test_euler_endpoint_sampler_exact_constant_target_and_deterministic_calls():
    model=FlowLatentPlanner(horizon=3,width=32,flow_steps=8)
    x=torch.randn(2,1206);prior=project(torch.randn(2,256));x[:,-262:-6]=prior
    # Zero-initialized residual head yields the constant prior endpoint.
    torch.testing.assert_close(model(x),prior[:,None].expand(-1,3,-1))
    state=torch.random.get_rng_state();a=model(x);b=model(x)
    assert torch.equal(torch.random.get_rng_state(),state)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(model(x.flip(0)),a.flip(0))
    ck=dict(planner=model.state_dict(),planner_arch='flow_endpoint_v1',observation_schema=SCHEMA_V3,
            horizon=3,width=32,flow_steps=8,noise_seed=model.noise_seed)
    torch.testing.assert_close(load_planner(ck,'cpu')(x),a)


def test_endpoint_is_not_projected_before_flow_training_loss():
    model=FlowLatentPlanner(horizon=2,width=32)
    x=torch.zeros(2,1206);x[:,-262]=3
    raw=model.endpoint(torch.randn(2,2,256),torch.tensor([0.,.8]),x)
    torch.testing.assert_close(raw.norm(dim=-1),torch.full((2,2),3.))
    torch.testing.assert_close(model(x).norm(dim=-1),torch.full((2,2),16.))
    raw.square().mean().backward()
    assert model.field[-1].weight.grad.abs().sum()>0


def test_command_encoder_changes_condition_and_roundtrips_checkpoint():
    model=FlowLatentPlanner(horizon=2,width=32,command_conditioning=True)
    x=torch.zeros(2,1206);x[0,-2]=1;x[1,-1]=1
    assert not torch.allclose(model.encode(x)[0],model.encode(x)[1])
    ck=dict(planner=model.state_dict(),planner_arch='flow_endpoint_v1',observation_schema=SCHEMA_V3,
            horizon=2,width=32,flow_steps=8,noise_seed=model.noise_seed,command_conditioning=True)
    restored=load_planner(ck,'cpu')
    torch.testing.assert_close(restored.encode(x),model.encode(x))
