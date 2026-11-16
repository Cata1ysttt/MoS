import copy
import torch
import pytest
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.planner import SCHEMA_V4,load_planner
from fiber_soccer.target_guided import TargetRegressor,TargetGuidedLatentPlanner,target_guided_payload


def setup():
    torch.manual_seed(91)
    prior=FlowLatentPlanner(horizon=2,width=16,schema=SCHEMA_V4,flow_steps=2)
    predictor=TargetRegressor(prior.obs_mean,prior.obs_std,torch.zeros(29),torch.ones(29))
    x=torch.randn(8,2406);x[:,-2:]=torch.eye(2).repeat(4,1)
    return prior,predictor,x


def test_zero_decoder_exact_prior_and_rng():
    prior,predictor,x=setup();expected=prior(x).detach();rng=torch.random.get_rng_state()
    model=TargetGuidedLatentPlanner(prior,predictor,32)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(model(x),expected)
    assert torch.allclose(model.act(x).norm(dim=-1),torch.full((8,),16.))


def test_training_updates_only_decoder_and_roundtrip_preserves_loading_rng():
    prior,predictor,x=setup();model=TargetGuidedLatentPlanner(prior,predictor,32)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('decoder.')}
    executor=torch.nn.Linear(256,29,bias=False).requires_grad_(False)
    target=torch.randn(8,29);optimizer=torch.optim.Adam(model.decoder.parameters(),lr=.002)
    before=(executor(model.act(x))-target).square().mean().item()
    for _ in range(12):
        loss=(executor(model.act(x))-target).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert (executor(model.act(x))-target).square().mean().item()<before
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items())
    assert all(p.grad is None for p in prior.parameters())
    assert all(p.grad is None for p in predictor.parameters())
    assert all(p.grad is None for p in executor.parameters())
    header=dict(planner_arch='flow_endpoint_v1',planner=prior.state_dict(),observation_schema=SCHEMA_V4,
                horizon=2,width=16,flow_steps=2,noise_seed=812150,command_conditioning=False)
    payload=target_guided_payload(header,model,{})
    rng=torch.random.get_rng_state();load_planner(header,'cpu');expected_rng=torch.random.get_rng_state()
    torch.random.set_rng_state(rng);restored=load_planner(payload,'cpu')
    assert torch.equal(expected_rng,torch.random.get_rng_state())
    assert torch.equal(model(x),restored(x))
    assert torch.equal(restored(x)[:,1:],prior(x)[:,1:])


def test_target_condition_changes_latent_and_cancellation_is_safe():
    prior,predictor,x=setup();model=TargetGuidedLatentPlanner(prior,predictor,32)
    z=prior.act(x).detach();a=torch.zeros(8,29);b=torch.ones(8,29)*.5
    with torch.no_grad():model.decoder[-1].weight.normal_(std=.1)
    assert not torch.equal(model.decode(x,z,a),model.decode(x,z,b))
    assert torch.allclose(model.decode(x,z,b).norm(dim=-1),z.norm(dim=-1),atol=2e-6)
    with torch.no_grad():model.decoder[-1].weight.zero_();model.decoder[-1].bias.copy_(-z[0])
    assert torch.equal(model.decode(x[:1],z[:1],a[:1]),z[:1])
    with pytest.raises(ValueError):model.decode(x,z,a[:1])
