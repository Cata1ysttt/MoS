import copy
import torch
import pytest
from fiber_soccer.flow_planner import FlowLatentPlanner
from fiber_soccer.planner import SCHEMA_V4,load_planner


def fixture():
    torch.manual_seed(813301)
    base=FlowLatentPlanner(horizon=2,width=32,schema=SCHEMA_V4,flow_steps=3,command_conditioning=True)
    torch.nn.init.normal_(base.field[-1].weight,std=.02)
    x=torch.randn(6,2406)
    x[:,-2:]=0
    x[::2,-2]=1;x[1::2,-1]=1
    return base,x


@pytest.mark.parametrize('observation_input',[False,True])
def test_adapter_learning_preserves_base_weights_and_other_skill_exactly(observation_input):
    base,x=fixture()
    rng=torch.random.get_rng_state()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,observation_input=observation_input)
    assert torch.equal(torch.random.get_rng_state(),rng)
    initial=base(x).detach()
    torch.testing.assert_close(model(x),initial,rtol=0,atol=0)
    old={k:v.clone() for k,v in base.state_dict().items()}
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    target=torch.randn_like(initial)
    for _ in range(4):
        opt.zero_grad();(model(x)-target).square().mean().backward();opt.step()
    assert not torch.equal(model(x)[::2],initial[::2])
    torch.testing.assert_close(model(x)[1::2],initial[1::2],rtol=0,atol=0)
    for key,value in old.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
    assert all(p.grad is None for name,p in model.named_parameters() if not name.startswith('adapter.'))
    assert all(p.requires_grad for p in model.adapter.parameters())
    # Preserve correction routing when slots are reordered or skills share a batch.
    torch.testing.assert_close(model(x.flip(0)),model(x).flip(0),rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('observation_input',[False,True])
def test_trained_adapter_roundtrips_and_remains_frozen_except_adapter(observation_input):
    base,x=fixture()
    model=base.enable_skill_adapter(0,width=16,observation_input=observation_input)
    torch.nn.init.normal_(model.adapter[-1].weight,std=.01)
    ck=dict(planner=model.state_dict(),planner_arch='flow_endpoint_adapter_v1',observation_schema=SCHEMA_V4,
            horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,command_conditioning=True,
            adapter_width=16,adapter_skill=0,adapter_observation_input=observation_input)
    restored=load_planner(ck,'cpu')
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    assert {n for n,p in restored.named_parameters() if p.requires_grad}=={n for n,p in restored.named_parameters() if n.startswith('adapter.')}
    with pytest.raises(ValueError,match='already installed'):restored.enable_skill_adapter(1)
    with pytest.raises(ValueError):FlowLatentPlanner(width=32).enable_skill_adapter(2)


def test_full_observation_adapter_can_use_history_missing_from_frozen_context():
    base,x=fixture()
    with torch.no_grad():base.condition[0].weight[:,1200:2400]=0
    changed=x.clone();changed[:,1200:2400]+=1
    torch.testing.assert_close(base(x),base(changed),rtol=0,atol=0)
    model=base.enable_skill_adapter(0,width=16,observation_input=True)
    torch.nn.init.normal_(model.adapter[-1].weight,std=.01)
    assert not torch.equal(model(x)[::2],model(changed)[::2])
    torch.testing.assert_close(model(x)[1::2],model(changed)[1::2],rtol=0,atol=0)


def test_distance_gate_preserves_far_policy_and_scales_endpoint_correction():
    base,x=fixture()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16)
    torch.nn.init.normal_(model.adapter[-1].weight,std=.01)
    gated=copy.deepcopy(model).set_adapter_ball_distance_gate(.4,1.2)
    x[0,933:935]=torch.tensor([.2,0.])
    x[2,933:935]=torch.tensor([2.,0.])
    x[4,933:935]=torch.tensor([.8,0.])
    torch.testing.assert_close(gated(x)[0],model(x)[0],rtol=0,atol=0)
    torch.testing.assert_close(gated(x)[2],base(x)[2],rtol=0,atol=0)
    torch.testing.assert_close(gated(x)[1::2],base(x)[1::2],rtol=0,atol=0)
    noise=torch.randn(6,2,256);time=torch.ones(6)*.3
    original=base.endpoint(noise,time,x)
    full=model.endpoint(noise,time,x)
    partial=gated.endpoint(noise,time,x)
    torch.testing.assert_close(partial[4]-original[4],.5*(full[4]-original[4]),atol=2e-6,rtol=1e-4)
    ck=dict(planner=gated.state_dict(),planner_arch='flow_endpoint_adapter_v1',observation_schema=SCHEMA_V4,
            horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,command_conditioning=True,
            adapter_width=16,adapter_skill=0,adapter_ball_distance_gate=[.4,1.2])
    torch.testing.assert_close(load_planner(ck,'cpu')(x),gated(x),rtol=0,atol=0)
    with pytest.raises(ValueError):gated.set_adapter_ball_distance_gate(1.,.5)
