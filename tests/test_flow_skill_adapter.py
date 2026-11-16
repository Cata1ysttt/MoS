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


@pytest.mark.parametrize('space',['endpoint','latent'])
@pytest.mark.parametrize('observation_input',[False,True])
def test_adapter_learning_preserves_base_weights_and_other_skill_exactly(observation_input,space):
    base,x=fixture()
    rng=torch.random.get_rng_state()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,observation_input=observation_input,space=space)
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


@pytest.mark.parametrize('space',['endpoint','latent'])
@pytest.mark.parametrize('observation_input',[False,True])
def test_trained_adapter_roundtrips_and_remains_frozen_except_adapter(observation_input,space):
    base,x=fixture()
    model=base.enable_skill_adapter(0,width=16,observation_input=observation_input,space=space)
    torch.nn.init.normal_(model.adapter[-1].weight,std=.01)
    ck=dict(planner=model.state_dict(),planner_arch='flow_latent_adapter_v1' if space=='latent' else 'flow_endpoint_adapter_v1',observation_schema=SCHEMA_V4,
            horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,command_conditioning=True,
            adapter_width=16,adapter_skill=0,adapter_observation_input=observation_input)
    restored=load_planner(ck,'cpu')
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    assert {n for n,p in restored.named_parameters() if p.requires_grad}=={n for n,p in restored.named_parameters() if n.startswith('adapter.')}
    with pytest.raises(ValueError,match='already installed'):restored.enable_skill_adapter(1)
    with pytest.raises(ValueError):FlowLatentPlanner(width=32).enable_skill_adapter(2)


@pytest.mark.parametrize('space',['endpoint','latent'])
def test_full_observation_adapter_can_use_history_missing_from_frozen_context(space):
    base,x=fixture()
    with torch.no_grad():base.condition[0].weight[:,1200:2400]=0
    changed=x.clone();changed[:,1200:2400]+=1
    torch.testing.assert_close(base(x),base(changed),rtol=0,atol=0)
    model=base.enable_skill_adapter(0,width=16,observation_input=True,space=space)
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


def test_latent_adapter_preserves_flow_endpoints_future_latents_and_radius():
    base,x=fixture()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,observation_input=True,space='latent')
    torch.nn.init.normal_(model.adapter[-1].weight,std=.1)
    noise=torch.randn(6,2,256);time=torch.full((6,),.3)
    torch.testing.assert_close(model.endpoint(noise,time,x),base.endpoint(noise,time,x),rtol=0,atol=0)
    actual=model(x);original=base(x)
    torch.testing.assert_close(actual[:,1:],original[:,1:],rtol=0,atol=0)
    torch.testing.assert_close(actual.norm(dim=-1),original.norm(dim=-1),rtol=2e-6,atol=2e-6)
    assert not torch.equal(actual[::2,0],original[::2,0])
    model.set_adapter_ball_distance_gate(.4,1.2)
    x[:,933:935]=torch.tensor([2.,0.])
    torch.testing.assert_close(model(x),base(x),rtol=0,atol=0)


def test_latent_adapter_cancelling_correction_retains_valid_latent():
    base,x=fixture()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,space='latent')
    original=base(x).detach()
    with torch.no_grad():model.adapter[-1].bias.copy_(-original[0,0])
    output=model(x)
    torch.testing.assert_close(output[0],original[0],rtol=0,atol=0)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.adapter.parameters())


def test_adapter_checkpoint_rejects_conflicting_architecture_metadata():
    base,x=fixture()
    model=base.enable_skill_adapter(0,width=16,space='latent')
    ck=dict(planner=model.state_dict(),planner_arch='flow_latent_adapter_v1',observation_schema=SCHEMA_V4,
            horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,command_conditioning=True,
            adapter_width=16,adapter_skill=0,adapter_space='endpoint')
    with pytest.raises(ValueError,match='conflicts'):load_planner(ck,'cpu')
    with pytest.raises(ValueError,match='space'):FlowLatentPlanner(width=32).enable_skill_adapter(0,space='invalid')


@pytest.mark.parametrize('space',['endpoint','latent'])
def test_post_kick_adapter_preserves_precontact_and_shooting_through_learning_and_reload(space):
    base,x=fixture()
    x[:,940]=torch.tensor([0.,0.,1.,1.,0.,1.])
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,observation_input=True,space=space).set_adapter_post_kick_only()
    original=base(x).detach()
    protected=~((x[:,-2]==1)&(x[:,940]>.5))
    opt=torch.optim.Adam(model.adapter.parameters(),lr=.01)
    target=torch.randn_like(original)
    for _ in range(3):
        opt.zero_grad();(model(x)-target).square().mean().backward();opt.step()
    torch.testing.assert_close(model(x)[protected],original[protected],rtol=0,atol=0)
    assert not torch.equal(model(x)[~protected],original[~protected])
    opt.zero_grad();model(x)[protected].square().sum().backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.adapter.parameters())
    ck=dict(planner=model.state_dict(),planner_arch='flow_latent_adapter_v1' if space=='latent' else 'flow_endpoint_adapter_v1',
        observation_schema=SCHEMA_V4,horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,
        command_conditioning=True,adapter_width=16,adapter_skill=0,adapter_observation_input=True,
        adapter_post_kick_only=True)
    restored=load_planner(ck,'cpu')
    assert restored.adapter_post_kick_only
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    with pytest.raises(ValueError):base.set_adapter_post_kick_only()
    with pytest.raises(ValueError):model.set_adapter_post_kick_only('true')


@pytest.mark.parametrize('space',['endpoint','latent'])
def test_runtime_delay_masks_only_correction_without_altering_contact_observation(space):
    from fiber_soccer.skill_transition import PostKickAdapterDelay
    base,x=fixture();x[:,940]=1.
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16,observation_input=True,space=space).set_adapter_post_kick_only()
    torch.nn.init.normal_(model.adapter[-1].weight,std=.02)
    untouched=x.clone();original=base(x).detach();full=model(x).detach()
    assert not torch.equal(original[::2],full[::2])
    delay=PostKickAdapterDelay(10)
    for _ in range(10):
        mask=delay(x[:,940]>.5)
        torch.testing.assert_close(model(x,adapter_mask=mask),original,atol=0,rtol=0)
    mask=delay(x[:,940]>.5)
    torch.testing.assert_close(model(x,adapter_mask=mask),full,atol=0,rtol=0)
    assert torch.equal(x,untouched)
