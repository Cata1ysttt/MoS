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


def test_adapter_learning_preserves_base_weights_and_other_skill_exactly():
    base,x=fixture()
    rng=torch.random.get_rng_state()
    model=copy.deepcopy(base).enable_skill_adapter(0,width=16)
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


def test_trained_adapter_roundtrips_and_remains_frozen_except_adapter():
    base,x=fixture()
    model=base.enable_skill_adapter(0,width=16)
    torch.nn.init.normal_(model.adapter[-1].weight,std=.01)
    ck=dict(planner=model.state_dict(),planner_arch='flow_endpoint_adapter_v1',observation_schema=SCHEMA_V4,
            horizon=2,width=32,flow_steps=3,noise_seed=model.noise_seed,command_conditioning=True,
            adapter_width=16,adapter_skill=0)
    restored=load_planner(ck,'cpu')
    torch.testing.assert_close(restored(x),model(x),rtol=0,atol=0)
    assert {n for n,p in restored.named_parameters() if p.requires_grad}=={n for n,p in restored.named_parameters() if n.startswith('adapter.')}
    with pytest.raises(ValueError,match='already installed'):restored.enable_skill_adapter(1)
    with pytest.raises(ValueError):FlowLatentPlanner(width=32).enable_skill_adapter(2)
