import copy
from types import SimpleNamespace
import pytest
import torch
from fiber_soccer.continuous_shooting import ContinuousShootingPhase,SequenceCritic
from fiber_soccer.joint_quality import JointQualityPhase,MemorySequenceCritic
from fiber_soccer.ppo_resume import assert_tree_equal


def observe(phase,velocity,preferred):
    n=len(velocity);phase.observe(velocity,torch.tensor([[1.,0.]]*n),torch.ones(n)*2,
        torch.ones(n,dtype=torch.bool),preferred,torch.ones(n,dtype=torch.bool),phase.age)


def test_joint_bonus_needs_both_first_impacts_and_survival_for_timeout():
    p=JointQualityPhase(4,'cpu',joint_impact_bonus=10,joint_timeout_bonus=20,pass_delay=1)
    p.age.fill_(1);v=torch.tensor([[2.,0.],[0.,2.],[2.,0.],[2.,0.]])
    observe(p,v,torch.tensor([True,True,False,True]))
    # Repeated contact cannot replace the first inaccurate pass.
    observe(p,torch.tensor([[2.,0.]]*4),torch.ones(4,dtype=torch.bool))
    assert p.passes==4 and p.accurate_passes==2
    p.age.fill_(2);ids=p.ready().nonzero().flatten();p.start_shooting(ids,torch.ones(4))
    assert p.pass_accurate.tolist()==[True,False,False,True]
    p.begin_step();p.age.fill_(3)
    for _ in range(2):observe(p,torch.tensor([[2.,0.]]*4),torch.ones(4,dtype=torch.bool))
    assert p.shots==4 and p.joint_accurate_shots==2
    terminated=torch.tensor([False,False,False,True]);truncated=torch.ones(4,dtype=torch.bool)
    terms=p.rewards(torch.ones(4),torch.ones(4),torch.zeros(4),terminated,truncated,.02)
    assert terms[:,-2].tolist()==[10.,0.,0.,10.]
    assert terms[:,-1].tolist()==[20.,0.,0.,0.]
    assert p.joint_successful_timeouts==1
    p.reset(torch.tensor([True,False,False,True]))
    assert not p.memory(100)[[0,3]].any()
    p.begin_step();assert not p.new_shot.any()


def test_wrong_shot_cannot_earn_joint_reward_after_an_accurate_pass():
    p=JointQualityPhase(3,'cpu',joint_impact_bonus=10,joint_timeout_bonus=20)
    p.age.fill_(1);observe(p,torch.tensor([[2.,0.]]*3),torch.ones(3,dtype=torch.bool))
    p.start_shooting(torch.arange(3),torch.ones(3));p.age.fill_(2)
    observe(p,torch.tensor([[0.,2.],[.5,0.],[2.,0.]]),torch.tensor([True,True,False]))
    terms=p.rewards(torch.ones(3),torch.ones(3),torch.zeros(3),torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),.02)
    assert not terms[:,-2:].any() and p.joint_accurate_shots==0


def test_control_retains_original_rewards_and_memory_tracks_delays():
    old=ContinuousShootingPhase(2,'cpu',pass_delay=2,shoot_delay=1)
    new=JointQualityPhase(2,'cpu',pass_delay=2,shoot_delay=1)
    for p in [old,new]:
        p.age.fill_(1);observe(p,torch.tensor([[2.,0.],[0.,2.]]),torch.ones(2,dtype=torch.bool))
        p.age.fill_(2)
    assert new.memory(100)[:,-2].tolist()==[.5,.5]
    for p in [old,new]:
        p.age.fill_(3);p.start_shooting(torch.arange(2),torch.ones(2));p.begin_step()
        observe(p,torch.tensor([[2.,0.],[2.,0.]]),torch.ones(2,dtype=torch.bool))
    args=(torch.ones(2)*.7,torch.ones(2),torch.ones(2)*.1,torch.tensor([False,True]),torch.tensor([True,False]),.02)
    original=old.rewards(*args);extended=new.rewards(*args)
    assert torch.equal(original,extended[:,:6]) and not extended[:,6:].any()
    assert new.joint_successful_timeouts==1
    assert new.memory(100)[:,-1].tolist()==[1.,1.]


def test_critic_memory_preserves_function_rng_and_existing_adam_state():
    torch.manual_seed(123)
    planner=SimpleNamespace(obs_mean=torch.zeros(2406),obs_std=torch.ones(2406))
    base=SequenceCritic(planner,100);opt=torch.optim.Adam(base.parameters(),lr=.003)
    x=torch.randn(8,2406);stage=torch.arange(8)%3;age=torch.arange(8);target=torch.randn(8)
    for _ in range(2):
        opt.zero_grad();(base(x,stage,age)-target).square().mean().backward();opt.step()
    original=copy.deepcopy(base);old_opt=torch.optim.Adam(original.parameters(),lr=.003);old_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
    rng=torch.random.get_rng_state();wrapped=MemorySequenceCritic(base)
    assert torch.equal(rng,torch.random.get_rng_state())
    memory=torch.randn(8,10)
    assert torch.equal(original(x,stage,age),wrapped(x,stage,age,memory))
    previous=copy.deepcopy(opt.state_dict());opt.add_param_group({'params':wrapped.memory_projection.parameters()})
    assert_tree_equal(previous['state'],opt.state_dict()['state'])
    # Zero memory makes the next old-parameter update identical; the new branch gets its own step.
    old_opt.zero_grad();(original(x,stage,age)-target).square().mean().backward();old_opt.step()
    opt.zero_grad();(wrapped(x,stage,age,torch.zeros_like(memory))-target).square().mean().backward();opt.step()
    assert_tree_equal(original.state_dict(),base.state_dict())
    assert all(int(state['step'])==3 for param,state in opt.state.items() if param is not wrapped.memory_projection.weight)
    assert int(opt.state[wrapped.memory_projection.weight]['step'])==1
    with torch.no_grad():wrapped.memory_projection.weight.fill_(.01)
    assert not torch.equal(wrapped(x,stage,age,memory),wrapped(x,stage,age,torch.zeros_like(memory)))


@pytest.mark.parametrize('value',[-1.,float('nan'),float('inf')])
def test_joint_bonus_must_be_finite_nonnegative(value):
    with pytest.raises(ValueError):JointQualityPhase(1,'cpu',joint_impact_bonus=value)
