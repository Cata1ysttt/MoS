import torch
from fiber_soccer.continuous_shooting import ContinuousShootingPhase
from fiber_soccer.latent_ppo import generalized_advantages


def test_sequence_phase_delays_match_control_ticks_and_reset_requires_a_new_pass():
    p=ContinuousShootingPhase(3,'cpu',pass_delay=2,shoot_delay=1)
    v=torch.tensor([[2.,0.]]*3);direction=v/2;speed=torch.ones(3)*2
    p.age.fill_(1);p.begin_step()
    p.observe(v,direction,speed,torch.tensor([True,False,True]),torch.ones(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),p.age)
    p.age.fill_(2);assert not p.ready().any()
    p.age.fill_(3);assert p.ready().tolist()==[True,False,True]
    ids=p.ready().nonzero().flatten();p.start_shooting(ids,torch.ones(3)*4)
    assert p.skills().tolist()==[1,0,1]
    preferred=torch.tensor([True,True,False]);eligible=torch.tensor([True,False,True])
    p.begin_step();p.age.fill_(4)
    for _ in range(2):p.observe(v*3,direction,speed*3,eligible,preferred,eligible,p.age)
    assert p.shots==2 and p.accurate_shots==1
    terms=p.rewards(torch.ones(3)*4,torch.ones(3),torch.zeros(3),torch.tensor([False,False,True]),torch.zeros(3,dtype=torch.bool),.02)
    assert terms[:,1].tolist()==[17.,0.,0.]
    assert terms[:,4].tolist()==[0.,0.,-10.]
    p.advance_recovery();assert p.skills().tolist()==[1,0,1]
    p.age.fill_(5);p.advance_recovery();assert p.stage.tolist()==[2,0,2]
    p.reset(torch.tensor([True,False,True]))
    assert p.stage.tolist()==[0,0,0] and p.first_pass.tolist()==[-1,-1,-1]
    assert not p.ready().any()


def test_prefix_rewards_are_zero_and_terminal_success_requires_accurate_shot():
    p=ContinuousShootingPhase(2,'cpu')
    terms=p.rewards(torch.ones(2),torch.ones(2),torch.ones(2),torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),.02)
    assert torch.equal(terms,torch.zeros_like(terms))
    p.stage.fill_(2);p.first_shot.fill_(1);p.accurate[0]=True
    terms=p.rewards(torch.ones(2),torch.ones(2),torch.zeros(2),torch.zeros(2,dtype=torch.bool),torch.ones(2,dtype=torch.bool),.02)
    assert terms[:,-1].tolist()==[5.,0.] and p.successful_timeouts==1


def test_finite_sequence_horizon_does_not_bootstrap_past_the_scored_episode():
    reward=torch.tensor([[2.]])
    values=torch.tensor([[3.]])
    done=torch.ones(1,1,dtype=torch.bool)
    adv,returns=generalized_advantages(reward,values,torch.tensor([[100.]]),done,torch.zeros_like(done),torch.zeros_like(values))
    assert returns.item()==2 and adv.item()==-1
