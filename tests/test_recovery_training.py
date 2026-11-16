import torch
from fiber_soccer.recovery_training import PostShotRecovery,stability_score,normalize_active_advantages


def test_recovery_waits_for_observed_contact_and_clears_after_reset():
    phase=PostShotRecovery(2,'cpu',delay=2)
    assert torch.equal(phase.skills(),torch.ones(2,dtype=torch.long))
    phase.advance(torch.tensor([True,False]),torch.tensor([False,False]))
    assert not phase.active.any()
    phase.advance(torch.tensor([True,True]),torch.tensor([False,True]))
    assert not phase.active.any()  # reset-contact cannot start a new recovery
    phase.advance(torch.tensor([True,False]),torch.tensor([False,False]))
    assert torch.equal(phase.skills(),torch.tensor([0,1])) and phase.entries==1
    phase.advance(torch.tensor([False,False]),torch.tensor([True,False]))
    assert not phase.active.any()
    for _ in range(3):phase.advance(torch.tensor([False,False]),torch.tensor([False,False]))
    assert not phase.active.any() and phase.entries==1


def test_stability_reward_prefers_upright_height_and_slow_motion():
    gravity=torch.tensor([[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.],[0.,0.,-1.]])
    height=torch.tensor([.76,.76,.4,.76]);v=torch.zeros(4,3);w=torch.zeros(4,3)
    v[3,0]=3.;w[3,0]=3.
    score=stability_score(gravity,height,v,w)
    assert score[0]==1 and score[1]==0
    assert 0<=score[2]<.001 and score[3]<score[0]
    assert ((score>=0)&(score<=1)).all()


def test_frozen_shot_advantages_do_not_change_recovery_normalization():
    values=torch.tensor([10000.,1.,3.,-10000.]);active=torch.tensor([False,True,True,False])
    torch.testing.assert_close(normalize_active_advantages(values,active),torch.tensor([0.,-1.,1.,0.]))
    assert torch.equal(normalize_active_advantages(values,torch.zeros_like(active)),torch.zeros_like(values))


def test_timeout_value_uses_next_phase_without_mutating_reset_state():
    phase=PostShotRecovery(1,'cpu',delay=1)
    kicked=torch.tensor([True]);done=torch.tensor([False])
    phase.advance(kicked,done)
    assert phase.skills().item()==1
    assert phase.next_skills_without_reset(kicked).item()==0
    assert phase.skills().item()==1 and phase.age.item()==1
    phase.advance(kicked,torch.tensor([True]))
    assert phase.skills().item()==1 and phase.first_kick.item()==-1
