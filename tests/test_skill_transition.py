import pytest
import torch
from fiber_soccer.skill_transition import SkillTransition


def test_skill_switch_interpolates_on_sphere_and_finishes_at_exact_target():
    transition=SkillTransition(4)
    a=torch.tensor([[16.,0.],[16.,0.]])
    b=torch.tensor([[0.,16.],[16.,0.]])
    assert torch.equal(transition(a,torch.tensor([0,0])),a)
    for frame in range(1,5):
        result=transition(b,torch.tensor([1,0]))
        theta=torch.tensor(frame*torch.pi/8)
        expected=torch.stack((16*theta.cos(),16*theta.sin()))
        torch.testing.assert_close(result[0],expected,atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(result.norm(dim=-1),torch.full((2,),16.))
        assert torch.equal(result[1],b[1])
    assert torch.equal(result,b)
    assert torch.equal(transition(b,torch.tensor([1,0])),b)
    assert transition.state_dict()=={}


def test_reset_lane_does_not_blend_from_previous_episode_and_other_lane_continues():
    transition=SkillTransition(4)
    a=torch.tensor([[16.,0.],[16.,0.]])
    b=torch.tensor([[0.,16.],[0.,16.]])
    transition(a,torch.tensor([0,0]))
    first=transition(b,torch.tensor([1,1]))
    transition.reset(torch.tensor([True,False]))
    second=transition(b,torch.tensor([1,1]))
    assert torch.equal(second[0],b[0])
    assert not torch.equal(second[1],b[1])
    assert second[1,1]>first[1,1]
    with pytest.raises(ValueError,match='Reset all'):
        transition(b[:1],torch.tensor([1]))
    transition.reset()
    assert torch.equal(transition(b[:1],torch.tensor([0])),b[:1])


def test_mid_transition_retarget_starts_from_last_executed_latent():
    from fiber_soccer.lpp import slerp
    transition=SkillTransition(4)
    a=torch.tensor([[16.,0.]])
    b=torch.tensor([[0.,16.]])
    transition(a,torch.tensor([0]))
    first=transition(b,torch.tensor([1])).clone()
    returned=transition(a,torch.tensor([0]))
    torch.testing.assert_close(returned,slerp(first,a,.25))


def test_post_kick_delay_tracks_contact_and_reset_per_environment():
    from fiber_soccer.skill_transition import PostKickAdapterDelay
    delay=PostKickAdapterDelay(2)
    assert not delay(torch.tensor([False,False])).any()
    assert not delay(torch.tensor([True,False])).any()
    assert not delay(torch.tensor([True,True])).any()
    assert torch.equal(delay(torch.tensor([True,True])),torch.tensor([True,False]))
    delay.reset(torch.tensor([True,False]))
    assert torch.equal(delay(torch.tensor([True,True])),torch.tensor([False,True]))
    assert not delay(torch.tensor([False,False])).any()
    assert not delay.state_dict()  # elapsed time is runtime state, not learned weights
