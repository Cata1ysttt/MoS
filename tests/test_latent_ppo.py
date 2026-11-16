import torch
from fiber_soccer.latent_ppo import (critic_features, gaussian_log_prob, mean_kl,
    generalized_advantages, clipped_policy_loss)


def test_timeout_uses_terminal_state_and_stops_trace():
    # Env 0 times out then resets to a high-value state; env 1 truly terminates.
    rewards=torch.tensor([[1., 1.], [100., 100.]])
    values=torch.zeros_like(rewards)
    next_values=torch.tensor([[999., 999.], [0., 0.]])
    terminated=torch.tensor([[False, True], [False, False]])
    truncated=torch.tensor([[True, True], [False, False]])
    terminal=torch.tensor([[5., 5.], [0., 0.]])
    adv,ret=generalized_advantages(rewards,values,next_values,terminated,truncated,terminal,gamma=.9,lam=1.)
    torch.testing.assert_close(adv[0],torch.tensor([5.5,1.]))
    torch.testing.assert_close(adv[1],torch.tensor([100.,100.]))
    torch.testing.assert_close(ret,adv)


def test_ordinary_transitions_include_future_advantage():
    r=torch.tensor([[1.],[2.]])
    z=torch.zeros_like(r);done=torch.zeros_like(r,dtype=torch.bool)
    adv,_=generalized_advantages(r,z,z,done,done,z,gamma=.9,lam=.5)
    torch.testing.assert_close(adv[:,0],torch.tensor([1.9,2.]))


def test_gaussian_probability_matches_distribution_before_projection():
    torch.manual_seed(7)
    mean=torch.randn(5,256);raw=mean+.05*torch.randn_like(mean)
    expected=torch.distributions.Normal(mean,.05).log_prob(raw).sum(-1)
    torch.testing.assert_close(gaussian_log_prob(raw,mean,.05),expected)
    torch.testing.assert_close(mean_kl(mean,mean,.05),torch.zeros(5))
    torch.testing.assert_close(mean_kl(mean,mean+.05,.05),torch.full((5,),128.))


def test_policy_clipping_limits_good_and_bad_actions():
    old=torch.zeros(2);new=torch.tensor([2.,.5]).log().requires_grad_()
    loss=clipped_policy_loss(new,old,torch.tensor([1.,-1.]))
    torch.testing.assert_close(loss,torch.tensor(-.2))
    loss.backward();torch.testing.assert_close(new.grad,torch.zeros(2))


def test_terminal_critic_ignores_histories_and_matches_live_features():
    x=torch.arange(2406.).expand(2,-1).clone()
    v2=torch.cat((x[:,:944],x[:,-6:]),-1)
    torch.testing.assert_close(critic_features(x),critic_features(v2))
    x[:,93:930]=-10000;x[:,944:2400]=10000
    torch.testing.assert_close(critic_features(x),critic_features(v2))
