"""PPO utilities for Gaussian raw latents projected by a fixed action transform.

Log probabilities refer to the sampled raw vector, before sphere projection.
The projection and frozen BFM are part of the environment action transform;
we do not pretend the projected vector has an ordinary Gaussian density.
"""
import math
import torch
from torch import nn


def critic_features(x):
    """Instantaneous state; accepts v2 or v4 without depending on terminal history."""
    if x.shape[-1] not in (950, 2406):
        raise ValueError('Critic requires v2 or v4 observations')
    return torch.cat((x[..., :93], x[..., 930:944], x[..., -6:]), -1)


class LatentCritic(nn.Module):
    def __init__(self, planner):
        super().__init__()
        self.register_buffer('mean', critic_features(planner.obs_mean).clone())
        self.register_buffer('std', critic_features(planner.obs_std).clamp_min(.05).clone())
        self.net = nn.Sequential(nn.Linear(113, 256), nn.ELU(), nn.Linear(256, 128),
                                 nn.ELU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(((critic_features(x)-self.mean)/self.std).clamp(-20, 20)).squeeze(-1)


def gaussian_log_prob(raw, mean, std):
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Exploration std must be finite and positive')
    return (-.5*((raw-mean)/std).square()-math.log(std)-.5*math.log(2*math.pi)).sum(-1)


def mean_kl(old_mean, new_mean, std):
    return ((old_mean-new_mean)/std).square().sum(-1)*.5


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           terminal_values, gamma=.99, lam=.95):
    """Stop traces at either reset, bootstrap truncations from pre-reset states."""
    shapes = [v.shape for v in (values, next_values, terminated, truncated, terminal_values)]
    if rewards.ndim != 2 or any(shape != rewards.shape for shape in shapes):
        raise ValueError('Expected matching time/environment tensors')
    bootstrap = torch.where(truncated, terminal_values, next_values)
    bootstrap = torch.where(terminated, torch.zeros_like(bootstrap), bootstrap)
    deltas = rewards+gamma*bootstrap-values
    advantages = torch.empty_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        tail = deltas[t]+gamma*lam*(~(terminated[t] | truncated[t]))*tail
        advantages[t] = tail
    return advantages, advantages+values


def clipped_policy_loss(log_prob, old_log_prob, advantage, clip=.2):
    ratio = (log_prob-old_log_prob).exp()
    return -torch.minimum(ratio*advantage, ratio.clamp(1-clip, 1+clip)*advantage).mean()
