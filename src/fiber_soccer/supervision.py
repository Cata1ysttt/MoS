"""Training-only proximity weights; never append privileged inputs to policy obs."""
import math
import torch


def proximity_weights(robot_position, ball_position, radius=0.6, multiplier=1.0):
    """Horizontal pelvis-ball neighborhood, not a claim of physical contact."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError('multiplier must be finite and >=1')
    distance = (robot_position[..., :2] - ball_position[..., :2]).norm(dim=-1)
    return torch.where(distance <= radius, multiplier, 1.0).detach()
