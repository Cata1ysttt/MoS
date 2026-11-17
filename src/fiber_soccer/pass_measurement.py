"""Passive first-valid post-kick velocity measurements, without changing control."""
import torch

MINIMUM_VALID_KICK_SPEED = .25


def first_kick_samples(velocity_xy, target_xy, target_speed, kicked, eligible):
    speed=velocity_xy.norm(dim=-1)
    valid=eligible & kicked & (speed>=MINIMUM_VALID_KICK_SPEED)
    direction=target_xy/target_xy.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    forward=(velocity_xy*direction).sum(-1)
    lateral=direction[:,0]*velocity_xy[:,1]-direction[:,1]*velocity_xy[:,0]
    angle=torch.rad2deg(torch.atan2(lateral,forward)).abs()
    relative_error=(forward-target_speed).abs()/target_speed.clamp_min(1e-6)
    return valid,torch.stack((speed,forward,angle,target_speed,relative_error),-1)
