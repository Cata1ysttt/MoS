"""Explicit candidate G1 backward observation map, pending author golden vectors.

Order follows the 29-joint control chain, pelvis first and a virtual head last.
XZ rotation convention and the head offset agree with checkpoint normalizer
statistics, but statistical agreement is not proof of the original extractor.
Keep this map opt-in until closed-loop comparisons and source confirmation.
"""
import torch
from .ufo import privileged, rotate

MAPPING = 'candidate_g1_31_xz_head035'


def backward_observation(body_names, pos, quat, vel, omega, joint_names):
    names=['pelvis']+[n.replace('_joint','_link') for n in joint_names]
    names[names.index('waist_pitch_link')]='torso_link'
    if len(names)!=30 or len(set(names))!=30:raise ValueError('Unexpected G1 joint chain')
    indices=[body_names.index(n) for n in names]
    p=pos[:,indices];q=quat[:,indices];v=vel[:,indices];w=omega[:,indices]
    torso=names.index('torso_link')
    offset=torch.zeros_like(p[:,torso]);offset[:,2]=.35;offset=rotate(q[:,torso],offset)
    p=torch.cat((p,(p[:,torso]+offset)[:,None]),1)
    v=torch.cat((v,(v[:,torso]+torch.cross(w[:,torso],offset,dim=-1))[:,None]),1)
    q=torch.cat((q,q[:,torso:torso+1]),1);w=torch.cat((w,w[:,torso:torso+1]),1)
    result=privileged(p,q,v,w)
    if result.shape[-1]!=463 or not torch.isfinite(result).all():raise ValueError('Invalid privileged state')
    return result


def privileged_from_env(core,bridge):
    robot=core.scene['robot'].data
    names=[bridge.robot.joint_names[i] for i in bridge.ids]
    return backward_observation(core.scene['robot'].body_names,
        robot.body_pos_w-core.scene.env_origins[:,None],robot.body_quat_w,
        robot.body_link_lin_vel_w,robot.body_ang_vel_w,names)
