import torch
from fiber_soccer.native_backward import backward_observation


def test_candidate_mapping_is_name_ordered_and_includes_virtual_head_velocity():
    joints=[f'j{i}_joint' for i in range(29)];joints[14]='waist_pitch_joint'
    names=['pelvis']+[n.replace('_joint','_link') for n in joints];names[15]='torso_link'
    pos=torch.zeros(1,30,3);pos[...,2]=.7
    quat=torch.zeros(1,30,4);quat[...,0]=1
    vel=torch.zeros_like(pos);omega=torch.zeros_like(pos);omega[:,15,0]=1
    b=backward_observation(names,pos,quat,vel,omega,joints)
    assert b.shape==(1,463)
    torch.testing.assert_close(b[:,88:91],torch.tensor([[0.,0.,.35]]))
    torch.testing.assert_close(b[:,367:370],torch.tensor([[0.,-.35,0.]]))
    permutation=torch.arange(29,-1,-1)
    reordered=backward_observation([names[i] for i in permutation],pos[:,permutation],quat[:,permutation],vel[:,permutation],omega[:,permutation],joints)
    torch.testing.assert_close(reordered,b)
    shifted=backward_observation(names,pos+torch.tensor([5.,-2.,0.]),quat,vel,omega,joints)
    torch.testing.assert_close(shifted,b)


def test_candidate_privileged_features_are_invariant_to_global_yaw():
    from fiber_soccer.ufo import rotate
    joints=[f'j{i}_joint' for i in range(29)];joints[14]='waist_pitch_joint'
    names=['pelvis']+[n.replace('_joint','_link') for n in joints];names[15]='torso_link'
    torch.manual_seed(17)
    pos=torch.randn(2,30,3);vel=torch.randn_like(pos);omega=torch.randn_like(pos)
    quat=torch.zeros(2,30,4);quat[...,0]=1
    angle=torch.tensor(.7);heading=torch.zeros_like(quat);heading[...,0]=torch.cos(angle/2);heading[...,3]=torch.sin(angle/2)
    expected=backward_observation(names,pos,quat,vel,omega,joints)
    rotated=backward_observation(names,rotate(heading,pos),heading,rotate(heading,vel),rotate(heading,omega),joints)
    torch.testing.assert_close(rotated,expected,atol=2e-6,rtol=1e-5)
