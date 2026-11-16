"""PD diagnostic must survive randomized resets and synchronize actuator buffers."""
import json
from types import SimpleNamespace
import torch
from fiber_soccer import ufo_gains


def test_pd_override_reapplies_only_reset_lanes(tmp_path,monkeypatch):
    p=tmp_path/'ufo/g1_policy/exported';p.mkdir(parents=True)
    (p/'FBcprAuxModel.meta.json').write_text(json.dumps({'deployment':{'joint_kp':{'b':300.,'a':99.},'joint_kd':{'b':5.,'a':6.}}}))
    monkeypatch.setattr(ufo_gains,'ASSETS',tmp_path)
    data=SimpleNamespace(joint_stiffness=torch.ones(3,2),joint_damping=torch.ones(3,2))
    act=SimpleNamespace(is_implicit_model=True,joint_indices=[0,1],stiffness=torch.ones(3,2),damping=torch.ones(3,2))
    def write(field,value,joint_ids,env_ids):getattr(data,field)[env_ids[:,None],joint_ids]=value
    robot=SimpleNamespace(joint_names=['a','b'],data=data,actuators={'all':act},
        write_joint_stiffness_to_sim=lambda value,**kw:write('joint_stiffness',value,**kw),
        write_joint_damping_to_sim=lambda value,**kw:write('joint_damping',value,**kw))
    def reset(ids):
        data.joint_stiffness[ids]=17;data.joint_damping[ids]=2
    core=SimpleNamespace(scene={'robot':robot},device='cpu',num_envs=3,_reset_idx=reset)
    ufo_gains.install(core,tmp_path)
    core._reset_idx(torch.tensor([1]))
    expected=torch.tensor([[99.,300.]]).expand(3,-1)
    torch.testing.assert_close(data.joint_stiffness,expected)
    torch.testing.assert_close(act.stiffness,expected)
    torch.testing.assert_close(act.damping,torch.tensor([[6.,5.]]).expand(3,-1))
