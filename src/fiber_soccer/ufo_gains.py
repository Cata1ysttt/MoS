"""Explicit PD-only diagnostic override; all other physics remain unchanged."""
import json
import types
import torch
from .paths import ASSETS


def install(core, output):
    robot=core.scene['robot']
    dep=json.loads((ASSETS/'ufo/g1_policy/exported/FBcprAuxModel.meta.json').read_text())['deployment']
    ids=[robot.joint_names.index(n) for n in dep['joint_kp']]
    names=list(dep['joint_kp'])
    kp=torch.tensor([dep['joint_kp'][n] for n in names],device=core.device).expand(core.num_envs,-1)
    kd=torch.tensor([dep['joint_kd'][n] for n in names],device=core.device).expand(core.num_envs,-1)
    before=dict(kp=robot.data.joint_stiffness[:,ids].cpu().tolist(),kd=robot.data.joint_damping[:,ids].cpu().tolist())
    def apply(env_ids=None):
        selected=torch.arange(core.num_envs,device=core.device) if env_ids is None else env_ids
        robot.write_joint_stiffness_to_sim(kp[selected],joint_ids=ids,env_ids=selected)
        robot.write_joint_damping_to_sim(kd[selected],joint_ids=ids,env_ids=selected)
        # Keep implicit actuator effort estimation synchronized with simulator gains.
        for actuator in robot.actuators.values():
            if not actuator.is_implicit_model:raise ValueError('PD diagnostic requires implicit actuators')
            actuator.stiffness[selected]=robot.data.joint_stiffness[selected][:,actuator.joint_indices]
            actuator.damping[selected]=robot.data.joint_damping[selected][:,actuator.joint_indices]
    original=core._reset_idx
    def reset(self,env_ids):
        result=original(env_ids);apply(env_ids);return result
    apply();core._reset_idx=types.MethodType(reset,core)
    torch.testing.assert_close(robot.data.joint_stiffness[:,ids],kp)
    torch.testing.assert_close(robot.data.joint_damping[:,ids],kd)
    (output/'pd_override.json').write_text(json.dumps(dict(mode='ufo_native_pd_only',joint_names=names,before=before,after=dict(kp=kp.cpu().tolist(),kd=kd.cpu().tolist()),note='Not a native dynamics replica: implicit actuator, geometry, friction, effort limits and other randomization unchanged.'),indent=2))
