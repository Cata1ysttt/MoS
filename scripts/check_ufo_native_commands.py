"""Official mjlab command baseline with numeric traces, no projector."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from ufo.inference.runtime import load_mjlab_env_cfg
from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir
from ufo.math import quat_rotate_inverse

base = Path('/home/jiongye/fiber-soccer')
assets = base / 'assets/ufo/g1_policy'
out = base / 'runs/ufo_standalone_native_v2'
out.mkdir(exist_ok=False)
cfg, _ = load_mjlab_env_cfg(assets, data_path=Path('/home/jiongye/UFO/ufo/data/lafan_29dof.pkl'),
                           device='cuda:0', disable_dr=True, disable_obs_noise=True,
                           max_episode_length_s=1000)
wrapped, _ = cfg.build(num_envs=1)
env = wrapped._env
env.is_evaluating = True
model = load_model_from_checkpoint_dir(assets / 'checkpoint', device='cpu')
model.to('cuda:0').eval()
ref = np.load(base / 'runs/ufo_reference_export/isaac_reference.npz')
walking = joblib.load(assets / 'reward_inference/reward_locomotion.pkl')
commands = [('tracking_motion7', ref['z'])] + [(k, np.repeat(np.asarray(v).reshape(1,256),500,0)) for k,v in walking.items()]
results = {}
try:
    for name, zs in commands:
        if name == 'tracking_motion7':
            root = torch.tensor(ref['root_state_wxyz'][0],device='cuda:0').unsqueeze(0)
            root[:, 3:7] = root[:, [4,5,6,3]].clone()
            root[:, :3] += env.env_origins
            joints = torch.tensor(np.stack((ref['joint_pos'][0],ref['joint_vel'][0]),-1),device='cuda:0').unsqueeze(0)
            target = dict(root_states=root,dof_states=joints)
        else:
            init = env.config.robot.init_state
            root = torch.zeros(1,13,device='cuda:0')
            root[:,:3] = torch.tensor(list(init.pos),device='cuda:0') + env.env_origins
            root[:,3:7] = torch.tensor(list(init.rot),device='cuda:0')
            joints = torch.zeros(1,env.num_dof,2,device='cuda:0')
            joints[:,:,0] = env.default_dof_pos
            target = dict(root_states=root,dof_states=joints)
        obs, _ = wrapped.reset(to_numpy=False,target_states=target)
        trace = []
        first_fall = None
        for tick in range(500):
            with torch.no_grad():
                action = model.act(obs,torch.tensor(zs[tick],device='cuda:0').unsqueeze(0),mean=True)
                obs, _, terminated, truncated, _ = wrapped.step(action,to_numpy=False)
                root = env.robot_root_states.clone()
                root[:, :3] -= env.env_origins
                tilt = torch.acos((-env.projected_gravity[:,2]).clamp(-1,1))
                vel = quat_rotate_inverse(env.base_quat,root[:,7:10],w_last=True)
                if first_fall is None and bool(((root[:,2]<.4)|(tilt>torch.pi/3)).any()):
                    first_fall = (tick+1)*.02
                trace.append(dict(root=root.cpu(),velocity=vel.cpu(),joint=env.dof_pos.cpu().clone(),tilt=tilt.cpu()))
                if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                    break
        result = dict(steps=len(trace),first_fall_seconds=first_fall,
                      terminated=bool(torch.as_tensor(terminated).any()),
                      mean_forward_velocity_last_half=float(torch.stack([t['velocity'] for t in trace])[len(trace)//2:,:,0].mean()))
        if name == 'tracking_motion7':
            q=torch.stack([t['joint'] for t in trace])
            result['joint_rmse_rad']=float(((q-torch.tensor(ref['joint_pos'][1:len(trace)+1])[:,None])**2).mean().sqrt())
        torch.save(trace,out / f'{name}_trace.pt')
        results[name]=result
        (out/'summary.json').write_text(json.dumps(results,indent=2))
        print(name,result,flush=True)
finally:
    wrapped.close()
