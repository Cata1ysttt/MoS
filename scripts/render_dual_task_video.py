"""Four fixed environment lanes, frozen at first failure, rendered from Isaac states."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
import argparse
import json
from pathlib import Path
import torch
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--envs',type=int,nargs=4,default=[0,1,2,3]);p.add_argument('--single-env',type=int);a=p.parse_args()
if a.single_env is not None:a.envs=[a.single_env]
width,height=(640,480) if len(a.envs)==1 else (480,320)
assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
t=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=True)
summary=json.loads((a.run/'summary.json').read_text())
states=t['states'].numpy();dt=t['dt'];failures={r['environment']:r for r in summary['first_resets']}
s=mujoco.MjSpec.from_file('/home/jiongye/I-BFM-Code/ibfm/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml')
s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.05],rgba=[.35,.45,.38,1])
s.visual.headlight.ambient=[.6,.6,.6];s.visual.headlight.diffuse=[.7,.7,.7]
for grid in range(-30,31):
 for axis in range(2):
  position=[0.,0.,.001];position[axis]=grid
  size=[30.,30.,.001];size[axis]=.006
  s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,pos=position,size=size,rgba=[.24,.34,.27,1])
b=s.worldbody.add_body(name='ball');b.add_freejoint(name='ball_joint');b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.1155,0,0],mass=.45,rgba=[1,.75,.2,1])
s.worldbody.add_light(pos=[0,-3,5],dir=[0,.5,-1],diffuse=[.8,.8,.8],ambient=[.4,.4,.4])
m=s.compile();d=mujoco.MjData(m);r=mujoco.Renderer(m,height=height,width=width)
addresses=[m.jnt_qposadr[m.joint(n).id] for n in t['joint_names']];ball=m.jnt_qposadr[m.joint('ball_joint').id]
cam=mujoco.MjvCamera();cam.azimuth=125;cam.elevation=-18
with imageio.get_writer(a.output,fps=10,codec='libx264',quality=8) as writer:
 for tick in range(0,len(states),5):
  canvas=Image.new('RGB',(width,height) if len(a.envs)==1 else (2*width,2*height))
  for cell,env in enumerate(a.envs):
   end=failures.get(env,{}).get('tick',len(states)-1)
   stop=min(tick,end);q=states[stop,env].copy();origin=states[0,env,:2]
   q[:2]-=origin;q[-3:-1]-=origin
   d.qpos[:7]=q[:7];d.qpos[addresses]=q[7:-3];d.qpos[ball:ball+3]=q[-3:];d.qpos[ball+3:ball+7]=[1,0,0,0]
   mujoco.mj_forward(m,d);cam.lookat[:]=[(q[0]+q[-3])/2,(q[1]+q[-2])/2,.55]
   cam.distance=max(3.,float(np.linalg.norm(q[:2]-q[-3:-1]))*1.5+1.5)
   r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
   im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,width,56),fill='black')
   status='FIRST FAILURE - frozen' if env in failures and tick>=end else 'continuous first episode'
   draw.text((5,3),f'env {env} | {stop*dt:.2f}s | {status}',fill='white')
   events=[]
   for label,key in [('pass','pass_impacts'),('shot','shoot_impacts')]:
    record=summary[key].get(str(env))
    if record and record['tick']<=stop:
     events.append(f"{label}: {record['direction_error_deg']:.1f}deg, speed err {100*record['relative_forward_speed_error']:.0f}%")
   draw.text((5,19),' | '.join(events) or 'approaching ball',fill='white')
   draw.text((5,35),'Recorded Isaac states; rendering only; no resets stitched',fill='white')
   canvas.paste(im,((cell%2)*width,(cell//2)*height))
  writer.append_data(np.asarray(canvas))
r.close()
print(a.output)
