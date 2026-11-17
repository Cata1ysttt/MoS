"""Render first recorded episode only, never stitch resets into a success video."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
import argparse,json
from pathlib import Path
import torch,numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--env',type=int,default=0);p.add_argument('--output',type=Path,help='Optional video destination outside the recorded run directory');a=p.parse_args()
output=a.output or a.run/f'env_{a.env}.mp4'
if output.exists():raise FileExistsError(f'Preserve existing video: {output}')
output.parent.mkdir(parents=True,exist_ok=True)
t=torch.load(a.run/'trace.pt',map_location='cpu',weights_only=True);rows=json.loads((a.run/'episodes.json').read_text());row=next(r for r in rows if r['environment']==a.env)
x=t['states'][:,a.env].numpy();x=x[:min(len(x),round(row['duration']/t['dt']))]
s=mujoco.MjSpec.from_file('/home/jiongye/I-BFM-Code/ibfm/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml')
s.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.05],rgba=[.16,.32,.2,1])
b=s.worldbody.add_body(name='ball');b.add_freejoint(name='ball_joint');b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.1155,0,0],mass=.45,rgba=[1,.75,.2,1])
s.worldbody.add_light(pos=[0,-3,5],dir=[0,.5,-1],diffuse=[.8,.8,.8],ambient=[.4,.4,.4])
m=s.compile();d=mujoco.MjData(m);r=mujoco.Renderer(m,height=240,width=320)
addresses=[m.jnt_qposadr[m.joint(n).id] for n in t['joint_names']];ball=m.jnt_qposadr[m.joint('ball_joint').id]
cam=mujoco.MjvCamera();cam.distance=3;cam.azimuth=125;cam.elevation=-18;origin=x[0,:2].copy()
with imageio.get_writer(output,fps=10) as w:
 for tick in range(0,len(x),5):
  q=x[tick].copy();q[:2]-=origin;q[-3:-1]-=origin
  d.qpos[:7]=q[:7];d.qpos[addresses]=q[7:-3];d.qpos[ball:ball+3]=q[-3:];d.qpos[ball+3:ball+7]=[1,0,0,0]
  mujoco.mj_forward(m,d);cam.lookat[:]=[(q[0]+q[-3])/2,(q[1]+q[-2])/2,.55];cam.distance=max(3.,float(np.linalg.norm(q[:2]-q[-3:-1]))*1.5+1.5);r.update_scene(d,camera=cam);r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
  im=Image.fromarray(r.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,320,34),fill='black');draw.text((4,3),a.run.name,fill='white');draw.text((4,18),f'env {a.env}, {tick*t["dt"]:.2f}s | recorded states',fill='white');w.append_data(np.asarray(im))
r.close()
