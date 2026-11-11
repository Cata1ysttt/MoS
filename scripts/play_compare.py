"""Replay measured Isaac trajectories with matched cameras; never run new dynamics."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
import argparse,json
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser()
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--expert',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--envs',type=int,nargs='+',default=[0,2,3,5])
p.add_argument('--analyze-only',action='store_true')
p.add_argument('--stills-only',action='store_true')
p.add_argument('--skip-full',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
run_paths=[a.candidate,a.expert]
traces=[torch.load(r/'trace.pt',weights_only=True) for r in run_paths]
episodes=[{v['environment']:v for v in json.loads((r/'episodes.json').read_text())} for r in run_paths]
summaries=[json.loads((r/'summary.json').read_text()) for r in run_paths]
assert summaries[0]['initial_state_sha256']==summaries[1]['initial_state_sha256']
assert json.loads((a.candidate/'ball_runtime_audit.json').read_text())==json.loads((a.expert/'ball_runtime_audit.json').read_text())
assert traces[0]['joint_names']==traces[1]['joint_names']
assert traces[0]['dt']==traces[1]['dt']==.02
dt=.02
diagnostics=torch.load(a.candidate/'action_diagnostics.pt',weights_only=True)
target_dirs={}
for env in a.envs:
    for row in diagnostics:
        # Verified policy term layout: target_dir at980:1010, chronological10-frame history.
        hist=row['projector_obs'][env,980:1010].reshape(10,3).numpy()
        direction=hist[-1,:2]
        if np.linalg.norm(direction)>.9:
            target_dirs[env]=direction/np.linalg.norm(direction);break

spec=mujoco.MjSpec.from_file('/home/jiongye/I-BFM-Code/ibfm/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml')
spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[30,30,.05],rgba=[.14,.25,.21,1])
ball=spec.worldbody.add_body(name='recorded_ball');ball.add_freejoint(name='recorded_ball_joint')
ball.add_geom(name='recorded_ball_geom',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.1155,0,0],mass=.45,rgba=[1,.7,.08,1])
spec.worldbody.add_light(pos=[1,-3,5],dir=[0,.4,-1],diffuse=[.9,.9,.9],ambient=[.5,.5,.5])
model=spec.compile();data=mujoco.MjData(model)
addresses=[model.jnt_qposadr[model.joint(n).id] for n in traces[0]['joint_names']]
ball_adr=model.jnt_qposadr[model.joint('recorded_ball_joint').id]
ball_geom=model.geom('recorded_ball_geom').id
feet=[model.body(f'{side}_ankle_roll_link').id for side in ['left','right']]
foot_geoms=[[i for i in range(model.ngeom) if model.geom(i).name.startswith(f'{side}_foot')] for side in ['left','right']]

def set_pose(q):
    data.qpos[:7]=q[:7];data.qpos[addresses]=q[7:-3]
    data.qpos[ball_adr:ball_adr+3]=q[-3:];data.qpos[ball_adr+3:ball_adr+7]=[1,0,0,0]
    mujoco.mj_forward(model,data)

def measure(q):
    foot_pos=[];foot_yaw=[];gap=[]
    for row in q:
        set_pose(row)
        foot_pos.append([data.xpos[b]+data.xmat[b].reshape(3,3)@np.array([.04,0,-.025]) for b in feet])
        foot_yaw.append([np.arctan2(data.xmat[b].reshape(3,3)[1,0],data.xmat[b].reshape(3,3)[0,0]) for b in feet])
        gap.append([min(mujoco.mj_geomDistance(model,data,g,ball_geom,10,None) for g in group) for group in foot_geoms])
    foot_pos=np.array(foot_pos);foot_yaw=np.array(foot_yaw);gap=np.array(gap)
    v=np.gradient(q[:,-3:],dt,axis=0);fv=np.gradient(foot_pos,dt,axis=0)
    root_distance=np.linalg.norm(q[:,-3:-1]-q[:,:2],axis=-1)
    # First close approach only, before later ball relaunches. Geometry is approximate.
    close=np.flatnonzero(root_distance<.9)
    if len(close):
        begin=int(close[0]);end=min(len(q),begin+150)
    else:begin=0;end=min(len(q),600)
    contacts=np.flatnonzero(gap[begin:end].min(1)<.008)
    tick=begin+int(contacts[0]) if len(contacts) else begin+int(np.argmin(gap[begin:end].min(1)))
    foot=int(np.argmin(gap[tick]));post=min(len(q)-1,tick+10);pre=max(0,tick-2)
    window=slice(max(0,tick-15),min(len(q),tick+16))
    report=dict(approx_contact=bool(len(contacts)),time_s=round(tick*dt,2),foot=['left','right'][foot],
                closest_surface_gap_m=float(gap[tick,foot]),foot_yaw_deg=float(np.rad2deg(foot_yaw[tick,foot])),
                foot_speed_xy_at_contact=float(np.linalg.norm(fv[tick,foot,:2])),
                max_foot_speed_xy_near_contact=float(np.linalg.norm(fv[window,foot,:2],axis=-1).max()),
                foot_velocity_xy_at_contact=fv[tick,foot,:2].tolist(),
                ball_velocity_xy_before=v[pre,:2].tolist(),ball_velocity_xy_0p2s_after=v[post,:2].tolist(),
                foot_height_m=float(foot_pos[tick,foot,2]),root_height_m=float(q[tick,2]),
                root_xy_at_contact=q[tick,:2].tolist())
    return report,dict(foot_pos=foot_pos,foot_yaw=foot_yaw,gap=gap,ball_velocity=v,foot_velocity=fv)

clips={};report={'source':'Recorded Isaac poses replayed with MuJoCo forward kinematics; no MuJoCo dynamics. Contact times/foot speeds are estimates from50Hz traces and proxy foot geometry.',
                  'candidate':str(a.candidate),'expert':str(a.expert),'paired_initial_state_and_ball_audit':True,'cases':{}}
for env in a.envs:
    origin=traces[0]['states'][0,env,:2].numpy()
    cases=[];arrays=[]
    for i,t in enumerate(traces):
        row=episodes[i][env];length=min(len(t['states']),round(row['duration']/dt))
        q=t['states'][:length,env].numpy().copy();q[:,:2]-=origin;q[:,-3:-1]-=origin
        stats,aux=measure(q)
        stats.update(duration=row['duration'],metrics=row['metrics'],termination_terms=row['termination_terms'])
        if env in target_dirs:
            direction=target_dirs[env]
            stats['target_yaw_deg']=float(np.rad2deg(np.arctan2(direction[1],direction[0])))
            stats['forward_speed_0p2s_after']=float(np.dot(stats['ball_velocity_xy_0p2s_after'],direction))
        cases.append(stats);arrays.append((q,aux))
    report['cases'][str(env)]={'candidate':cases[0],'expert':cases[1]};clips[env]=arrays
    print('CASE',env,[(v['time_s'],v['foot'],v['foot_speed_xy_at_contact'],v.get('forward_speed_0p2s_after')) for v in cases],flush=True)
(a.output/'kinematics.json').write_text(json.dumps(report,indent=2))
if a.analyze_only:raise SystemExit(0)

renderer=mujoco.Renderer(model,height=360,width=640)
cam=mujoco.MjvCamera();cam.azimuth=-70;cam.elevation=-22
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
viewopt=mujoco.MjvOption();viewopt.geomgroup[3:]=0

def arrow(start,end):
    if renderer.scene.ngeom>=renderer.scene.maxgeom:return
    g=renderer.scene.geoms[renderer.scene.ngeom]
    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.2,.9,.95,1]))
    mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.018,start,end)
    renderer.scene.ngeom+=1

def frame(env,tick,closeup=False,speed=1):
    arrays=clips[env];actual=[min(tick,len(q)-1) for q,_ in arrays]
    center=np.mean([q[t,:2] for (q,_),t in zip(arrays,actual)],axis=0)
    distance=3.25 if closeup else max(3.25,min(8.,1.3*max(np.linalg.norm(q[t,-3:-1]-center) for (q,_),t in zip(arrays,actual))+1.4))
    panels=[]
    for i,((q,aux),t) in enumerate(zip(arrays,actual)):
        set_pose(q[t]);images=[]
        for top in [False,True]:
            cam.lookat[:]=[center[0],center[1],.22 if top else .6]
            cam.distance=distance;cam.azimuth=90 if top else -70;cam.elevation=-89 if top else -22
            renderer.update_scene(data,camera=cam,scene_option=viewopt)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
            if closeup and env in target_dirs:
                start=np.array([0.,-.8,.025]);end=start+np.r_[target_dirs[env]*.9,0]
                arrow(start,end)
            images.append(Image.fromarray(renderer.render()))
        im=Image.new('RGB',(640,720));im.paste(images[0],(0,0));im.paste(images[1],(0,360))
        draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,53),fill='#101820')
        draw.text((10,5),'BFM + learned latent (frozen actor)' if i==0 else 'Expert policy',font=font,fill='#ffca75' if i==0 else '#90dbff')
        draw.text((10,30),f'env {env} | t={tick*dt:.2f}s | {speed:g}x | recorded Isaac states',font=small,fill='white')
        draw.rectangle((0,685,640,720),fill='#101820')
        foot_v=np.linalg.norm(aux['foot_velocity'][t,:,:2],axis=-1)
        suffix=' | cyan: first-ball target' if closeup else ' | pose replay'
        draw.text((10,693),f'Foot speed L/R: {foot_v[0]:.2f}/{foot_v[1]:.2f} m/s'+suffix,font=small,fill='white')
        if tick>=len(q):draw.text((170,160),'FIRST EPISODE ENDED',font=font,fill='red')
        panels.append(im)
    out=Image.new('RGB',(1280,720));out.paste(panels[0],(0,0));out.paste(panels[1],(640,0))
    return out

if not a.stills_only and not a.skip_full:
    # Fixed env0 full first episode, same wall of simulation time on both sides.
    with imageio.get_writer(a.output/'env0_full_comparison.mp4',fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
        for tick in range(0,max(len(q) for q,_ in clips[a.envs[0]]),2):
            writer.append_data(np.array(frame(a.envs[0],tick)))
            if tick%500==0:print('FULL_VIDEO_FRAME',tick,flush=True)
    print('FULL_VIDEO_COMPLETE',flush=True)
for env in a.envs:
    case=report['cases'][str(env)];times=[case[k]['time_s'] for k in ['candidate','expert']]
    start=max(0,int((min(times)-.8)/dt));end=min(max(len(q) for q,_ in clips[env]),int((max(times)+1.2)/dt))
    if not a.stills_only:
        with imageio.get_writer(a.output/f'env{env}_contact_slow.mp4',fps=12.5,codec='libx264',quality=8,macro_block_size=1) as writer:
            for tick in range(start,end):writer.append_data(np.array(frame(env,tick,True,.25)))
    # Inspect five same-time frames around the first interaction.
    ticks=np.linspace(start,end-1,5).astype(int);sheet=Image.new('RGB',(1280,360*len(ticks)))
    for k,tick in enumerate(ticks):sheet.paste(frame(env,int(tick),True,.25).resize((640,360)),(0,k*360))
    # Keep a separate full-resolution contact image and compact contact sheet.
    frame(env,int(round(min(times)/dt)),True,.25).save(a.output/f'env{env}_contact.png')
    sheet.crop((0,0,640,360*len(ticks))).save(a.output/f'env{env}_sheet.jpg')
    print('CONTACT_VIDEO_COMPLETE',env,flush=True)
renderer.close()
