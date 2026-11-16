"""Count independent training episodes and heading coverage near the first kick.

Repeated frames and later ball resamples do not increase the episode counts.
"""
import argparse
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,nargs='+',required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--radius',type=float,default=.6)
a=p.parse_args()
assert a.radius>0
torch.set_num_threads(4)
results=[]
for path in a.data:
    d=torch.load(path,map_location='cpu',weights_only=True)
    assert d['metadata']['schema'] in ('soccer_state_v2_950','soccer_state_v3_backward_1206','soccer_state_v4_task_history_2406')
    rows=d['rows'];cut=len(rows[0]['x'])*3//4
    seen=set();past_kick=set();episodes={};near={}
    for row in rows:
        for lane in range(cut):
            key=(lane,int(row['episode'][lane]))
            x=row['x'][lane]
            if key not in seen:
                seen.add(key)
                direction=x[-6:-4]
                angle=float(torch.atan2(direction[1],direction[0])*180/torch.pi)
                episodes[key]=dict(heading_degrees=angle,teacher=bool(row['teacher_executed'][lane]))
            if bool(x[940]>.5):past_kick.add(key)
            if key in past_kick or key in near:continue
            if bool(x[939]>.5) and float(x[933:935].norm())<=a.radius:
                direction=x[-6:-4]
                near[key]=dict(heading_degrees=float(torch.atan2(direction[1],direction[0])*180/torch.pi),teacher=bool(row['teacher_executed'][lane]))
    groups={}
    for phase,values in [('episode_start',episodes),('first_near_before_kick',near)]:
        groups[phase]={}
        for teacher in [False,True]:
            selected=[v['heading_degrees'] for v in values.values() if v['teacher']==teacher]
            histogram=torch.histc(torch.tensor(selected,dtype=torch.float32),bins=12,min=-180,max=180).int().tolist() if selected else [0]*12
            groups[phase]['teacher' if teacher else 'student']=dict(episodes=len(selected),heading_bin_counts=histogram)
    results.append(dict(data=str(path),seed=d['metadata']['seed'],skill=d['metadata']['skill'],training_lanes=cut,groups=groups))
report=dict(radius=a.radius,heading_bin_edges_degrees=list(range(-180,181,30)),results=results,
            note='Training lanes only; each (dataset,lane,episode) counted once. First near heading is body-yaw-relative at first observed approach before any observed kick. Different phase headings can differ because the robot turns; incomplete episodes remain included as coverage, not success.')
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
