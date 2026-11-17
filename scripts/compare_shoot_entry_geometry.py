"""Describe native-reset versus continuous-handoff ball geometry; no causal/OOD claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch


def describe(rows):
    result={'count':len(rows)}
    for key in ['distance_xy','absolute_angle_deg']:
        values=[r[key] for r in rows]
        result[key]=dict(mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values)) if values else {}
    result['behind']=sum(r['absolute_angle_deg']>90 for r in rows)
    result['distance_at_least_2_5m']=sum(r['distance_xy']>=2.5 for r in rows)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    initial=torch.load(a.native/'initial_state.pt',map_location='cpu',weights_only=True)
    assert initial.ndim==2 and initial.shape[1]==55  # Robot root13, joints29, ball root13.
    native_summary=json.loads((a.native/'summary.json').read_text())
    digest=hashlib.sha256(initial.numpy().tobytes()).hexdigest()
    assert digest==native_summary['initial_state_sha256']
    q=initial[:,3:7];yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
    delta=initial[:,42:44]-initial[:,:2]
    dx=torch.cos(yaw)*delta[:,0]+torch.sin(yaw)*delta[:,1]
    dy=-torch.sin(yaw)*delta[:,0]+torch.cos(yaw)*delta[:,1]
    native=[dict(environment=i,distance_xy=float(delta[i].norm()),absolute_angle_deg=abs(math.degrees(math.atan2(float(dy[i]),float(dx[i]))))) for i in range(len(initial))]
    study=json.loads(a.study.read_text());assert study['stage']=='completed'
    groups={label:[] for label in study['results']['protocol']['models']}
    for seed,case in study['results']['sequences'].items():
        assert not case.get('audit_pending',False)
        for label,entry in case['models'].items():
            s=entry['summary'];shot={int(k) for k in s['shoot_impacts']};failed={v['environment'] for v in s['first_resets']}
            for event in s['switches']:
                for j,i in enumerate(event['environments']):
                    x,y,_=event['ball_offset_yaw'][j]
                    distance=math.hypot(x,y)
                    assert abs(distance-event['distance_to_robot'][j])<1e-4
                    groups[label].append(dict(seed=int(seed),environment=i,distance_xy=distance,
                        absolute_angle_deg=abs(math.degrees(math.atan2(y,x))),reached_shot=i in shot,
                        failed_before_shot=i in failed and i not in shot))
    result=dict(native=dict(source=str(a.native),initial_state_sha256=digest,summary=describe(native),rows=native),
        continuous={k:dict(summary=describe(v),reached_shot=describe([r for r in v if r['reached_shot']]),
            failed_before_shot=describe([r for r in v if r['failed_before_shot']]),rows=v) for k,v in groups.items()},
        scope='Native reference is one observed 16-environment reset sample, not the full expert training distribution. Handoff statistics condition on reaching handoff; full failure denominators remain in the study. Descriptive geometry only; no causal or out-of-distribution classification, no changed handoff rule.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native=result['native']['summary'],continuous={k:v['summary'] for k,v in result['continuous'].items()})))

if __name__=='__main__':main()
