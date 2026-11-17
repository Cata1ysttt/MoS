"""Separate recorded wrong-foot shots from wrong-foot failures without a scored shot."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();study=json.loads(a.study.read_text())
    assert study['stage']=='completed'
    result={label:dict(requested=0,wrong_foot_shots=[],wrong_foot_terminations_without_shot=[],all_wrong_foot_terminations=0)
        for label in study['results']['protocol']['models']}
    for seed,e in study['results']['sequences'].items():
        assert not e.get('audit_pending',False)
        for label,r in e['models'].items():
            sm=r['summary'];out=result[label];out['requested']+=sm['requested']
            shots={int(i):v for i,v in sm['shoot_impacts'].items()}
            failures={v['environment']:v for v in sm['first_resets']}
            recovery={i:v['tick'] for v in sm['recovery_switches'] for i in v['environments']}
            handoffs={i:dict(tick=v['tick'],offset=v['ball_offset_yaw'][j]) for v in sm['switches'] for j,i in enumerate(v['environments'])}
            def row(i):
                f=failures.get(i);h=handoffs.get(i)
                value=dict(seed=int(seed),environment=i,recovery_entered=i in recovery)
                if h:
                    x,y,_=h['offset'];value.update(handoff_tick=h['tick'],handoff_distance_xy=math.hypot(x,y),
                        handoff_ball_side='left' if y>0 else 'right_or_center',handoff_ball_angle_deg=math.degrees(math.atan2(y,x)))
                if f:
                    value.update(failure_tick=f['tick'],failure_terms=[k for k,v in f['terms'].items() if v],
                        failure_root_height=f['root_height'],failure_root_tilt_rad=f['root_tilt_rad'])
                    if h:value['handoff_to_failure_seconds']=(f['tick']-h['tick'])*.02
                return value
            for i,shot in shots.items():
                if shot['preferred_foot']:continue
                v=row(i);v['shot']=shot
                v['direction_and_speed_valid_without_foot']=shot['direction_error_deg']<=10 and shot['relative_forward_speed_error']<=.3
                if i in handoffs:v['handoff_to_shot_seconds']=(shot['tick']-handoffs[i]['tick'])*.02
                out['wrong_foot_shots'].append(v)
            for i,f in failures.items():
                if not f['terms'].get('wrong_foot_kick',False):continue
                out['all_wrong_foot_terminations']+=1
                if i not in shots:
                    v=row(i);v['phase']='shoot_approach' if i in handoffs else 'pass'
                    out['wrong_foot_terminations_without_shot'].append(v)
    for out in result.values():
        rows=out['wrong_foot_shots']
        out['wrong_foot_shot_count']=len(rows)
        out['direction_and_speed_valid_without_foot']=sum(v['direction_and_speed_valid_without_foot'] for v in rows)
        out['all_wrong_foot_shots_failed_before_recovery']=all('wrong_foot_kick' in v.get('failure_terms',[]) and not v['recovery_entered'] for v in rows)
    report=dict(models=result,scope='Descriptive completed-development audit. Preferred-foot flag and original failure terms retained; low-speed/unrecorded impacts are not reclassified as scored shots. Handoff geometry does not establish contact geometry or causation.')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{x:y for x,y in v.items() if not isinstance(y,list)} for k,v in result.items()}))

if __name__=='__main__':main()
