"""Describe handoff conditions and failure timing from a completed frozen study."""
import argparse
import json
import math
from pathlib import Path
import statistics
from summarize_continuous_shoot_ppo import outcomes


def distribution(values):
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        minimum=min(values),maximum=max(values)) if values else dict(count=0)


def describe(rows):
    keys=['height','speed_xy','angular_speed','tilt_rad','ball_distance','absolute_ball_angle_deg',
          'handoff_to_shot_s','handoff_to_failure_s','shot_to_failure_s','recovery_to_failure_s',
          'failure_root_height','failure_root_tilt_rad']
    failure_terms={}
    for row in rows:
        for term in row.get('failure_terms',[]):failure_terms[term]=failure_terms.get(term,0)+1
    return dict(states=len(rows),triggered_failure_terms=failure_terms,
        **{key:distribution([r[key] for r in rows if key in r]) for key in keys})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    status=json.loads(args.study.read_text());assert status['stage']=='completed'
    results=status['results']
    entries=list(results['sequences'].values());assert entries,'No completed sequence evaluations'
    labels=set(entries[0]['models'])
    assert labels and all(set(entry['models'])==labels for entry in entries),'Incomplete model pairs'
    if 'models' in results['protocol']:assert set(results['protocol']['models'])==labels
    all_rows={label:[] for label in sorted(labels)}
    paired=[]
    for seed,entry in results['sequences'].items():
        assert not entry.get('audit_pending',False)
        models={}
        for label,record in entry['models'].items():
            summary=record['summary'];assert outcomes(summary)==record['outcomes']
            shot={int(i):v for i,v in summary['shoot_impacts'].items()}
            reset={v['environment']:v for v in summary['first_resets']}
            recovery={i:v['tick'] for v in summary['recovery_switches'] for i in v['environments']}
            rows={}
            for event in summary['switches']:
                for j,i in enumerate(event['environments']):
                    dx,dy,_=event['ball_offset_yaw'][j]
                    row=dict(seed=int(seed),environment=i,handoff_tick=event['tick'],
                        height=event['root_height'][j],speed_xy=event['root_speed_xy'][j],
                        angular_speed=event['root_angular_speed'][j],tilt_rad=event['root_tilt_rad'][j],
                        ball_distance=event['distance_to_robot'][j],absolute_ball_angle_deg=abs(math.degrees(math.atan2(dy,dx))),
                        shot=i in shot,failed=i in reset,
                        full_success=i in record['outcomes']['full_success_environments'])
                    if i in shot:row['handoff_to_shot_s']=(shot[i]['tick']-event['tick'])*.02
                    if i in reset:
                        row['failure_root_height']=reset[i]['root_height']
                        row['failure_root_tilt_rad']=reset[i]['root_tilt_rad']
                        row['failure_terms']=[k for k,v in reset[i]['terms'].items() if v]
                        row['handoff_to_failure_s']=(reset[i]['tick']-event['tick'])*.02
                        if i in shot:row['shot_to_failure_s']=(reset[i]['tick']-shot[i]['tick'])*.02
                        if i in recovery:row['recovery_to_failure_s']=(reset[i]['tick']-recovery[i])*.02
                    rows[i]=row;all_rows[label].append(row)
            models[label]=rows
        common=set.intersection(*(set(rows) for rows in models.values()))
        paired.append(dict(seed=int(seed),common_handoff_environments=sorted(common),
            shots={label:sum(rows[i]['shot'] for i in common) for label,rows in models.items()},
            full_success={label:sum(rows[i]['full_success'] for i in common) for label,rows in models.items()}))
    analysis={}
    for label,rows in all_rows.items():
        groups={'all':rows,'reached_shot':[r for r in rows if r['shot']],
            'failed_before_shot':[r for r in rows if not r['shot'] and r['failed']],
            'survived_without_shot':[r for r in rows if not r['shot'] and not r['failed']],
            'failed_after_shot':[r for r in rows if r['shot'] and r['failed']],
            'survived_after_shot':[r for r in rows if r['shot'] and not r['failed']]}
        opportunities={}
        for name,predicate in [('front',lambda r:r['absolute_ball_angle_deg']<=90),
            ('behind',lambda r:r['absolute_ball_angle_deg']>90),
            ('distance_below_1_5m',lambda r:r['ball_distance']<1.5),
            ('distance_1_5_to_2_5m',lambda r:1.5<=r['ball_distance']<2.5),
            ('distance_above_2_5m',lambda r:r['ball_distance']>=2.5)]:
            subset=[r for r in rows if predicate(r)]
            opportunities[name]=dict(handoffs=len(subset),shots=sum(r['shot'] for r in subset),
                full_success=sum(r['full_success'] for r in subset))
        analysis[label]=dict(groups={k:describe(v) for k,v in groups.items()},opportunities=opportunities,rows=rows)
    args.output.write_text(json.dumps(dict(totals=results['totals'],handoff_analysis=analysis,common_handoffs=paired,
        scope='Descriptive frozen-evaluation diagnostics; different policies reach different body states. No causal claim or new switching threshold inferred.'),indent=2)+'\n')


if __name__=='__main__':main()
