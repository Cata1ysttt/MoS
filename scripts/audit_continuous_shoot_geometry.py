"""Describe recorded shooting outcomes by initial ball geometry, without new rollouts."""
import hashlib
import json
import math
from pathlib import Path
from summarize_continuous_shoot_ppo import outcomes

ROOT=Path(__file__).resolve().parents[1]
GROUPS={
    'full_history_development':[f'planner_post_pass_settle_r1_{s}_control' for s in [816201,816202]],
    'settling_development':[f'planner_post_pass_settle_r1_{s}_candidate' for s in [816201,816202]],
    'full_history_completed_confirmation':[f'planner_new_ball_task_confirmation_r1_{s}_candidate' for s in [816601,816602,816603,816604]],
    'corrected_history_expert_diagnostic':[f'planner_shoot_expert_history_study_r1_{s}_expert_full' for s in [816201,816202]],
}


def summarize(rows):
    result=dict(handoffs=len(rows),shots=sum(r['shot'] for r in rows),precise_shots=sum(r['precise_shot'] for r in rows),
        full_success=sum(r['full_success'] for r in rows),no_shot_failure=sum(r['no_shot_failure'] for r in rows),
        no_shot_survivor=sum(not r['shot'] and not r['failed'] for r in rows),
        wrong_foot_shots=sum(r['wrong_foot'] for r in rows),direction_bad_shots=sum(r['direction_bad'] for r in rows),
        speed_bad_shots=sum(r['speed_bad'] for r in rows))
    assert result['shots']+result['no_shot_failure']+result['no_shot_survivor']==len(rows)
    return result


def main():
    output=ROOT/'reports/planner_continuous_shoot_geometry.json';assert not output.exists()
    result=dict(scope='Retrospective descriptive stratification of existing completed runs. Confirmation outcomes have already been inspected; this is not a fresh validation or a causal comparison. Initial ball geometry does not describe contact geometry. All original environments remain in overall denominators; strata include only actual handoffs. Expert groups are diagnostic and not unified-policy scores.',groups={})
    for name,directories in GROUPS.items():
        rows=[];requested=0;sources={}
        for directory in directories:
            path=ROOT/'runs'/directory/'summary.json';source=path.read_bytes();s=json.loads(source)
            out=outcomes(s);requested+=out['requested'];sources[directory]=hashlib.sha256(source).hexdigest()
            assert s['steps']==1000 and s['failed_lane_control']=='zero'
            assert s['expert_loaded']==('expert_diagnostic' in name)
            failed={r['environment'] for r in s['first_resets']}
            seen=set()
            for event in s['switches']:
                for j,i in enumerate(event['environments']):
                    assert i not in seen;seen.add(i)
                    x,y,_=event['ball_offset_yaw'][j];impact=s['shoot_impacts'].get(str(i))
                    precise=bool(impact and impact['preferred_foot'] and impact['direction_error_deg']<=10 and impact['relative_forward_speed_error']<=.3)
                    rows.append(dict(seed=s['seed'],environment=i,ball_x_m=x,ball_y_m=y,distance_m=math.hypot(x,y),
                        angle_deg=math.degrees(math.atan2(y,x)),shot=impact is not None,precise_shot=precise,
                        full_success=i in out['full_success_environments'],failed=i in failed,
                        no_shot_failure=impact is None and i in failed,
                        wrong_foot=bool(impact and not impact['preferred_foot']),
                        direction_bad=bool(impact and impact['direction_error_deg']>10),
                        speed_bad=bool(impact and impact['relative_forward_speed_error']>.3)))
            assert len(seen)==out['handoffs']
        partitions={
            'front_vs_behind':{'front':lambda r:r['ball_x_m']>=0,'behind':lambda r:r['ball_x_m']<0},
            'left_vs_right':{'left':lambda r:r['ball_y_m']>=0,'right':lambda r:r['ball_y_m']<0},
            'distance':{'at_most_2m':lambda r:r['distance_m']<=2,'over_2m':lambda r:r['distance_m']>2},
            'quadrant':{label:(lambda r,front=front,left=left:(r['ball_x_m']>=0)==front and (r['ball_y_m']>=0)==left)
                for label,front,left in [('front_left',True,True),('front_right',True,False),('behind_left',False,True),('behind_right',False,False)]},
        }
        strata={key:{label:summarize([r for r in rows if predicate(r)]) for label,predicate in partition.items()} for key,partition in partitions.items()}
        assert all(sum(v['handoffs'] for v in partition.values())==len(rows) for partition in strata.values())
        result['groups'][name]=dict(original_requested=requested,no_handoff=requested-len(rows),overall=summarize(rows),strata=strata,source_sha256=sources,rows=rows)
    output.write_text(json.dumps(result,indent=2)+'\n')
    for name,group in result['groups'].items():
        print(name,'requested',group['original_requested'])
        for key in ['front_vs_behind','left_vs_right','distance']:
            print(key,{k:{metric:v[metric] for metric in ['handoffs','shots','precise_shots','no_shot_failure']} for k,v in group['strata'][key].items()})


if __name__=='__main__':main()
