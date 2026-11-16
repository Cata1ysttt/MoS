"""Check sequence accounting and compare the observed prefix before recovery starts."""
import argparse
import json
from pathlib import Path


def validate(result):
    n=result['requested']
    passed={int(k):v for k,v in result['pass_impacts'].items()}
    shot={int(k):v for k,v in result['shoot_impacts'].items()}
    resets={v['environment']:v for v in result['first_resets']}
    assert len(resets)==len(result['first_resets'])
    assert set(passed)|set(shot)|set(resets) <= set(range(n))
    switched={}
    for event in result['switches']:
        assert event['robot_and_history_preserved']
        for i in event['environments']:
            assert 0<=i<n and i not in switched
            assert i not in resets or event['tick']<=resets[i]['tick']
            if result['initial_skill']=='pass':
                assert i in passed and event['tick']>=passed[i]['tick']+result['recovery_steps']
            switched[i]=event['tick']
    assert result['switched']==len(switched)
    for i,row in shot.items():assert i in switched and row['tick']>=switched[i]
    both=set(passed)&set(shot)
    accurate={i for i in both if passed[i]['direction_error_deg']<=10 and shot[i]['direction_error_deg']<=10}
    usable={i for i in accurate if passed[i]['preferred_foot'] and shot[i]['preferred_foot']}
    speed={i for i in usable if passed[i]['relative_forward_speed_error']<=.3 and shot[i]['relative_forward_speed_error']<=.3}
    expected=dict(both_observed_kicks=len(both),both_direction_le_10deg=len(accurate),
        both_direction_and_preferred_foot=len(usable),both_direction_preferred_foot_and_speed_30pct=len(speed),
        both_accurate_and_continuous_through_end=len(usable-set(resets)),
        both_kicks_and_no_reset_through_end=len(both-set(resets)),continuous_survivors=n-len(resets))
    for key,value in expected.items():assert result[key]==value,(key,result[key],value)
    for records in [passed,shot]:
        for i,row in records.items():
            assert i not in resets or row['tick']<=resets[i]['tick'],'Counted impact after first reset'
    recovered=set()
    delay=result.get('post_shoot_recovery_steps')
    for event in result.get('recovery_switches',[]):
        assert delay is not None and event['robot_and_history_preserved']
        for i in event['environments']:
            assert i not in recovered and i in shot
            assert event['tick']>=shot[i]['tick']+delay
            assert i not in resets or event['tick']<=resets[i]['tick']
            recovered.add(i)
    assert result.get('recovery_started',0)==len(recovered)
    return dict(derived_counts=expected,recovered=sorted(recovered),
                recovered_without_reset=sorted(recovered-set(resets)),reset_environments=sorted(resets))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists(),'Preserve previous audit'
    base,new=[json.loads((path/'summary.json').read_text()) for path in [a.baseline,a.recovery]]
    assert base.get('post_shoot_recovery_steps') is None
    assert new['post_shoot_recovery_steps'] is not None
    for key in ['checkpoint_sha256','seed','requested','steps','scene_base','initial_skill','transition_steps','recovery_steps','one_policy_instance','expert_loaded']:
        assert base[key]==new[key],key
    protocols=[json.loads((path/'protocol.json').read_text()) for path in [a.baseline,a.recovery]]
    ignored={'post_shoot_recovery_steps','post_shoot_recovery_skill','post_shoot_command_preserved'}
    assert {k:v for k,v in protocols[0].items() if k not in ignored}=={k:v for k,v in protocols[1].items() if k not in ignored}
    counts={name:validate(result) for name,result in [('baseline',base),('recovery',new)]}
    first=min((row['tick'] for row in new['recovery_switches']),default=new['steps'])
    for key in ['pass_impacts','shoot_impacts']:
        left={k:v for k,v in base[key].items() if v['tick']<first}
        right={k:v for k,v in new[key].items() if v['tick']<first}
        assert left==right,('Observed prefix differs',key)
    for key in ['switches','first_resets']:
        assert [v for v in base[key] if v['tick']<first]==[v for v in new[key] if v['tick']<first],key
    report=dict(baseline=str(a.baseline),recovery=str(a.recovery),seed=new['seed'],checkpoint_sha256=new['checkpoint_sha256'],
        first_recovery_tick=first if new['recovery_switches'] else None,observed_prefix_equal=True,results=counts,
        scope='Validates impact/reset accounting and exact recorded events before recovery; does not establish equality of all unrecorded simulator states or generalization to other seeds.')
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
