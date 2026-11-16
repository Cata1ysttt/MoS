"""Verify planner-only recovery speed intervention and unchanged pre-intervention events."""
import argparse
import json
from pathlib import Path
from audit_sequence_recovery import validate


def audit(baseline,candidate):
    runs=[baseline,candidate]
    results=[json.loads((run/'summary.json').read_text()) for run in runs]
    protocols=[json.loads((run/'protocol.json').read_text()) for run in runs]
    base,new=results
    assert base.get('post_shoot_recovery_speed') is None
    assert new['post_shoot_recovery_speed']>0
    assert base['post_shoot_recovery_steps']==new['post_shoot_recovery_steps']==10
    assert base['initial_state_sha256']==new['initial_state_sha256']
    for key in ['checkpoint_sha256','seed','requested','steps','scene_base','initial_skill',
                'transition_steps','recovery_steps','handoff_seed','one_policy_instance','expert_loaded']:
        assert base[key]==new[key],key
    assert base['one_policy_instance'] and not base['expert_loaded']
    assert {k:v for k,v in protocols[0].items() if k!='post_shoot_recovery_speed'}=={
        k:v for k,v in protocols[1].items() if k!='post_shoot_recovery_speed'}
    assert json.loads((baseline/'scene_contracts.json').read_text())==json.loads((candidate/'scene_contracts.json').read_text())
    assert protocols[1]['post_shoot_command_preserved']
    assert new['recovery_switches'],'No intervention was exercised'
    first=min(row['tick'] for row in new['recovery_switches'])
    for key in ['pass_impacts','shoot_impacts']:
        assert {k:v for k,v in base[key].items() if v['tick']<first}=={
            k:v for k,v in new[key].items() if v['tick']<first},key
    for key in ['switches','first_resets']:
        assert [v for v in base[key] if v['tick']<first]==[v for v in new[key] if v['tick']<first],key
    return dict(baseline=str(baseline),candidate=str(candidate),first_intervention_tick=first,
        initial_state_equal=True,observed_prefix_equal=True,scene_and_task_command_unchanged=True,
        results={name:validate(result) for name,result in zip(['baseline','candidate'],results)},
        scope='Only planner recovery speed changes. Shooting impacts retain their original task speed. Same initial tensors and observed prefix do not prove identical subsequent physical trajectories. Reused development seed; not independent validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.baseline,a.candidate)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
