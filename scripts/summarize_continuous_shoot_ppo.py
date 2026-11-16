"""Reconstruct continuous PPO outcomes from completed paired evaluation records.

Refuses incomplete drivers and keeps all first-failure environments in the
original denominator. The report does not automatically promote a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
from audit_sequence_recovery import validate


def outcomes(summary):
    validate(summary)
    passed={int(k):v for k,v in summary['pass_impacts'].items()}
    shot={int(k):v for k,v in summary['shoot_impacts'].items()}
    switched={i for event in summary['switches'] for i in event['environments']}
    resets={r['environment']:r for r in summary['first_resets']}
    failure=dict(before_pass=0,after_pass_before_handoff=0,after_handoff_before_shot=0,after_shot=0)
    terms={}
    for i,row in resets.items():
        stage='after_shot' if i in shot else 'after_handoff_before_shot' if i in switched else 'after_pass_before_handoff' if i in passed else 'before_pass'
        failure[stage]+=1
        for term,triggered in row['terms'].items():
            if triggered:terms[term]=terms.get(term,0)+1
    assert sum(failure.values())+summary['continuous_survivors']==summary['requested']
    def precise(row):
        return row['preferred_foot'] and row['direction_error_deg']<=10 and row['relative_forward_speed_error']<=.3
    both=set(passed)&set(shot)
    precise_pass={i for i,row in passed.items() if precise(row)}
    precise_shot={i for i,row in shot.items() if precise(row)}
    full=(precise_pass&precise_shot)-set(resets)
    assert len(full)==summary['both_direction_foot_speed_and_continuous_through_end']
    return dict(requested=summary['requested'],passed=len(passed),handoffs=len(switched),shots=len(shot),
        both_kicks=len(both),precise_pass=len(precise_pass),precise_shot=len(precise_shot),
        wrong_foot_shots=sum(not row['preferred_foot'] for row in shot.values()),
        direction_bad_shots=sum(row['direction_error_deg']>10 for row in shot.values()),
        speed_bad_shots=sum(row['relative_forward_speed_error']>.3 for row in shot.values()),
        slow_shots=sum(row['forward_speed_mps']<.7*row['target_speed_mps'] for row in shot.values()),
        fast_shots=sum(row['forward_speed_mps']>1.3*row['target_speed_mps'] for row in shot.values()),
        both_precise=len(precise_pass&precise_shot),full_success=len(full),survivors=summary['continuous_survivors'],
        failures=failure,triggered_failure_terms=terms,full_success_environments=sorted(full))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default='planner_continuous_shoot_ppo_r1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert a.tag.replace('_','').isalnum() and not a.output.exists()
    root=Path(__file__).resolve().parents[1];run=root/'runs'/a.tag
    state=json.loads((root/'runs'/f'{a.tag}_status.json').read_text())
    assert state['stage']=='completed','Wait for all preselected evaluations'
    complete=json.loads((run/'complete.json').read_text())
    checkpoint_hash=hashlib.sha256((run/'final.pt').read_bytes()).hexdigest()
    assert checkpoint_hash==complete['checkpoint_sha256']==state['checkpoint_sha256']
    audit=json.loads((root/'runs'/f'{a.tag}_audit.json').read_text())
    assert audit['checkpoint_sha256']==checkpoint_hash
    result=dict(checkpoint_sha256=checkpoint_hash,training=complete,export_audit=audit,native={},sequences={},
        scope='All preselected paired seeds, original failure denominator, zero actions after first failure; no automatic promotion.')
    for skill,pair in state['results']['native'].items():
        destination=root/'runs'/f'{a.tag}_{skill}_eval'
        baseline=root/'runs'/f'planner_history_r3_contact_{skill}_eval'
        for name in ['episodes.json','first_kick_velocity.json']:
            assert json.loads((destination/name).read_text())==json.loads((baseline/name).read_text()),(skill,name)
        result['native'][skill]=dict(episodes_and_first_impacts_exact=True,summary=pair['candidate'])
    totals={label:{} for label in ['baseline','candidate']}
    numeric=['requested','passed','handoffs','shots','both_kicks','precise_pass','precise_shot','both_precise','full_success','survivors','wrong_foot_shots','direction_bad_shots','speed_bad_shots','slow_shots','fast_shots']
    for seed in state['seeds']:
        pair=state['results'][str(seed)];entry={}
        assert all(pair['audit'][key] for key in ['initial_tensors_equal','protocol_and_scene_equal','observed_prefix_equal'])
        for label in totals:
            summary=pair[label]['summary']
            raw=json.loads((root/'runs'/f'{a.tag}_{seed}_{label}'/'summary.json').read_text())
            assert summary==raw and summary['failed_lane_control']=='zero'
            entry[label]=row=outcomes(summary)
            for key in numeric:totals[label][key]=totals[label].get(key,0)+row[key]
            failures=totals[label].setdefault('failures',{})
            for key,value in row['failures'].items():failures[key]=failures.get(key,0)+value
        entry['audit']=pair['audit'];result['sequences'][str(seed)]=entry
    result['totals']=totals
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
