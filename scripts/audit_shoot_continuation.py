"""Audit paired static-shooting continuation with an explicitly labeled expert control."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from audit_sequence_recovery import validate


def audit(planner,expert,protocol_path):
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads(protocol_path.read_text())
    for path,digest in protocol['source_files_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
    results=[json.loads((run/'summary.json').read_text()) for run in [planner,expert]]
    configs=[json.loads((run/'scene_contracts.json').read_text()) for run in [planner,expert]]
    assert configs[0]==configs[1]
    assert results[0]['initial_state_sha256']==results[1]['initial_state_sha256']
    assert results[0]['one_policy_instance'] and not results[0]['expert_loaded']
    assert results[1]['expert_loaded'] and not results[1]['one_policy_instance']
    source=configs[0]['original_shoot'];shared=configs[0]['shared']
    assert shared['terminations']['event_kick_elapsed'] is None
    for key,value in source['terminations'].items():
        if key!='event_kick_elapsed':assert shared['terminations'][key]==value,key
    rows={}
    for name,result in zip(['planner','expert'],results):
        assert result['initial_skill']=='shoot' and result['scene_base']=='shoot'
        assert not result['incoming_pass_command'] and result['continuous_after_kick']
        assert result['post_shoot_recovery_steps'] is None
        assert result['seed']==protocol['seed'] and result['handoff_seed']==protocol['handoff_seed']
        assert result['requested']==protocol['num_envs'] and result['steps']==protocol['steps']
        assert result['checkpoint_sha256']==protocol['source_files_sha256']['runs/planner_history_r3_contact_train/best.pt']
        validate(result)
        first=result['switches'][0]
        assert first['tick']==0 and len(first['environments'])==result['requested']
        dt=shared['sim_dt']*shared['decimation']
        shots={int(k):v for k,v in result['shoot_impacts'].items()}
        resets={row['environment']:row for row in result['first_resets']}
        delays=[(resets[i]['tick']-row['tick'])*dt for i,row in shots.items() if i in resets]
        good={i for i,row in shots.items() if row['direction_error_deg']<=10 and row['preferred_foot'] and row['relative_forward_speed_error']<=.3}
        rows[name]=dict(kicks=len(shots),direction_foot_speed=len(good),
            direction_foot_speed_and_survive=len(good-set(resets)),survivors=result['continuous_survivors'],
            post_shot_failure_delays_s=delays,median_post_shot_failure_s=statistics.median(delays) if delays else None,
            failure_term_counts=dict(Counter(k for row in resets.values() for k,value in row['terms'].items() if value)),
            post_shot_failure_term_counts=dict(Counter(k for i,row in resets.items() if i in shots for k,value in row['terms'].items() if value)))
    assert results[0]['switches'][0]['ball_offset_yaw']==results[1]['switches'][0]['ball_offset_yaw']
    return dict(planner=str(planner),expert=str(expert),protocol=str(protocol_path),
        initial_tensors_equal=True,initial_static_ball_offsets_equal=True,source_hashes_unchanged=True,
        source_normal_post_kick_termination=source['terminations']['event_kick_elapsed'],results=rows,
        scope='Static-ball continuation only. Expert control uses ONNX actions and is not unified autonomous success. Conditional failure medians include task failures such as wrong-foot contact, not only falls; different trajectories prevent a paired causal interpretation of delays.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planner',type=Path,required=True)
    p.add_argument('--expert',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.planner,a.expert,a.protocol)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
