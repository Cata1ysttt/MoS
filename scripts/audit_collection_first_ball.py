"""Audit initial collection cohorts without treating censored episodes as completed.

Teacher cohorts use expert-assisted latent optimization. These are diagnostics,
not paired autonomous policy evaluations.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--collection', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
snapshot = json.loads((a.collection/'current_episode_snapshot.json').read_text())
rows = json.loads((a.collection/'episode_outcomes.json').read_text())
first = {}
for row in rows:
    if row['episode'] == 0:
        assert row['environment'] not in first, 'Duplicate first episode'
        first[row['environment']] = row
cohorts = {}
for teacher in [False, True]:
    ids = [i for i,value in enumerate(snapshot['initial_teacher_executed']) if value == teacher]
    known, completed, pending = [], [], []
    for i in ids:
        if i in first:
            assert first[i]['teacher_executed'] == teacher
            completed.append(i)
            known.append((i,first[i]['metrics']))
        elif snapshot['episode'][i] == 0 and snapshot['first_ball_quality_frozen'] is not None and snapshot['first_ball_quality_frozen'][i]:
            known.append((i,{key:value[i] for key,value in snapshot['metrics'].items()}))
        else:
            assert snapshot['episode'][i] == 0, 'Reset first episode is missing from outcomes'
            pending.append(i)
    cohorts['teacher_assisted' if teacher else 'student'] = dict(
        initial_environments=len(ids),completed_first_episodes=len(completed),
        frozen_first_ball_in_unfinished_episode=len(known)-len(completed),
        pending_first_ball=len(pending),
        known_kick_success=sum(metric['quality_kick_success']>.5 for _,metric in known),
        known_direction_success=sum(metric['quality_ball_direction_success_10deg']>.5 for _,metric in known),
        pending_environment_ids=pending,
        note='Success counts use the initial cohort denominator; pending environments remain explicit')
result = dict(collection=str(a.collection),snapshot_tick=snapshot['tick'],cohorts=cohorts,
              collection_complete=(a.collection/'complete.json').exists(),
              note=__doc__)
a.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
