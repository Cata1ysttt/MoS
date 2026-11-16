"""Audit complete four-arm PD diagnosis; no claims from partial results."""
import json
from pathlib import Path
from collections import Counter
import statistics

root=Path(__file__).resolve().parents[1]/'runs'
report={'scope':'Existing-policy PD sensitivity, not equal-budget retraining or native dynamics reproduction.','arms':{}}
initial=[];balls=[]
for actor in ['expert','ufo']:
    for mode in ['original','native']:
        name=f'ufo_pd_{actor}_{mode}16';folder=root/name
        summary=json.loads((folder/'summary.json').read_text())
        episodes=json.loads((folder/'episodes.json').read_text())
        assert summary['completed']==summary['requested']==16
        assert {r['environment'] for r in episodes}==set(range(16))
        initial.append(summary['initial_state_sha256'])
        balls.append(json.loads((folder/'ball_runtime_audit.json').read_text()))
        if mode=='native':
            pd=json.loads((folder/'pd_override.json').read_text())
            assert pd['mode']=='ufo_native_pd_only'
            assert len(pd['after']['kp'])==16
        reasons=Counter(k for row in episodes for k,v in row['termination_terms'].items() if v)
        report['arms'][name]={'kick_count':sum(r['metrics']['quality_kick_success'] for r in episodes),
            'direction_count':sum(r['metrics']['quality_ball_direction_success_10deg'] for r in episodes),
            'early_termination':summary['terminated'],'median_duration_s':statistics.median(r['duration'] for r in episodes),
            'termination_terms':dict(reasons)}
assert len(set(initial))==1,'Initial states do not match'
assert all(b==balls[0] for b in balls),'Ball parameters differ'
report['initial_state_and_ball_match']=True
path=root/'ufo_pd_comparison.json';path.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
