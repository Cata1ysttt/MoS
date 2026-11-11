"""Summarize measured task metrics, never equate any termination to a fall."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted((root/'runs').glob('*/episodes.json')):
 data=json.loads(p.read_text())
 if not data:continue
 counts=collections.Counter(k for e in data for k,v in e.get('termination_terms',{}).items() if v)
 kicked=[e for e in data if e['metrics'].get('quality_kick_success',0)>.5]
 def conditional(key):
  values=[e['metrics'][key] for e in kicked if key in e['metrics']]
  return sum(values)/len(values) if values else None
 n=len(data);m=lambda key:sum(e['metrics'].get(key,0) for e in data)/n
 rows.append(dict(run=p.parent.name,trials=n,kick_rate=m('quality_kick_success'),direction_10deg=m('quality_ball_direction_success_10deg'),mean_duration_s=sum(e['duration'] for e in data)/n,terminated=sum(e['terminated'] for e in data),termination_terms=dict(counts),kicked_trials=len(kicked),conditional_first_ball_direction_error_deg=conditional('quality_ball_direction_error_deg_after_kick'),conditional_first_ball_relative_speed_error=conditional('quality_ball_speed_error_after_kick'),conditional_first_ball_forward_speed_mps=conditional('quality_ball_forward_speed_after_kick')))
(root/'runs/evaluation_comparison.json').write_text(json.dumps(rows,indent=2))
for row in rows:print(json.dumps(row))
