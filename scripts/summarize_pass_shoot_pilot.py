"""Summarize paired first-episode metrics without calling normal kick-end a fall."""
import collections,json
from pathlib import Path
r=Path(__file__).resolve().parents[1]/'runs';result={}
for skill in ['pass','shoot']:
 pair={}
 for policy in ['expert','joint']:
  path=r/f'pass_shoot_{skill}_{policy}_eval_v1'
  summary=json.loads((path/'summary.json').read_text())
  episodes=json.loads((path/'episodes.json').read_text())
  assert summary['completed']==summary['requested']==16
  reasons=collections.Counter(k for x in episodes for k,v in x['termination_terms'].items() if v)
  pair[policy]=dict(summary=summary,termination_terms=dict(reasons))
 assert pair['expert']['summary']['initial_state_sha256']==pair['joint']['summary']['initial_state_sha256']
 result[skill]=pair
(r/'pass_shoot_pilot_results.json').write_text(json.dumps(result,indent=2))
for skill,pair in result.items():
 for policy,row in pair.items():
  s=row['summary'];print(skill,policy,'kick',s['quality_kick_success'],'direction',s['quality_ball_direction_success_10deg'],'termination_terms',row['termination_terms'])
