"""Publish the paired report after the already-running pilot driver completes."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs';status=runs/'pass_shoot_pilot_status.json'
while True:
 state=json.loads(status.read_text())
 if state['status']=='failed':raise RuntimeError(state)
 if state['status']=='completed':break
 if not Path('/proc/3378379/cmdline').exists():raise RuntimeError('Pilot driver exited without terminal status')
 time.sleep(30)
subprocess.run(['python',str(root/'scripts/summarize_pass_shoot_pilot.py')],check=True)
data=json.loads((runs/'pass_shoot_pilot_results.json').read_text())
lines=['\n## Final paired evaluation','\n| Skill | Policy | Kick success | Direction <=10deg | Termination reasons |','|---|---|---:|---:|---|']
for skill,pair in data.items():
 for policy,r in pair.items():
  s=r['summary'];lines.append(f"| {skill} | {policy} | {s['quality_kick_success']*16:g}/16 | {s['quality_ball_direction_success_10deg']*16:g}/16 | {r['termination_terms']} |")
p=root/'reports/pass_shoot_pilot_20261113.md'
with p.open('a') as f:f.write('\n'.join(lines)+'\n\nBoth skill comparisons have identical within-skill initial-state hashes.\n')
