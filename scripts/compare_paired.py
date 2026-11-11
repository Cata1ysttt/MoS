"""Paired first-episode bootstrap intervals; diagnostic metrics are not full success."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--expert',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def read(path):
 rows=json.loads((path/'episodes.json').read_text());return {r['environment']:r for r in rows}
e,c=read(a.expert),read(a.candidate)
if set(e)!=set(c):raise ValueError('Paired comparison requires exactly matching environment ids')
es=json.loads((a.expert/'summary.json').read_text());cs=json.loads((a.candidate/'summary.json').read_text())
seed_e=es.get('seed');seed_c=cs.get('seed')
if seed_e is not None and seed_c is not None and seed_e!=seed_c:raise ValueError('Evaluation seeds differ')
hash_e=es.get('initial_state_sha256');hash_c=cs.get('initial_state_sha256')
if hash_e and hash_c and hash_e!=hash_c:raise ValueError('Initial physical states differ; not a paired comparison')
ids=sorted(e);rng=np.random.default_rng(42);draw=rng.integers(len(ids),size=(10000,len(ids)))
metrics={'kick':lambda x:x['metrics']['quality_kick_success'],'direction_10deg':lambda x:x['metrics']['quality_ball_direction_success_10deg'],'early_termination':lambda x:x['terminated'],'duration_s':lambda x:x['duration']}
report={'trials':len(ids),'seed_verified':seed_e is not None and seed_c is not None,'initial_states_verified':bool(hash_e and hash_c),'warning':'Same IDs and seed are necessary, not sufficient for identical scenario configuration. Kick and direction are diagnostic first-ball metrics, not composite passing success.','metrics':{}}
for name,fn in metrics.items():
 x=np.array([float(fn(c[i]))-float(fn(e[i])) for i in ids]);means=x[draw].mean(1)
 report['metrics'][name]={'expert':float(np.mean([fn(e[i]) for i in ids])),'candidate':float(np.mean([fn(c[i]) for i in ids])),'difference':float(x.mean()),'bootstrap_95pct':np.quantile(means,[.025,.975]).tolist()}
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
