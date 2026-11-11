"""Report paired direction and strength diagnostics without redefining success."""
import argparse
import json
from pathlib import Path
import numpy as np

def report(reference, candidate):
    def read(path):
        summary=json.loads((path/'summary.json').read_text())
        rows=json.loads((path/'episodes.json').read_text())
        if len(rows)!=summary['requested'] or len({r['environment'] for r in rows})!=len(rows):
            raise ValueError('Incomplete or duplicated episodes')
        return summary,{r['environment']:r for r in rows}
    a,ar=read(reference);b,br=read(candidate)
    if a['initial_state_sha256']!=b['initial_state_sha256'] or a['seed']!=b['seed'] or ar.keys()!=br.keys():
        raise ValueError('Unmatched scenarios')
    if json.loads((reference/'ball_runtime_audit.json').read_text())!=json.loads((candidate/'ball_runtime_audit.json').read_text()):
        raise ValueError('Ball settings differ')
    ids=sorted(ar);rng=np.random.default_rng(42);draw=rng.integers(len(ids),size=(10000,len(ids)))
    def composite(r,tolerance):
        m=r['metrics']
        return bool(m['quality_kick_success'] and m['quality_ball_direction_success_10deg'] and m['quality_ball_speed_error_after_kick']<=tolerance)
    metrics={'direction_10deg':lambda r:r['metrics']['quality_ball_direction_success_10deg'],
             'relative_forward_speed_error':lambda r:r['metrics']['quality_ball_speed_error_after_kick'],
             'early_termination':lambda r:r['terminated']}
    for t in (.25,.5):metrics[f'direction_and_relative_speed_error_le_{t}']=lambda r,t=t:composite(r,t)
    result={'reference':str(reference),'candidate':str(candidate),'episodes':len(ids),'paired_state_and_ball_audit':True,
            'interpretation':'Frozen first-ball snapshot diagnostics. Speed is relative forward-speed error, not m/s. 25% and 50% are sensitivity thresholds, not official success criteria. Episode termination can follow later balls.','metrics':{}}
    for name,fn in metrics.items():
        x=np.array([float(fn(ar[i])) for i in ids]);y=np.array([float(fn(br[i])) for i in ids]);d=y-x
        result['metrics'][name]={'reference':float(x.mean()),'candidate':float(y.mean()),'difference':float(d.mean()),'paired_bootstrap_95pct':np.quantile(d[draw].mean(1),[.025,.975]).tolist()}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=report(a.reference,a.candidate);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
