"""Summarize standing diagnostics without mixing reset states into trajectory metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def describe(values):
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
        minimum=min(values),maximum=max(values)) if values else dict(count=0)


def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text())
    trace=json.loads((directory/'trace.json').read_text())
    n=summary['requested'];failures={r['environment']:r for r in summary['failures']}
    assert len(failures)==len(summary['failures'])==n-summary['survivors']
    assert all(0<=i<n and 0<=r['tick']<summary['steps'] for i,r in failures.items())
    assert trace and trace[0]['tick']==0
    assert all(a['tick']<b['tick'] for a,b in zip(trace,trace[1:]))
    values={k:[] for k in ['height','tilt_rad','displacement_xy','target_mse']}
    for row in trace:
        assert 0<=row['tick']<summary['steps']
        assert len(row['alive'])==n
        for k in values:assert len(row[k])==n
        for i,alive in enumerate(row['alive']):
            # Trace is captured before the action/termination at this tick.
            assert alive==(i not in failures or row['tick']<=failures[i]['tick'])
            if alive:
                for k in values:values[k].append(row[k][i])
    terms={}
    for failure in failures.values():
        for k,v in failure['terms'].items():
            if v:terms[k]=terms.get(k,0)+1
    dt=summary['duration_s']/summary['steps']
    return dict(summary=summary,triggered_failure_terms=terms,
        failure_seconds=describe([(r['tick']+1)*dt for r in failures.values()]),
        failure_time_convention='End of the physics/control step whose zero-based index is tick',
        initial_pose={k:describe(trace[0][k]) for k in ['height','tilt_rad']},
        alive_trace_only={k:describe(v) for k,v in values.items()},
        final_sample_tick=trace[-1]['tick'],
        survivors_last_sample={k:describe([trace[-1][k][i] for i in range(n) if i not in failures]) for k in values},
        trace_sha256=hashlib.sha256((directory/'trace.json').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    status=json.loads(a.study.read_text());assert status['stage']=='completed'
    tag=a.study.name.removesuffix('_status.json')
    result={name:analyze(a.study.parent/f'{tag}_{name}') for name in status['results'] if name!='smoke'}
    assert all(v['summary']==status['results'][k] for k,v in result.items())
    a.output.write_text(json.dumps(dict(models=result,
        scope='Only pre-first-failure samples contribute; survivor samples are explicitly restricted. Standalone source-reset standing, not post-kick recovery.'),indent=2)+'\n')

if __name__=='__main__':main()
