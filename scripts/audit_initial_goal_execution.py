"""Check recovery activation counts against live-lane event and failure times."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(directory):
    summary=json.loads((directory/'summary.json').read_text())
    records=json.loads((directory/'predicted_target_solver.json').read_text())
    assert summary['standing_recovery_mode']=='initial_backward_goal'
    assert len(records)==summary['steps']
    failures={v['environment']:v['tick'] for v in summary['first_resets']}
    starts={i:v['tick'] for v in summary['recovery_switches'] for i in v['environments']}
    assert len(starts)==sum(len(v['environments']) for v in summary['recovery_switches'])
    expected=[0]*summary['steps'];durations={}
    for i,start in starts.items():
        # Action is applied before termination at that tick; include the failure step.
        end=min(summary['steps'],failures.get(i,summary['steps']-1)+1)
        assert 0<=i<summary['requested'] and 0<=start<end
        durations[str(i)]=end-start
        for tick in range(start,end):expected[tick]+=1
    actual=[r['standing_recovery_environments'] for r in records]
    assert actual==expected,'Recovery counts differ from alive lanes that entered recovery'
    assert sum(actual)==summary['standing_recovery_active_control_steps']
    for r in records:
        assert r['standing_recovery_mode']=='initial_backward_goal'
        assert r['refinement_scope']=='initial_pass'
    goal=directory/'initial_backward_goal.pt';assert goal.exists()
    return dict(source=str(directory),steps=len(records),recovery_environments=len(starts),
        active_control_steps=sum(actual),alive_recovery_counts_exact=True,active_steps_per_environment=durations,
        initial_goal_sha256=hashlib.sha256(goal.read_bytes()).hexdigest(),
        scope='Per-tick aggregate activation matches live recovery intervals; no claim of per-row action identity from aggregate records alone')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    result=audit(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['recovery_environments','active_control_steps','alive_recovery_counts_exact']}))

if __name__=='__main__':main()
