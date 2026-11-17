"""Check per-tick new-ball history activation against handoffs, recovery and failures."""
import argparse
import json
from pathlib import Path


def audit(directory):
    summary=json.loads((directory/'summary.json').read_text())
    records=json.loads((directory/'predicted_target_solver.json').read_text())
    assert summary['new_ball_history_mode'] in ('initialize_shoot_rma','initialize_shoot_ball_task')
    assert len(records)==summary['steps']
    failures={v['environment']:v['tick'] for v in summary['first_resets']}
    recovery={i:v['tick'] for v in summary['recovery_switches'] for i in v['environments']}
    starts={i:v['tick'] for v in summary['switches'] for i in v['environments']}
    expected=[0]*summary['steps'];durations={}
    for i,start in starts.items():
        end=min(start+50,summary['steps'],failures.get(i,summary['steps']-1)+1,recovery.get(i,summary['steps']))
        assert 0<=start<end
        durations[str(i)]=end-start
        for tick in range(start,end):expected[tick]+=1
    actual=[r['new_ball_history_environments'] for r in records]
    assert actual==expected,'New-ball history did not follow live new-object intervals'
    assert sum(actual)==summary['new_ball_history_active_control_steps']
    extra={}
    if summary['new_ball_history_mode']=='initialize_shoot_ball_task':
        expected_task=[0]*summary['steps']
        for i,start in starts.items():
            end=min(start+10,summary['steps'],failures.get(i,summary['steps']-1)+1,recovery.get(i,summary['steps']))
            for tick in range(start,end):expected_task[tick]+=1
        actual_task=[r['new_ball_task_history_environments'] for r in records]
        assert actual_task==expected_task
        assert sum(actual_task)==summary['new_ball_task_history_active_control_steps']
        extra=dict(ball_task_history_active_control_steps=sum(actual_task),alive_ball_task_history_counts_exact=True)
    return dict(**extra,handoffs=len(starts),active_control_steps=sum(actual),active_steps_per_environment=durations,
        alive_new_ball_history_counts_exact=True,
        scope='Aggregate activation audit; policy routing and unchanged non-RMA input are checked separately')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    result=audit(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
