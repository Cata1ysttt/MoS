"""Launch the fixed fresh confirmation only after a useful completed development study."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_target_guided_planner import ROOT,read,sha
from confirm_new_ball_task_history import useful_development


def main():
    runs=ROOT/'runs';status=runs/'planner_new_ball_task_confirmation_queue_status.json'
    assert not status.exists() and not (runs/'planner_new_ball_task_confirmation_r1_status.json').exists()
    driver=ROOT/'scripts/confirm_new_ball_task_history.py';driver_sha=sha(driver)
    dependency=runs/'planner_new_ball_task_history_study_r1_status.json'
    def state(stage,**details):
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),
            confirmation_driver_sha256=driver_sha,**details),indent=2)+'\n');tmp.replace(status)
    try:
        while True:
            study=read(dependency)
            if study['stage'] in ['completed','failed']:break
            process=Path(f"/proc/{study['pid']}/cmdline")
            assert process.exists() and b'run_new_ball_task_history_study.py' in process.read_bytes(),'Development driver is no longer live'
            state('waiting_for_development',dependency_pid=study['pid'])
            time.sleep(10)
        assert study['stage']=='completed',study.get('error','Development failed')
        assert sha(driver)==driver_sha,'Confirmation driver changed while queued'
        totals=study['results']['totals'];control=totals['control'];candidate=totals['candidate']
        gain=useful_development(totals)
        if not gain:
            state('not_launched_development_not_improved',development_totals=totals);return
        state('running_confirmation',development_totals=totals)
        with (runs/'planner_new_ball_task_confirmation_r1_driver.log').open('x') as log:
            subprocess.run([sys.executable,str(driver)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(runs/'planner_new_ball_task_confirmation_r1_status.json')['stage']=='completed'
        state('completed',confirmation_status='planner_new_ball_task_confirmation_r1_status.json')
    except Exception as error:
        state('failed',error=repr(error));raise

if __name__=='__main__':main()
