"""Wait for recovery data, then compare matched offline input-information probes."""
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runs = root/'runs'
status = runs/'planner_recovery_r2_information_status.json'


def state(stage,**details):
    status.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))


try:
    while True:
        upstream = json.loads((runs/'planner_recovery_r2_status.json').read_text())
        if upstream['status'] == 'completed':
            break
        if upstream['status'] == 'failed':
            raise RuntimeError('Collection failed: '+upstream.get('error','unknown'))
        process = Path(f"/proc/{upstream['pid']}/cmdline")
        if not process.exists() or b'collect_planner_recovery.py' not in process.read_bytes():
            raise RuntimeError('Recovery collector is no longer live')
        state('waiting_for_data',status='running',collector_pid=upstream['pid'])
        time.sleep(30)
    results = {}
    for mode in ['state','expert_input']:
        name = 'planner_recovery_r2_information_'+mode
        command = [str(root/'scripts/container_exec.sh'),'/workspace/fiber-soccer/scripts/diagnose_action_regression.py',
                   '--data','/workspace/fiber-soccer/runs/planner_recovery_r2_semantic/pass_trajectory.pt',
                   '/workspace/fiber-soccer/runs/planner_recovery_r2_semantic/shoot_trajectory.pt',
                   '--reference','/workspace/fiber-soccer/runs/planner_flow_dagger_r1_train/best.pt',
                   '--output','/workspace/fiber-soccer/runs/'+name,'--updates','20000']
        if mode == 'expert_input':
            command.append('--expert-observation')
        state(name,status='running')
        with (runs/(name+'.log')).open('x') as stream:
            subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        results[mode] = json.loads((runs/name/'complete.json').read_text())
    state('completed',status='completed',results=results,diagnostic_only=True)
except Exception as error:
    state('failed',status='failed',error=repr(error))
    raise
