"""Paired online correction with/without oracle-state replay; no test-set tuning."""
import json
import subprocess
import time
import fcntl
from pathlib import Path
from run_near_ball_ablation import BASE, ROOT, RUNS
STATUS=RUNS/'mixed_replay_status.json'
def status(**values):
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(values,indent=2));temp.replace(STATUS)
def run(name,args,artifact):
    if (RUNS/name).exists():raise RuntimeError(f'Refuse overwrite {name}')
    status(stage=name,status='running')
    with (RUNS/f'{name}.log').open('x') as log:
        subprocess.run(BASE+args+['--output',f'/workspace/fiber-soccer/runs/{name}'],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((RUNS/name/artifact).read_text())
    if artifact=='summary.json' and result['completed']!=64:raise RuntimeError('Incomplete evaluation')
    if artifact=='parameter_audit.json' and not result['bfm_gradients_absent']:raise RuntimeError('Actor freeze failed')
    return result

def main():
    with (RUNS/'mixed_replay.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status(stage='waiting_for_oracle_collection',status='queued')
        while True:
            probe=subprocess.run(['docker','exec','fiber-soccer-runtime','pgrep','-f','[p]ython3.*expert_baseline.py'],capture_output=True)
            if probe.returncode==1:break
            if probe.returncode!=0:raise RuntimeError('Cannot inspect simulator')
            time.sleep(30)
        source=RUNS/'oracle_training64_seed810181'
        summary=json.loads((source/'summary.json').read_text())
        if summary['completed']!=64 or not (source/'action_diagnostics.pt').is_file():raise RuntimeError('Incomplete oracle collection')
        evaluation=['--evaluate','--num-envs','64','--steps','2000','--seed','810127']
        run('passing_impact_expert64',evaluation,'summary.json')
        run('passing_impact_s1_64',evaluation+['--projector-checkpoint','/workspace/fiber-soccer/runs/passing_50hz_long_s1/latest.pt'],'summary.json')
        common=['--train-lpp','--num-envs','64','--seed','810191','--latent-interval','1',
                '--resume-projector','/workspace/fiber-soccer/runs/passing_50hz_long_s1/latest.pt',
                '--expert-replay','/workspace/fiber-soccer/runs/expert_dense_v1','--learning-rate','1e-5']
        extra=['--diagnostic-replay','/workspace/fiber-soccer/runs/oracle_training64_seed810181/action_diagnostics.pt','--diagnostic-replay-weight','0.5']
        run('mixed_replay_smoke',common+extra+['--steps','3200'],'parameter_audit.json')
        results={}
        for label,additional in [('control',[]),('mixed',extra)]:
            name=f'passing_online_{label}_64k'
            run(name,common+additional+['--steps','64000'],'parameter_audit.json')
            results[label]=run('eval_'+name,['--evaluate','--num-envs','64','--steps','2000','--seed','810127',
                '--projector-checkpoint',f'/workspace/fiber-soccer/runs/{name}/latest.pt'],'summary.json')
        status(stage='paired_development_complete',status='completed',results=results,note='Autonomous evaluation; not final held-out test. Retain s1 unless improvement verified.')
if __name__=='__main__':
    try:main()
    except Exception as e:
        status(stage='error',status='failed',error=repr(e));raise
