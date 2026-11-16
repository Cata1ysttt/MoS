"""Alternate shared-projector DAgger stages; validate both skills per round."""
import json,subprocess,time,hashlib,os
from pathlib import Path
root=Path(__file__).resolve().parents[1];runs=root/'runs'
status=runs/'joint_v2_status.json';prefix=[str(root/'scripts/container_exec.sh')]
expert='/workspace/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/model_520000.pt'
def container(p):return '/workspace/fiber-soccer/'+str(p.relative_to(root))
def execute(name, args):
    out=runs/name
    if out.exists():raise RuntimeError(f'Refuse overwrite {out}')
    status.write_text(json.dumps(dict(status='running',stage=name,driver_pid=os.getpid(),started=time.time()),indent=2))
    with (runs/(name+'.log')).open('x') as f:subprocess.run(prefix+args+['--output',container(out)],stdout=f,stderr=subprocess.STDOUT,check=True)
    return out
try:
    latest=runs/'joint_v2_shoot_smoke/latest.pt'
    replay={'shoot':runs/'joint_v2_shoot_smoke/student_replay.pt'}
    rounds=[]
    for round_id in range(1,4):
        for skill in ['shoot','pass']:
            name=f'joint_v2_r{round_id}_{skill}_train'
            entry='probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'
            args=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--joint-train','--joint-resume',container(latest),'--joint-replay',*[container(x) for x in replay.values()],'--steps','32000','--num-envs','32','--seed',str(811330+round_id*2+(skill=='pass')),'--teacher-ratio',str(.25 if round_id==1 else .1 if round_id==2 else 0.)]
            if skill=='pass':args+=['--checkpoint',expert]
            out=execute(name,args)
            assert (out/'parameter_audit.json').exists()
            latest=out/'latest.pt';replay[skill]=out/'student_replay.pt'
        pair={}
        for skill in ['shoot','pass']:
            name=f'joint_v2_r{round_id}_{skill}_eval'
            entry='probe_shoot_expert.py' if skill=='shoot' else 'expert_baseline.py'
            args=['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0','--evaluate','--projector-checkpoint',container(latest),'--num-envs','16','--steps','2000','--seed','811314']
            if skill=='pass':args+=['--checkpoint',expert]
            out=execute(name,args)
            summary=json.loads((out/'summary.json').read_text())
            reference=json.loads((runs/f'pass_shoot_{skill}_expert_eval_v1/summary.json').read_text())
            assert summary['completed']==16
            assert summary['initial_state_sha256']==reference['initial_state_sha256']
            pair[skill]=summary
        rounds.append(dict(round=round_id,checkpoint=str(latest),metrics=pair))
        (runs/'joint_v2_results.json').write_text(json.dumps(rounds,indent=2))
        # Retain the best worst-skill direction score; never overwrite a baseline.
        best=max(rounds,key=lambda r:min(r['metrics'][s]['quality_ball_direction_success_10deg'] for s in ['pass','shoot']))
        (runs/'joint_v2_best.json').write_text(json.dumps(best,indent=2))
    status.write_text(json.dumps(dict(status='completed',rounds=len(rounds),best=best),indent=2))
except Exception as e:
    status.write_text(json.dumps(dict(status='failed',error=repr(e)),indent=2));raise
