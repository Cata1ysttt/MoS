"""Broader current-policy recovery data with exact expert observation histories."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--student',type=Path,required=True)
p.add_argument('--pass-expert',type=Path,help='Override active passing expert checkpoint; sibling params directory is required')
p.add_argument('--pass-config',type=Path,default=Path('/home/jiongye/g1-rollingball/logs/rsl_rl/g1_rolling_pass_event_amp/2026-11-03_11-31-56_rolling_pass_v1_stage2/params/env.pkl'),help='Fixed collection scene, independent of the selected expert training package')
p.add_argument('--skills',choices=['pass','shoot'],nargs='+',default=['shoot','pass'],help='Collect selected skills; defaults to both original tasks')
p.add_argument('--labeling-gpu',type=int,help='Physical GPU for offline semantic labels; simulation keeps its existing device')
p.add_argument('--tag',default='planner_recovery_r2')
p.add_argument('--seed',type=int,default=812401)
p.add_argument('--num-envs',type=int,default=128)
p.add_argument('--steps',type=int,default=1000)
p.add_argument('--teacher-ratio',type=float,default=.5)
p.add_argument('--resume',action='store_true')
p.add_argument('--wait-for-validation',help='Start collection only after this paired validation tag completes')
a = p.parse_args()
assert a.tag.replace('_','').isalnum() and 0 <= a.teacher_ratio <= 1
if len(a.skills)!=len(set(a.skills)):p.error('Skills must not be repeated')
if a.labeling_gpu is not None and a.labeling_gpu<0:p.error('Labeling GPU must be nonnegative')
assert a.seed not in (811314,812801,812901,812902,812903,812904,812905) and a.seed+1 not in (811314,812801,812901,812902,812903,812904,812905)
root = Path(__file__).resolve().parents[1]
runs = root/'runs'
student = a.student.resolve()
assert student.exists() and (student.parent/'complete.json').exists()
student_hash=hashlib.sha256(student.read_bytes()).hexdigest()
lock = (runs/(a.tag+'.lock')).open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status_path = runs/(a.tag+'_status.json')
expert_config = json.loads((root/'config/soccer_experts.json').read_text())['pass']
expert_path = (a.pass_expert or root/expert_config['checkpoint']).resolve()
if 'pass' in a.skills:
    assert expert_path.is_file() and all((expert_path.parent/'params'/name).is_file() for name in ['env.pkl','agent.pkl'])
    expert_hash = hashlib.sha256(expert_path.read_bytes()).hexdigest()
    if a.pass_expert is None:
        assert expert_hash == expert_config['checkpoint_sha256'], 'Active expert hash mismatch'
    provenance_path = runs/(a.tag+'_pass_expert.json')
    pass_config=a.pass_config.resolve()
    assert pass_config.is_file()
    provenance = dict(checkpoint=str(expert_path), sha256=expert_hash, env_config=str(pass_config), env_sha256=hashlib.sha256(pass_config.read_bytes()).hexdigest())
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance, 'Cannot resume with another passing expert'
    else:
        assert not (runs/(a.tag+'_pass_collect')).exists(), 'Existing collection lacks expert provenance; use a new tag'
        provenance_path.write_text(json.dumps(provenance,indent=2)+'\n')
    # The existing runtime mounts the project and g1 source at these locations.
    if expert_path.is_relative_to(root):
        expert = '/workspace/fiber-soccer/'+str(expert_path.relative_to(root))
    elif expert_path.is_relative_to(Path('/home/jiongye/g1-rollingball')):
        expert = '/workspace/g1-rollingball/'+str(expert_path.relative_to('/home/jiongye/g1-rollingball'))
    elif str(expert_path).startswith('/workspace/'):
        expert = str(expert_path)
    else:
        raise ValueError('Passing expert must be in an existing runtime mount')
    if pass_config.is_relative_to(root):
        expert_env='/workspace/fiber-soccer/'+str(pass_config.relative_to(root))
    elif pass_config.is_relative_to(Path('/home/jiongye/g1-rollingball')):
        expert_env='/workspace/g1-rollingball/'+str(pass_config.relative_to('/home/jiongye/g1-rollingball'))
    elif str(pass_config).startswith('/workspace/'):
        expert_env=str(pass_config)
    else:
        raise ValueError('Passing config must be in an existing runtime mount')


def c(path):
    return '/workspace/fiber-soccer/'+str(path.relative_to(root))


def state(stage,**details):
    status_path.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=time.time(),**details),indent=2))


def execute(name,command):
    output = runs/name
    if a.resume and (output/'complete.json').exists():
        state(name,status='reused_completed')
        return
    if output.exists():
        raise RuntimeError('Incomplete stage needs explicit recovery: '+str(output))
    log = runs/(name+'.log')
    attempt = 1
    while a.resume and log.exists():
        log = runs/f'{name}.attempt{attempt}.log'
        attempt += 1
    state(name,status='running',log=str(log))
    environment=os.environ.copy()
    if a.labeling_gpu is not None and Path(command[0]).name=='prepare_semantic_planner_data.py':
        environment['MOS_CUDA_VISIBLE_DEVICES']=str(a.labeling_gpu)
    with log.open('x') as stream:
        subprocess.run([str(root/'scripts/container_exec.sh')]+command,stdout=stream,stderr=subprocess.STDOUT,check=True,env=environment)


try:
    if a.wait_for_validation:
        assert a.wait_for_validation.replace('_','').isalnum()
        upstream_path=runs/(a.wait_for_validation+'_status.json')
        while True:
            upstream=json.loads(upstream_path.read_text())
            if upstream['status']=='completed':break
            if upstream['status']=='failed':raise RuntimeError('Paired validation failed: '+upstream.get('error','unknown'))
            process=Path(f"/proc/{upstream['pid']}/cmdline")
            if not process.exists() or b'validate_planner_candidate.py' not in process.read_bytes():
                raise RuntimeError('Paired validation driver is no longer live')
            state('waiting_for_validation',status='running',validation=a.wait_for_validation,validation_pid=upstream['pid'])
            time.sleep(30)
    collected = []
    for skill,seed in [('shoot',a.seed),('pass',a.seed+1)]:
        if skill not in a.skills:continue
        name = a.tag+'_'+skill+'_collect'
        entry = 'probe_shoot_expert.py' if skill == 'shoot' else 'evaluate_selected_pass_expert.py'
        source = 'joint_v2_r2_pass_train' if skill == 'shoot' else 'passing_50hz_long_s1'
        command = ['/workspace/fiber-soccer/scripts/'+entry,'--headless','--device','cuda:0',
                   '--planner-collect','--planner-observation-schema','soccer_state_v2_950',
                   '--projector-checkpoint',c(runs/source/'latest.pt'),'--planner-student',c(student),
                   '--planner-teacher-ratio',str(a.teacher_ratio),'--oracle-latent-steps','64',
                   '--seed',str(seed),'--num-envs',str(a.num_envs),'--steps',str(a.steps),
                   '--output',c(runs/name)]
        if skill == 'pass':
            command += ['--checkpoint',expert,'--env-config',expert_env]
        execute(name,command)
        meta = json.loads((runs/name/'collection.json').read_text())
        complete = json.loads((runs/name/'complete.json').read_text())
        assert meta['student']==c(student) and meta['seed']==seed and meta['num_envs']==a.num_envs
        assert meta['student_sha256']==student_hash==hashlib.sha256(student.read_bytes()).hexdigest(),'Student checkpoint changed'
        assert meta['teacher_episode_probability']==a.teacher_ratio and complete['ticks']==a.steps
        assert 'expert_observation_groups' in meta
        collected.append(runs/name/'trajectory.pt')
    output = runs/(a.tag+'_semantic')
    execute(output.name,['/workspace/fiber-soccer/scripts/prepare_semantic_planner_data.py',
                         '--data',*map(c,collected),'--output',c(output)])
    assert (output/'complete.json').exists()
    state('completed',status='completed',student=str(student),student_sha256=student_hash,skills=a.skills,datasets=[str(output/(s+'_trajectory.pt')) for s in a.skills])
except Exception as error:
    state('failed',status='failed',error=repr(error))
    raise
