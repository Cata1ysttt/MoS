"""Dense latent rollout collection with explicit teacher assistance provenance."""
import json,hashlib,time
import torch
from .paths import ROOT,ASSETS
from .bridge import Bridge
from .multiskill import load_projector,ConditionedProjector
from .native_bfm import load_native,NativeActor
from .latent_oracle import fit_latent
from .planner import observation,SCHEMA,load_planner


def collect(env,expert,obs,args):
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    core=env.unwrapped;device=core.device;n=core.num_envs
    bridge=Bridge(core,ASSETS);actor=NativeActor(load_native(device))
    ck=torch.load(args.projector_checkpoint,map_location='cpu',weights_only=False)
    teacher=load_projector(ck,device).eval()
    if ck.get('multiskill'):teacher=ConditionedProjector(teacher,args.skill)
    student=None
    if args.planner_student:
        student=load_planner(torch.load(args.planner_student,map_location='cpu',weights_only=False),device)
    if not 0 <= args.planner_teacher_ratio <= 1:raise ValueError('Teacher ratio outside [0,1]')
    teacher_lane=torch.rand(n,device=device)<args.planner_teacher_ratio
    episode=torch.zeros(n,dtype=torch.long,device=device)
    episode_tick=episode.clone();rows=[];start=time.monotonic()
    metadata=dict(schema=SCHEMA,skill=args.skill,seed=args.seed,dt=float(core.step_dt),
        projector=str(args.projector_checkpoint),expert_assisted=args.oracle_latent_steps>0,
        oracle_steps=args.oracle_latent_steps,latent_interval=1,execution='hold',
        student=str(args.planner_student),teacher_episode_probability=args.planner_teacher_ratio,
        label_source='expert_action_inversion' if args.oracle_latent_steps else 'skill_projector',
        actor_sha256=hashlib.sha256((ROOT/'AH-BFM/AH-BFM.pt').read_bytes()).hexdigest())
    (args.output/'collection.json').write_text(json.dumps(metadata,indent=2))
    for tick in range(args.steps):
        with torch.no_grad():
            x=observation(core,bridge,args.skill);state=bridge.observe()
            z=teacher(torch.cat((obs['policy'],obs['rma_obs']),-1))
            target=bridge.expert_targets(expert.act_inference(obs['policy'],obs['rma_obs']))
        if args.oracle_latent_steps:
            z,_=fit_latent(actor,state,target,z,bridge.targets,args.oracle_latent_steps)
        with torch.no_grad():
            executed=z
            if student is not None:
                student_z=student.act(x)
                executed=torch.where(teacher_lane[:,None],z,student_z)
            action=actor(*state,executed);physical=bridge.targets(action)
            row=dict(x=x.cpu().clone(),z=executed.cpu().clone(),label_z=z.cpu().clone(),
                teacher_executed=(teacher_lane if student is not None else torch.ones_like(teacher_lane)).cpu().clone(),
                target=target.cpu().clone(),
                episode=episode.cpu().clone(),episode_tick=episode_tick.cpu().clone())
            obs,_,term,trunc,_=env.step(bridge.env_action(physical));done=term|trunc
            row['done']=done.cpu().clone();rows.append(row)
            bridge.advance(action,done);episode_tick+=1;episode_tick[done]=0;episode[done]+=1
            teacher_lane[done]=torch.rand(int(done.sum()),device=device)<args.planner_teacher_ratio
        if (tick+1)%100==0:
            print('COLLECT',tick+1,'episodes',int(episode.sum()),'seconds',time.monotonic()-start,flush=True)
        if (tick+1)%250==0 or tick+1==args.steps:
            torch.save(dict(metadata=metadata,rows=rows),args.output/'trajectory.tmp')
            (args.output/'trajectory.tmp').replace(args.output/'trajectory.pt')
    (args.output/'complete.json').write_text(json.dumps(dict(ticks=args.steps,episodes=int(episode.sum()))))
