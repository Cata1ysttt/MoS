"""Shared-projector DAgger, balanced skill replay and passing latent anchoring."""
import json
import time
from pathlib import Path
import torch
from onnx2torch import convert
from .paths import ASSETS, ROOT
from .bridge import Bridge
from .lpp import Projector
from .latent_oracle import fit_latent
from .multiskill import condition, load_projector, WarmStartedJointProjector


def pack_rows(rows, skill):
    buckets = [[], [], [], [], []]
    for row in rows:
        if 'projector_obs' in row:
            x, states, y = row['projector_obs'], row['bfm_inputs'], row['targets']
        else:
            x = torch.cat((row['policy_obs'], row['rma_obs']), -1)
            states, y = row['inputs'], row['target']
        # Old collections reserve the last four lanes; never consume them.
        count = len(x) - 4
        for bucket, value in zip(buckets, (condition(x, skill), *states, y)):
            bucket.append(value[:count].float().cpu())
    return tuple(torch.cat(parts) for parts in buckets)


def make_joint_optimizer(model, ck):
    # The task offset starts at zero to preserve s1 exactly. Its scale needs a
    # higher learning rate than the pretrained trunk to make conditioning useful.
    opt = torch.optim.AdamW([
        {'params': [model.skill_offset], 'lr': 1e-2},
        {'params': list(model.base.parameters()), 'lr': 1e-4},
    ], weight_decay=1e-4)
    if 'optimizer' in ck:
        saved = ck['optimizer']
        if len(saved['param_groups']) == 1:
            old = saved['param_groups'][0]
            saved = dict(saved, param_groups=[
                dict(old, params=old['params'][:1], lr=1e-2),
                dict(old, params=old['params'][1:], lr=1e-4),
            ])
        opt.load_state_dict(saved)
        opt.param_groups[0]['lr'] = 1e-2
        opt.param_groups[1]['lr'] = 1e-4
    return opt


def train(env, expert, obs, args):
    torch.set_num_threads(4)
    core = env.unwrapped
    device, n = core.device, core.num_envs
    torch.manual_seed(args.seed)
    skill = args.skill
    bridge = Bridge(core, ASSETS)
    actor = convert(str(ASSETS / 'actor.onnx')).to(device).eval().requires_grad_(False)
    source = torch.load(args.pass_anchor, map_location='cpu', weights_only=False)
    if source.get('latent_interval') != 1:
        raise ValueError('Anchor must use 50Hz execution')
    anchor = Projector(1200).to(device).eval().requires_grad_(False)
    anchor.load_state_dict(source['projector'])
    if args.joint_resume:
        ck = torch.load(args.joint_resume, map_location='cpu', weights_only=False)
        model = load_projector(ck, device)
        if ck.get('projector_arch') != 'warm_shared_v2':
            raise ValueError('Expected version-2 warm shared checkpoint')
    else:
        ck = {}
        model = WarmStartedJointProjector().to(device)
        model.base.load_state_dict(source['projector'])
    opt = make_joint_optimizer(model, ck)
    original = {
        'pass': pack_rows(torch.load(ROOT/'runs/expert_dense_v1/expert_dense.pt', weights_only=True), 'pass'),
        'shoot': pack_rows(torch.load(ROOT/'runs/shoot_expert_probe_v1/teacher_records.pt', weights_only=True), 'shoot'),
    }
    replay = {}
    for path in args.joint_replay:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        replay[saved['skill']] = saved['data']
    current = []
    teacher_lane = torch.rand(n, device=device) < args.teacher_ratio
    valid_lanes = max(1, n * 3 // 4)
    config = dict(method='shared_projector_dagger',skill=skill,seed=args.seed,
                  teacher_episode_probability=args.teacher_ratio,pass_anchor=str(args.pass_anchor),
                  resume=str(args.joint_resume),replay=[str(x) for x in args.joint_replay],
                  additional_transitions=args.steps,updates_per_50_ticks=20,
                  batch_per_skill=64,anchor_weight=.02,actor_frozen=True,
                  shared_learning_rate=1e-4,task_offset_learning_rate=1e-2,
                  shoot_latent_label_iterations=32,shoot_latent_label_weight=.1,
                  training_lanes=list(range(valid_lanes)),eval_seed_excluded=811314)
    (args.output/'joint_config.json').write_text(json.dumps(config,indent=2))
    import swanlab
    swanlab.init(project='MoS', name=args.output.name, config=config, mode='online')
    if not args.joint_resume:
        with torch.no_grad():
            x = original['pass'][0][:32].to(device)
            error = float((model(x)-anchor(x[:,:1200])).abs().max())
            assert error == 0, error
        (args.output/'initialization_audit.json').write_text(json.dumps(dict(pass_latent_max_abs_error=error)))
    start=time.monotonic();transitions=0;update=0;term_count=0
    def batch(task):
        sources=[original[task]]
        if task in replay:sources.append(replay[task])
        samples=[]
        for data in sources:
            ids=torch.randint(len(data[0]),(64//len(sources),))
            samples.append(tuple(v[ids].to(device) for v in data))
        return tuple(torch.cat([s[k] for s in samples]) for k in range(5))
    def save():
        payload=dict(projector=model.state_dict(),optimizer=opt.state_dict(),obs_dim=1202,
                     multiskill=True,projector_arch='warm_shared_v2',skills=['pass','shoot'],
                     latent_interval=1,latent_execution='hold',steps=transitions,
                     cumulative_steps=ck.get('cumulative_steps',0)+transitions,config=config)
        torch.save(payload,args.output/'latest.tmp');(args.output/'latest.tmp').replace(args.output/'latest.pt')
        if skill in replay:
            torch.save(dict(skill=skill,data=replay[skill],seed=args.seed),args.output/'replay.tmp')
            (args.output/'replay.tmp').replace(args.output/'student_replay.pt')
    try:
        while transitions < args.steps:
            for _ in range(min(50, (args.steps-transitions+n-1)//n)):
                with torch.no_grad():
                    x=condition(torch.cat((obs['policy'],obs['rma_obs']),-1),skill)
                    state=bridge.observe()
                    target=bridge.expert_targets(expert.act_inference(obs['policy'],obs['rma_obs']))
                    action=actor(*state,model(x))
                    current.append(tuple(v[:valid_lanes].detach().cpu().clone() for v in (x,*state,target)))
                    command=torch.where(teacher_lane[:,None],target,bridge.targets(action))
                    obs,_,term,trunc,_=env.step(bridge.env_action(command))
                    done=term|trunc;term_count+=int(term.sum())
                    bridge.advance(torch.where(teacher_lane[:,None],(target-bridge.default)/(6*bridge.scale),action),done)
                    teacher_lane[done]=torch.rand(int(done.sum()),device=device)<args.teacher_ratio
                    transitions+=n
            # Bound the recent per-task state distribution; keep original expert replay separately.
            current=current[-800:]
            replay[skill]=tuple(torch.cat([r[k] for r in current]) for k in range(5))
            losses={};model.train()
            for _ in range(20):
                opt.zero_grad()
                for task in ('pass','shoot'):
                    x,s,last,h,target=batch(task)
                    z=model(x);a=actor(s,last,h,z);prediction=bridge.targets(a)
                    imitation=(prediction-target).square().mean()
                    regularizer=prediction.new_zeros(())
                    if task=='pass':
                        with torch.no_grad():anchor_z=anchor(x[:,:1200])
                        regularizer=(z-anchor_z).square().mean()*.02
                    else:
                        # Invert ONLY training-state expert actions. Deployment
                        # remains one projector forward, with no expert/oracle.
                        fitted, fit = fit_latent(actor, (s,last,h), target,
                                                z.detach(), bridge.targets, 32)
                        regularizer=(z-fitted.detach()).square().mean()*.1
                        losses['shoot_label_initial_mse']=fit['initial_mse']
                        losses['shoot_label_final_mse']=fit['final_mse']
                        losses['shoot_latent_penalty']=float(regularizer.detach())
                    loss=.5*(imitation+regularizer)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    loss.backward();losses[task+'_action_mse']=float(imitation.detach())
                    if task=='pass':losses['pass_anchor_penalty']=float(regularizer.detach())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();update+=1
            model.eval()
            row=dict(transitions=transitions,updates=update,terminations=term_count,
                     elapsed=time.monotonic()-start,**losses)
            print('JOINT_TRAIN',row,flush=True);swanlab.log(row,step=transitions)
            with (args.output/'joint_metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            save()
        assert all(p.grad is None for p in actor.parameters())
        assert all(p.grad is None for p in anchor.parameters())
        (args.output/'parameter_audit.json').write_text(json.dumps(dict(actor_frozen=True,anchor_frozen=True,updates=update)))
    finally:
        swanlab.finish()
