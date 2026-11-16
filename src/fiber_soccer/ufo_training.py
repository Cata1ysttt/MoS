"""UFO action and native backward-semantic projection on expert dynamics."""
import json,time,os
import torch
from .ufo import Actor,UFOBridge,load_export
from .paths import ASSETS
from .lpp import Projector

def train(env,expert,obs,args):
    core=env.unwrapped;device=core.device;torch.set_num_threads(4)
    bridge=UFOBridge(core);actor=Actor(device)
    encoder=load_export(ASSETS/'ufo/g1_policy/exported/backward_encoder.onnx',device)
    # Audit actual randomized simulator gains, not nominal source defaults.
    dep=bridge.meta['deployment'];d=bridge.robot.data
    dynamics={'native_kp':[dep['joint_kp'][n] for n in bridge.names],
              'native_kd':[dep['joint_kd'][n] for n in bridge.names],
              'joint_names':bridge.names,'actuators':{}}
    for key in ['joint_stiffness','joint_damping','joint_effort_limits','joint_armature','joint_friction_coeff']:
        value=getattr(d,key,None)
        if value is not None:dynamics[key]=value[:,bridge.ids].detach().cpu().tolist()
    for name,act in bridge.robot.actuators.items():
        dynamics['actuators'][name]={'type':type(act).__name__,'config':str(act.cfg)}
    (args.output/'dynamics_audit.json').write_text(json.dumps(dynamics,indent=2))
    p=Projector(obs['policy'].shape[-1]+obs['rma_obs'].shape[-1]).to(device)
    opt=torch.optim.AdamW(p.parameters(),lr=args.learning_rate or 1e-4,weight_decay=1e-4)
    if args.resume_projector:
        ck=torch.load(args.resume_projector,map_location=device,weights_only=False)
        if ck.get('backbone')!='ufo':raise ValueError('Not a UFO checkpoint')
        p.load_state_dict(ck['projector']);opt.load_state_dict(ck['optimizer'])
    import onnxruntime as ort
    sessions=[ort.InferenceSession(str(ASSETS/'ufo/g1_policy/exported'/f),providers=['CPUExecutionProvider']) for f in ['FBcprAuxModel.onnx','backward_encoder.onnx']]
    with torch.no_grad():
        state=bridge.observe();inputs=bridge.encoder_inputs();z=encoder(*inputs)
        expected=sessions[1].run(None,{k.name:v.cpu().numpy() for k,v in zip(sessions[1].get_inputs(),inputs)})[0]
        be=float((z-torch.tensor(expected,device=device)).abs().max())
        a=actor(*state,z);expected=sessions[0].run(None,{'actor_obs':torch.cat((*state,z),-1).cpu().numpy()})[0]
        ae=float((a-torch.tensor(expected,device=device)).abs().max())
        assert max(ae,be)<1e-4
    audit=dict(actor_conversion_error=ae,encoder_conversion_error=be,control_dt=core.step_dt,backbone='ufo',dynamics='unchanged expert environment; native UFO PD NOT installed',semantic_weight=args.semantic_weight)
    (args.output/'contract.json').write_text(json.dumps(audit,indent=2))
    import swanlab
    swanlab.init(project=os.environ.get('MOS_SWANLAB_PROJECT','MoS'),name=args.output.name,config=audit)
    steps=0;iteration=0;terminated=0;start=time.monotonic()
    while steps<args.steps:
        records=[]
        for _ in range(50):
            with torch.no_grad():
                x=torch.cat((obs['policy'],obs['rma_obs']),-1);state=bridge.observe();bz=encoder(*bridge.encoder_inputs())
                target=bridge.expert_targets(expert.act_inference(obs['policy'],obs['rma_obs']))
                z=p(x);a=actor(*state,z)
                # Optional teacher warm-start supplies stable states before autonomous DAgger.
                command=target if args.ufo_teacher_collect else bridge.targets(a)
                records.append((x.clone(),*[s.clone() for s in state],target.clone(),bz.clone()))
                obs,_,term,trunc,_=env.step(bridge.env_action(command));done=term|trunc
                actual=(command-bridge.default)/(5*bridge.scale)
                bridge.advance(actual,done);steps+=core.num_envs;terminated+=int(term.sum())
        data=[torch.cat([r[k] for r in records]) for k in range(6)];losses=[]
        for _ in range(args.optimization_passes):
            for ids in torch.randperm(len(data[0]),device=device).split(64):
                x,s,last,h,y,bz=[v[ids] for v in data];z=p(x)
                al=(bridge.targets(actor(s,last,h,z))-y).square().mean()
                sl=(z-bz).square().sum(-1).mean();loss=al+args.semantic_weight*sl
                assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(p.parameters(),1);opt.step();losses.append((float(al),float(sl)))
        iteration+=1
        row=dict(steps=steps,action_loss=sum(v[0] for v in losses)/len(losses),semantic_loss=sum(v[1] for v in losses)/len(losses),terminated=terminated,elapsed=time.monotonic()-start)
        print(row,flush=True);swanlab.log(row,step=steps)
        with (args.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        torch.save(dict(backbone='ufo',projector=p.state_dict(),optimizer=opt.state_dict(),obs_dim=data[0].shape[-1],steps=steps,latent_interval=1,latent_execution='slerp'),args.output/'latest.tmp');(args.output/'latest.tmp').replace(args.output/'latest.pt')
    assert all(v.grad is None for v in actor.parameters()) and all(v.grad is None for v in encoder.parameters())
    (args.output/'parameter_audit.json').write_text(json.dumps(dict(actor_frozen=True,encoder_frozen=True)))
    swanlab.finish()
