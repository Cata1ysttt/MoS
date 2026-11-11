"""Expert-trajectory warm start through the frozen actor; not a skill evaluation."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from onnx2torch import convert
from fiber_soccer.expert_data import expert_windows
from fiber_soccer.lpp import Projector,execute_latent

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--resume',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--epochs',type=int,default=30)
p.add_argument('--lr',type=float,default=1e-4)
p.add_argument('--batch-size',type=int,default=64)
p.add_argument('--contact-weight',type=float,default=4)
p.add_argument('--device',default='cuda:0')
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(82)
a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
rows=torch.load(a.data/'expert_dense.pt',weights_only=True)
data=expert_windows(rows);del rows
data={k:tuple(x.to(a.device) for x in v) if k=='states' else v.to(a.device) for k,v in data.items()}
n=int(data['env_ids'].max())+1;cut=max(1,n-3)
train=torch.where(data['env_ids']<cut)[0];val=torch.where(data['env_ids']>=cut)[0]
ck=torch.load(a.resume,map_location=a.device,weights_only=False)
model=Projector(ck['obs_dim']).to(a.device);model.load_state_dict(ck['projector'])
actor=convert(str(root/'assets/actor.onnx')).eval().to(a.device).requires_grad_(False)
before={k:v.detach().clone() for k,v in actor.named_parameters()}
meta=json.loads((root/'assets/actor_meta.json').read_text());scene=json.loads((root/'assets/pickup_scene.json').read_text())
scale=torch.tensor(meta['action_rescale'],device=a.device)*1.5;default=torch.tensor(scene['default_q'],device=a.device)
stand=torch.tensor(json.loads((root/'assets/z_bank.json').read_text())['stand'],device=a.device)
optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
config=dict(method='expert_window_warmstart',data=str(a.data),data_sha256=hashlib.sha256((a.data/'expert_dense.pt').read_bytes()).hexdigest(),parent=str(a.resume),train_envs=list(range(cut)),validation_envs=list(range(cut,n)),train_windows=len(train),validation_windows=len(val),contact_weight=a.contact_weight,lr=a.lr,epochs=a.epochs,latent_execution='slerp',latent_interval=5)
(a.output/'config.json').write_text(json.dumps(config,indent=2))
import swanlab
swanlab.init(project='g1-soccer-bfm',workspace='cata1yst',name=a.output.name,config=config)

def errors(idx):
    with torch.no_grad():
        old=model(data['previous_obs'][idx])
        old=torch.where(data['previous_valid'][idx,None],old,stand)
    z=model(data['obs'][idx]);pred=[]
    for h in range(5):
        out=actor(*(v[idx,h] for v in data['states']),execute_latent(old,z,(h+1)/5,'slerp'))
        pred.append(default+out.clamp(-1,1)*scale)
    err=(torch.stack(pred,1)-data['targets'][idx]).square().mean(-1)
    mask=data['mask'][idx]
    return (err*mask).sum(1)/mask.sum(1).clamp_min(1)

@torch.no_grad()
def validate():
    model.eval();err=torch.cat([errors(idx) for idx in val.split(a.batch_size)])
    contact=data['contact'][val]
    return dict(validation_mse=float(err.mean()),validation_contact_mse=float(err[contact].mean()) if contact.any() else None)

weights=torch.where(data['contact'][train],a.contact_weight,1.).float()
start=time.monotonic();best=float('inf');initial=validate()
print('INITIAL',initial,flush=True);(a.output/'initial_validation.json').write_text(json.dumps(initial))
for epoch in range(1,a.epochs+1):
    model.train();samples=train[torch.multinomial(weights,len(train),replacement=True)];losses=[]
    for idx in samples.split(a.batch_size):
        optimizer.zero_grad();loss=errors(idx).mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite warmstart loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
    report=dict(epoch=epoch,train_mse=sum(losses)/len(losses),elapsed=time.monotonic()-start,**validate())
    print(report,flush=True)
    with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
    swanlab.log(report,step=epoch)
    checkpoint=dict(projector=model.state_dict(),optimizer=optimizer.state_dict(),obs_dim=ck['obs_dim'],latent_execution='slerp',steps=0,cumulative_steps=ck.get('cumulative_steps',ck['steps']),warmstart_epoch=epoch,warmstart_config=config)
    torch.save(checkpoint,a.output/'latest.tmp');(a.output/'latest.tmp').replace(a.output/'latest.pt')
    if report['validation_mse']<best:
        best=report['validation_mse'];torch.save(checkpoint,a.output/'best.pt')
assert all(v.grad is None and torch.equal(v,before[k]) for k,v in actor.named_parameters())
(a.output/'parameter_audit.json').write_text(json.dumps(dict(actor_unchanged=True,expert_actions_fixed=True,best_validation_mse=best,initial_validation=initial)))
swanlab.finish()
