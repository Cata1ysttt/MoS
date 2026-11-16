"""UFO paired exports and robot-state bridge. Keeps expert environment dynamics."""
import json
import torch
import onnx
from onnx2torch import convert
from .bridge import Bridge
from .paths import ASSETS


def load_export(path, device):
    graph=onnx.load(str(path))
    for node in graph.graph.node:
        if node.op_type=='Clip':
            while node.input and node.input[-1]=='':node.input.pop()
    return convert(graph).eval().to(device).requires_grad_(False)


class Actor(torch.nn.Module):
    def __init__(self, device):
        super().__init__();self.model=load_export(ASSETS/'ufo/g1_policy/exported/FBcprAuxModel.onnx',device)
    def forward(self,state,last,history,z):
        return self.model(torch.cat((state,last,history,z),-1))


def rotate(q,v):
    # wxyz quaternions, active rotation.
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)


def privileged(pos,quat,vel,omega):
    q=quat[:,0];yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    inv=torch.zeros_like(q);inv[:,0]=torch.cos(yaw/2);inv[:,3]=-torch.sin(yaw/2)
    inv=inv[:,None].expand_as(quat)
    local=rotate(inv,pos-pos[:,:1])[:,1:].flatten(1)
    # Rotation representation is tangent (x axis) followed by normal (z axis).
    x=torch.zeros_like(pos);x[...,0]=1;z=torch.zeros_like(pos);z[...,2]=1
    rotation=torch.cat((rotate(inv,rotate(quat,x)),rotate(inv,rotate(quat,z))),-1).flatten(1)
    return torch.cat((pos[:,0,2:3],local,rotation,rotate(inv,vel).flatten(1),rotate(inv,omega).flatten(1)),-1)


class UFOBridge(Bridge):
    def __init__(self,env):
        self.env=env;self.robot=env.scene['robot'];self.device=env.device
        self.meta=json.loads((ASSETS/'ufo/g1_policy/exported/FBcprAuxModel.meta.json').read_text())
        self.names=self.meta['control_joint_names'];dep=self.meta['deployment']
        self.ids=[self.robot.joint_names.index(n) for n in self.names]
        self.default=torch.tensor([dep['default_joint_pos'][n] for n in self.names],device=self.device)
        self.scale=torch.tensor([dep['action_scale'][n] for n in self.names],device=self.device)
        self.last=torch.zeros(env.num_envs,29,device=self.device);self.history=torch.zeros(env.num_envs,5,93,device=self.device)
        terms=env.action_manager.active_terms
        if len(terms)!=1:raise ValueError(terms)
        self.term=env.action_manager.get_term(terms[0]);self.order=[self.term._joint_names.index(n) for n in self.names]
        self.inverse=torch.argsort(torch.tensor(self.order,device=self.device))
        self.body_ids=[self.robot.body_names.index(n) for n in self.meta['reference_body_names']]
        self.parent=self.meta['reference_body_names'].index('torso_link')
        self.observe(append=True)
    def observe(self,append=False):
        d=self.robot.data;q=d.joint_pos[:,self.ids]-self.default;v=d.joint_vel[:,self.ids];w=d.root_ang_vel_b;g=d.projected_gravity_b
        frame=torch.cat((self.last,w,q,v,g),-1)
        if append:self.history=torch.cat((frame[:,None],self.history[:,:4]),1)
        hist=torch.cat([self.history[:,1:,a:b].flatten(1) for a,b in [(0,29),(29,32),(32,61),(61,90),(90,93)]],-1)
        return torch.cat((q,v,g,w),-1),self.last.clone(),hist
    def targets(self,a):return self.default+(a*5).clamp(-5,5)*self.scale
    def advance(self,a,done):
        self.last=(a*5).clamp(-5,5).detach();self.last[done]=0;self.history[done]=0;self.observe(append=True)
    def encoder_inputs(self):
        d=self.robot.data
        pos=d.body_pos_w[:,self.body_ids].clone();pos-=self.env.scene.env_origins[:,None]
        q=d.body_quat_w[:,self.body_ids];w=d.body_ang_vel_w[:,self.body_ids]
        # body_link velocity avoids conflating COM linear velocity with body-frame origin.
        v=d.body_link_lin_vel_w[:,self.body_ids]
        offset=torch.zeros_like(pos[:,self.parent]);offset[:,2]=.35;offset=rotate(q[:,self.parent],offset)
        pos=torch.cat((pos,(pos[:,self.parent]+offset)[:,None]),1)
        v=torch.cat((v,(v[:,self.parent]+torch.cross(w[:,self.parent],offset,dim=-1))[:,None]),1)
        w=torch.cat((w,w[:,self.parent:self.parent+1]),1);q=torch.cat((q,q[:,self.parent:self.parent+1]),1)
        result=privileged(pos,q,v,w)
        assert result.shape[-1]==463 and torch.isfinite(result).all()
        return self.observe()[0],result
