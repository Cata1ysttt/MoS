"""Post-shot recovery phase and bounded stability rewards, without expert actions."""
import torch


class PostShotRecovery:
    def __init__(self,num_envs,device,delay=10):
        if not isinstance(delay,int) or delay<0:raise ValueError('Invalid recovery delay')
        self.delay=delay
        self.age=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.first_kick=torch.full_like(self.age,-1)
        self.active=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.entries=0

    def advance(self,kicked,done):
        if kicked.shape!=self.age.shape or done.shape!=self.age.shape:
            raise ValueError('Expected one kick/reset flag per environment')
        self.age+=1
        self.age[done]=0;self.first_kick[done]=-1
        observed=kicked & ~done & (self.first_kick<0)
        self.first_kick[observed]=self.age[observed]
        active=(self.first_kick>=0)&(self.age-self.first_kick>=self.delay)&~done
        self.entries+=int((active&~self.active).sum())
        self.active=active

    def next_skills_without_reset(self,kicked):
        age=self.age+1
        first=torch.where(kicked&(self.first_kick<0),age,self.first_kick)
        active=(first>=0)&(age-first>=self.delay)
        return (~active).long()

    def skills(self):
        return (~self.active).long()  # frozen shooting first, passing recovery second


def stability_score(gravity,height,velocity,angular_velocity,target_height=.76):
    upright=(-gravity[...,2]).clamp(0,1).square()
    height_score=torch.exp(-((height-target_height)/.12).square())
    slow=torch.exp(-velocity[...,:2].square().sum(-1)/.25)
    quiet=torch.exp(-angular_velocity.square().sum(-1))
    return upright*height_score*(.5+.25*slow+.25*quiet)


def recovery_stability(env,target_height=.76):
    robot=env.scene['robot'].data
    height=robot.root_pos_w[:,2]-env.scene.env_origins[:,2]
    return stability_score(robot.projected_gravity_b,height,robot.root_lin_vel_w,
        robot.root_ang_vel_b,target_height)*env._soccer_recovery_active


def recovery_action_rate(env):
    delta=(env.action_manager.action-env.action_manager.prev_action)/10.
    return delta.square().mean(-1).clamp(max=1.)*env._soccer_recovery_active


def recovery_failure(env):
    return env.termination_manager.terminated.float()*env._soccer_recovery_active


def normalize_active_advantages(advantages,active):
    result=torch.zeros_like(advantages)
    if active.any():
        selected=advantages[active]
        result[active]=(selected-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8)
    return result
