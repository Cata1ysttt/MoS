"""Experimental continuous pass/shoot/recovery helpers for latent PPO.

The scheduler and rewards have unit coverage; the simulator integration and
end-to-end PPO trainer are not yet validated. This is not a trained policy.
"""
import copy
import torch
from torch import nn
from .pass_measurement import first_kick_samples
from .latent_ppo import critic_features


class ContinuousShootingPhase:
    def __init__(self,n,device,pass_delay=50,shoot_delay=10):
        if pass_delay<1 or shoot_delay<0:raise ValueError('Invalid phase delays')
        self.pass_delay=pass_delay;self.shoot_delay=shoot_delay
        self.age=torch.zeros(n,dtype=torch.long,device=device)
        self.stage=torch.zeros_like(self.age)  # pass=0, shoot=1, recovery=2
        self.first_pass=torch.full_like(self.age,-1);self.first_shot=torch.full_like(self.age,-1)
        self.quality=torch.zeros(n,device=device);self.preferred=torch.zeros(n,dtype=torch.bool,device=device);self.accurate=torch.zeros(n,dtype=torch.bool,device=device)
        self.previous_distance=torch.zeros(n,device=device)
        self.new_shot=torch.zeros_like(self.accurate)
        self.entries=0;self.shots=0;self.accurate_shots=0;self.successful_timeouts=0

    def reset(self,done):
        for value in [self.age,self.stage,self.quality,self.preferred,self.accurate,self.previous_distance,self.new_shot]:value[done]=0
        self.first_pass[done]=-1;self.first_shot[done]=-1

    def begin_step(self):self.new_shot.zero_()

    def observe(self,velocity,direction,speed,kicked,preferred,eligible,age):
        valid,values=first_kick_samples(velocity,direction,speed,kicked,eligible)
        passed=valid&(self.stage==0)&(self.first_pass<0)
        self.first_pass[passed]=age[passed]
        shot=valid&(self.stage==1)&(self.first_shot<0)
        self.first_shot[shot]=age[shot]
        quality=torch.exp(-(values[:,2]/10).square()-(values[:,4]/.3).square())*preferred
        self.quality[shot]=quality[shot];self.preferred[shot]=preferred[shot]
        self.accurate[shot]=preferred[shot]&(values[shot,2]<=10)&(values[shot,4]<=.3)
        self.new_shot|=shot
        self.shots+=int(shot.sum());self.accurate_shots+=int((shot&self.accurate).sum())

    def ready(self):
        return (self.stage==0)&(self.first_pass>=0)&(self.age-self.first_pass>=self.pass_delay)

    def start_shooting(self,ids,distance):
        self.stage[ids]=1;self.previous_distance[ids]=distance[ids];self.entries+=len(ids)

    def advance_recovery(self):
        ready=(self.stage==1)&(self.first_shot>=0)&(self.age-self.first_shot>=self.shoot_delay)
        self.stage[ready]=2

    def skills(self):return (self.stage==1).long()

    def rewards(self,distance,stability,action_rate,terminated,truncated,dt):
        done=terminated|truncated;after_handoff=self.stage>0
        approach=(self.stage==1)&(self.first_shot<0)&~done
        progress=2*(self.previous_distance-distance)*approach
        # Direction/speed quality controls the large event and sustained stability bonuses.
        impact=(2*self.preferred.float()+10*self.quality+5*self.accurate)*self.new_shot
        recovery=5*dt*self.quality*stability*(self.first_shot>=0)*~done
        smooth=-.05*dt*action_rate.clamp(0,1)*(self.stage==1)*~done
        failure=-10*terminated*after_handoff
        success=self.accurate&truncated&~terminated&after_handoff
        finish=5*success.float();self.successful_timeouts+=int(success.sum())
        self.previous_distance=distance.clone()
        return torch.stack((progress,impact,recovery,smooth,failure,finish),-1)


class SequenceCritic(nn.Module):
    def __init__(self,planner,horizon_steps):
        super().__init__();self.horizon_steps=horizon_steps
        self.register_buffer('mean',critic_features(planner.obs_mean).clone())
        self.register_buffer('std',critic_features(planner.obs_std).clamp_min(.05).clone())
        self.net=nn.Sequential(nn.Linear(117,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,1))

    def forward(self,x,stage,age):
        features=((critic_features(x)-self.mean)/self.std).clamp(-20,20)
        phase=torch.nn.functional.one_hot(stage.long(),3).to(x.dtype)
        remaining=(1-age.to(x.dtype)/self.horizon_steps).clamp(0,1)[:,None]
        return self.net(torch.cat((features,phase,remaining),-1)).squeeze(-1)


def sample_shooting_ball(core,shooting,ids,seed=None):
    """Same event reset and source ball sampling as the continuous evaluation probe."""
    from .isolated_sampling import isolated_torch_rng,event_seed
    command=core.command_manager.get_term('motion');robot=core.scene['robot'].data
    keys=[k for k in vars(shooting.commands.motion) if k.startswith(('ball_init','ball_target','ball_launch','ball_sample_point'))]
    saved={key:copy.deepcopy(getattr(command.cfg,key)) for key in keys}
    command._freeze_first_ball_quality(ids)
    if command._event_aligned_reference_enabled():
        command._event_reference_started[ids]=False
        command._event_anchor_xy_offset_w[ids]=0.;command._event_reference_yaw_offset_w[ids]=0.
    try:
        for key in keys:setattr(command.cfg,key,copy.deepcopy(getattr(shooting.commands.motion,key)))
        def sample(selected):
            velocity=command._sample_ball_episode_state(selected,robot.root_pos_w[selected].clone(),
                robot.root_quat_w[selected].clone(),robot.root_pos_w[selected].clone(),
                robot.root_quat_w[selected].clone(),launch_from_current_step=True)
            command._reset_event_tracking_buffers(selected,reset_pass_quality=False)
            command._ball_last_vel_xy[selected]=velocity
        if seed is None:sample(ids)
        else:
            for i in ids.tolist():
                with isolated_torch_rng(event_seed(seed,i),core.device):sample(ids.new_tensor([i]))
        assert not command._event_reference_started[ids].any() and not command.ball_kicked[ids].any()
    finally:
        for key,value in saved.items():setattr(command.cfg,key,value)
    core.scene.write_data_to_sim();core.sim.forward()
    assert float(core.scene['ball'].data.root_lin_vel_w[ids].abs().max())<1e-5


def continuous_training_scene(passing,shooting,horizon_s=20.):
    from .shared_scene import apply_shooting_reset
    cfg=copy.deepcopy(shooting)
    cfg.observations=copy.deepcopy(passing.observations)
    cfg.commands.motion=copy.deepcopy(passing.commands.motion)
    cfg.scene.num_envs=passing.scene.num_envs;cfg.sim.device=passing.sim.device;cfg.seed=passing.seed
    cfg.commands.motion.debug_vis=False;apply_shooting_reset(cfg,shooting)
    term=cfg.terminations.event_kick_elapsed
    assert term.func.__name__=='event_kick_elapsed'
    cfg.terminations.event_kick_elapsed=None
    cfg.commands.motion.ball_resample_after_kick_enabled=False
    cfg.commands.motion.ball_reset_after_kick_enabled=False
    cfg.episode_length_s=horizon_s;cfg.is_finite_horizon=True
    return cfg
