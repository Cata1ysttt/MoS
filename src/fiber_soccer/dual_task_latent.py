"""Two task-reward latent heads around the complete-history frozen planner."""
from pathlib import Path
import torch
from torch import nn
from .initial_goal_recovery import InitialGoalRecoveryPolicy
from .post_pass_latent_adapter import LatentResidual
from .new_ball_rma import initialize_new_ball_history
from .new_ball_task_history import initialize_new_ball_task_history
from .joint_quality import JointQualityPhase


class DualTaskResidual(nn.Module):
    def __init__(self, mean, std, width=128):
        super().__init__()
        self.width = width
        self.heads = nn.ModuleList([LatentResidual(mean, std, width) for _ in range(2)])

    def forward(self, x, base):
        return torch.where((x[:, -1] > .5)[:, None], self.heads[1](x, base), self.heads[0](x, base))


class DualTaskPolicy(InitialGoalRecoveryPolicy):
    def __init__(self, checkpoint, device='cpu', **kwargs):
        assert checkpoint['new_ball_history_mode'] == 'initialize_shoot_ball_task'
        config = checkpoint['dual_task_adapter']
        assert config['version'] == 1
        super().__init__(checkpoint, device, **kwargs)
        self.task_adapter = DualTaskResidual(self.planner.obs_mean, self.planner.obs_std, config['width'])
        self.task_adapter.load_state_dict(config['model'], strict=True)
        self.task_adapter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, observation, skill=None, target_direction=None, target_speed=None):
        raw = self.condition_observation(observation, skill, target_direction, target_speed)
        output = super().forward(raw)
        memory = self.new_ball_history
        x = initialize_new_ball_history(raw, memory.age-1, memory.initial_measurement)
        x = initialize_new_ball_task_history(x, memory.age-1, memory.initial_task_measurement)
        active = x[:, 940] < .5
        self.last_adapter_input = x
        self.last_base_latent = output['latent']
        self.last_adapter_active = active
        z = self.task_adapter(x, output['latent'])
        action = self.actor(x[:, :64], x[:, 64:93], x[:, 93:930], z)
        scaled = (6*action).clamp(-6, 6)
        proposed = dict(latent=z, action_normalized=action, action_scaled=scaled,
                        joint_targets=self.default+scaled*self.scale)
        return {key: torch.where(active[:, None], proposed[key], value) for key, value in output.items()}


def load_dual_task_policy(checkpoint, device='cpu', **kwargs):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'dual_task_adapter' not in checkpoint:
        from .planner_policy import load_soccer_policy
        return load_soccer_policy(checkpoint, device, **kwargs)
    with torch.inference_mode(False):
        return DualTaskPolicy(checkpoint, device, **kwargs)


class DualTaskPhase(JointQualityPhase):
    """Add pass quality events and pre-pass shaping to existing shot rewards."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_pass = torch.zeros_like(self.new_shot)

    def reset(self, done):
        super().reset(done)
        self.new_pass[done] = False

    def begin_step(self):
        super().begin_step()
        self.new_pass.zero_()

    def observe(self, *args, **kwargs):
        before = self.first_pass.clone()
        super().observe(*args, **kwargs)
        self.new_pass |= (before < 0) & (self.first_pass >= 0)

    def rewards(self, distance, stability, action_rate, terminated, truncated, dt):
        old_distance = self.previous_distance.clone()
        original = super().rewards(distance, stability, action_rate, terminated, truncated, dt)
        approaching = (self.stage == 0) & (self.first_pass < 0) & ~(terminated | truncated)
        progress = 2*(old_distance-distance)*approaching
        impact = (10*self.pass_quality+5*self.pass_accurate)*self.new_pass
        stable = .5*dt*stability*approaching
        failure = -10*terminated*(self.stage == 0)
        smooth = -.05*dt*action_rate.clamp(0, 1)*approaching
        return torch.cat((original, torch.stack((progress, impact, stable, failure, smooth), -1)), -1)
