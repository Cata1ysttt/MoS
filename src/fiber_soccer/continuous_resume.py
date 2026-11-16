"""Guard fixed-objective continuous PPO continuation; simulator starts fresh."""

CONTINUOUS_MATCHED_OPTIONS = (
    'seed', 'num_envs', 'rollout_steps', 'epochs', 'minibatches',
    'learning_rate', 'critic_learning_rate', 'exploration_std',
    'target_kl', 'anchor_weight', 'smoke_test',
)
CONTINUOUS_OBJECTIVE = dict(
    training='continuous_post_pass_shooting_ppo_v1', gamma=.995,
    gae_lambda=.97, finite_horizon=True, timeout_bootstrap=False,
    expert_actions_used=False, pass_delay=50, shoot_delay=10,
    checkpoint_selection='final_iteration',
)


def validate_continuous_resume(config, complete, current, base_hash):
    if config.get('parent_sha256') != base_hash:
        raise ValueError('Original frozen base changed')
    for key, value in CONTINUOUS_OBJECTIVE.items():
        if config.get(key) != value:
            raise ValueError('Continuous objective mismatch: ' + key)
    for key in CONTINUOUS_MATCHED_OPTIONS:
        if key not in config or key not in current or config[key] != current[key]:
            raise ValueError('Resume configuration mismatch: ' + key)
    if complete.get('smoke_test') != current['smoke_test']:
        raise ValueError('Cannot change smoke status on continuation')
    if not complete.get('shared_tensors_unchanged') or not complete.get('deployable_roundtrip'):
        raise ValueError('Source lacks completed preservation checks')
    if complete.get('shooting_entries', 0) <= 0 or complete.get('active_policy_steps', 0) <= 0:
        raise ValueError('Source did not exercise continuous shooting')
    if complete.get('handoff_history_checks') != complete['shooting_entries']:
        raise ValueError('Source lacks complete handoff history checks')
    iterations = complete.get('total_iterations', complete['iterations'])
    steps = complete.get('total_environment_steps', complete['environment_steps'])
    if iterations <= 0 or steps != iterations * config['num_envs'] * config['rollout_steps']:
        raise ValueError('Inconsistent prior training accounting')
    return dict(iterations=iterations, environment_steps=steps)
