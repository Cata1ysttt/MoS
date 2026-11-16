"""Validate and restore PPO learning state without claiming simulator restoration."""
import torch

MATCHED_OPTIONS=('seed','num_envs','rollout_steps','epochs','minibatches','learning_rate',
                 'critic_learning_rate','exploration_std','target_kl','anchor_weight',
                 'reference_reward_scale','smoke_test')


def validate_resume(config, complete, current, base_hash):
    if config.get('training')!='passing_latent_ppo_v1':raise ValueError('Resume requires a PPO training run')
    if config['parent_sha256']!=base_hash:raise ValueError('Original frozen base changed')
    if not complete.get('shared_tensors_unchanged') or not complete.get('deployable_roundtrip'):
        raise ValueError('Source PPO run did not complete its preservation checks')
    for key in MATCHED_OPTIONS:
        previous=config.get(key,1. if key=='reference_reward_scale' else None)
        if previous!=current[key]:raise ValueError('Resume configuration mismatch: '+key)
    if complete['smoke_test']!=current['smoke_test']:raise ValueError('Cannot turn a smoke run into a candidate')
    return dict(iterations=complete.get('total_iterations',complete['iterations']),
        environment_steps=complete.get('total_environment_steps',
            complete['iterations']*config['rollout_steps']*config['num_envs']))


def assert_tree_equal(left,right):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left.detach().cpu(),right.detach().cpu()):
            raise ValueError('Restored tensor differs')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise ValueError('Restored keys differ')
        for key in left:assert_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        if len(left)!=len(right):raise ValueError('Restored sequence differs')
        for old,new in zip(left,right):assert_tree_equal(old,new)
    elif left!=right:raise ValueError('Restored scalar differs')


def restore_learning_state(planner,critic,optimizer,critic_optimizer,generator,payload,state,fallback_seed):
    planner.load_state_dict(payload['planner'])
    critic.load_state_dict(state['critic'])
    optimizer.load_state_dict(state['optimizer'])
    critic_optimizer.load_state_dict(state['critic_optimizer'])
    assert_tree_equal(payload['planner'],planner.state_dict())
    assert_tree_equal(state['critic'],critic.state_dict())
    assert_tree_equal(state['optimizer'],optimizer.state_dict())
    assert_tree_equal(state['critic_optimizer'],critic_optimizer.state_dict())
    restored='generator_state' in state
    if restored:
        generator.set_state(state['generator_state'].cpu())
        assert_tree_equal(state['generator_state'],generator.get_state())
    else:generator.manual_seed(fallback_seed)
    return dict(planner_tensors_exact=True,critic_tensors_exact=True,optimizer_states_exact=True,
        exploration_rng_restored=restored,fallback_exploration_seed=None if restored else fallback_seed,
        simulator_restored=False,observation_histories_restored=False)
