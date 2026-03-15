"""PPO training loop for arbitrary teams."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from flock.env.types import EnvConfig, Policy
from flock.env.rules import Rules
from flock.env.core import reset, step
from flock.env.obs import observe
from flock.train.ppo.policy import ActorCritic, flatten_obs


class PPOConfig(NamedTuple):
    """All PPO hyperparameters in one place."""
    # GAE
    gamma: float = 0.99
    lam: float = 0.95
    # Clipped surrogate
    clip_eps: float = 0.2
    # Loss coefficients
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    # Training loop
    n_iters: int = 100
    n_arenas: int = 128
    n_epochs: int = 4
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    minibatch_size: int = 4096


class Trainee(NamedTuple):
    """Specifies which team to train and with what reward."""
    team_idx: int
    reward_fn: object  # callable (env_config, rules, state, new_state, info, team_idx) -> (n_agents,)


def _gaussian_log_prob(mean, log_std, actions):
    """Log probability of actions under diagonal Gaussian."""
    var = jnp.exp(2 * log_std)
    return -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + (actions - mean) ** 2 / var).sum(axis=-1)


def collect_rollout(env_config, rules, policies, trainees, key, n_arenas):
    """Collect trajectories for all trainees from shared rollouts.

    Returns:
        per_team: list of (trajectory, last_values) per trainee
        env_scores: (n_arenas, T, n_teams_in_env) raw env scores for logging
    """
    n_teams = len(rules.teams)
    policy_states = tuple(p.init_state() for p in policies)

    trainee_indices = [t.team_idx for t in trainees]

    def episode_step(carry, _):
        state, done, key, pstates = carry
        keys = jax.random.split(key, n_teams + 1)
        key = keys[0]

        # Observe and act for each team
        obs_all = tuple(observe(rules, env_config, state, i) for i in range(n_teams))
        actions_and_states = tuple(
            policies[i](obs_all[i], keys[i + 1], pstates[i])
            for i in range(n_teams)
        )
        team_actions = tuple(a for a, _ in actions_and_states)
        new_pstates = tuple(s for _, s in actions_and_states)

        # For trainees that are ActorCritic, get value estimates and log probs
        trainee_data = []
        for t in trainees:
            i = t.team_idx
            mean, log_std, value = policies[i].evaluate(obs_all[i])
            log_prob = _gaussian_log_prob(mean, log_std, team_actions[i])
            trainee_data.append((obs_all[i], team_actions[i], log_prob, value))

        # Step environment
        new_state, info = step(env_config, rules, state, team_actions)

        # Compute rewards per trainee
        alive_mask = 1 - done.astype(jnp.float32)
        trainee_rewards = []
        for t in trainees:
            r = t.reward_fn(env_config, rules, state, new_state, info, team_idx=t.team_idx)
            trainee_rewards.append(r * alive_mask)

        # Env scores for logging (per trainee)
        trainee_scores = tuple(
            info.scores[t.team_idx] * alive_mask for t in trainees
        )

        # Freeze on done
        done = done | info.done
        state = jax.tree.map(
            lambda old, new: jnp.where(done, old, new),
            state, new_state,
        )

        # Pack per-trainee outputs
        per_trainee_out = tuple(
            (td[0], td[1], td[2], td[3], tr, done, ts)
            for td, tr, ts in zip(trainee_data, trainee_rewards, trainee_scores)
        )

        return (state, done, key, new_pstates), per_trainee_out

    def single_episode(key):
        key, reset_key = jax.random.split(key)
        init_state = reset(env_config, rules, reset_key)
        init_carry = (init_state, jnp.bool_(False), key, policy_states)
        (final_state, final_done, _, _), trajectories = jax.lax.scan(
            episode_step, init_carry, None, length=env_config.max_steps,
        )

        # Bootstrap values per trainee
        last_values = []
        for t in trainees:
            final_obs = observe(rules, env_config, final_state, t.team_idx)
            _, _, lv = policies[t.team_idx].evaluate(final_obs)
            lv = jnp.where(final_done, 0.0, lv)
            last_values.append(lv)

        return trajectories, tuple(last_values)

    keys = jax.random.split(key, n_arenas)
    trajectories, last_values = jax.vmap(single_episode)(keys)
    # trajectories: tuple of n_trainees, each element is tuple of arrays (n_arenas, T, ...)
    # last_values: tuple of n_trainees, each (n_arenas, n_agents)

    per_team = [(trajectories[i], last_values[i]) for i in range(len(trainees))]
    return per_team


def compute_gae(rewards, values, dones, last_value, cfg):
    """GAE-lambda. All inputs: (T, n_agents). last_value: (n_agents,) bootstrap for truncation."""
    gamma, lam = cfg.gamma, cfg.lam
    T = rewards.shape[0]

    def scan_fn(carry, t):
        next_value, gae = carry
        idx = T - 1 - t
        delta = rewards[idx] + gamma * next_value * (1 - dones[idx, None]) - values[idx]
        gae = delta + gamma * lam * (1 - dones[idx, None]) * gae
        return (values[idx], gae), gae

    n_agents = rewards.shape[1]
    _, advantages = jax.lax.scan(scan_fn, (last_value, jnp.zeros(n_agents)), jnp.arange(T))
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(policy, obs_flat, actions, old_log_probs, advantages, returns, cfg):
    """PPO clipped surrogate loss + value loss. All inputs: (batch, ...)."""
    h = jax.vmap(policy.trunk)(obs_flat)
    mean = jax.vmap(policy.actor_mean)(h)
    log_std = jnp.broadcast_to(policy.actor_log_std, mean.shape)
    value = jax.vmap(policy.critic)(h).squeeze(-1)

    log_probs = _gaussian_log_prob(mean, log_std, actions)
    ratio = jnp.exp(log_probs - old_log_probs)

    # Clipped surrogate
    adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * adv_normalized
    surr2 = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_normalized
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    # Value loss
    value_loss = 0.5 * ((value - returns) ** 2).mean()

    # Entropy bonus
    entropy = (0.5 * jnp.log(2 * jnp.pi * jnp.e) + log_std).sum(axis=-1).mean()

    return policy_loss + cfg.value_coeff * value_loss - cfg.entropy_coeff * entropy


@eqx.filter_jit
def train_step(policy, opt_state, obs_flat, actions, old_log_probs, advantages, returns, optimizer, cfg):
    """One PPO gradient step."""
    loss, grads = eqx.filter_value_and_grad(ppo_loss)(
        policy, obs_flat, actions, old_log_probs, advantages, returns, cfg,
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(policy, eqx.is_array))
    new_policy = eqx.apply_updates(policy, updates)
    return new_policy, new_opt_state, loss


def _update_policy(policy, opt_state, obs, actions, log_probs, values, rewards, dones, last_values, optimizer, cfg):
    """Run PPO epochs for a single policy. Returns (policy, opt_state, loss)."""
    obs_flat = jax.vmap(jax.vmap(flatten_obs))(obs)

    na, T, n_agents = actions.shape[:3]

    # Compute GAE per arena
    dones_float = dones.astype(jnp.float32)
    advantages, returns = jax.vmap(lambda r, v, d, lv: compute_gae(r, v, d, lv, cfg))(
        rewards, values, dones_float, last_values,
    )

    # Flatten to (batch, ...)
    batch_size = na * T * n_agents
    obs_b = obs_flat.reshape(batch_size, -1)
    act_b = actions.reshape(batch_size, 2)
    lp_b = log_probs.reshape(batch_size)
    adv_b = advantages.reshape(batch_size)
    ret_b = returns.reshape(batch_size)

    key = jax.random.key(0)
    for _epoch in range(cfg.n_epochs):
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, batch_size)
        for start in range(0, batch_size, cfg.minibatch_size):
            idx = perm[start:start + cfg.minibatch_size]
            policy, opt_state, loss = train_step(
                policy, opt_state,
                obs_b[idx], act_b[idx], lp_b[idx], adv_b[idx], ret_b[idx],
                optimizer, cfg,
            )

    return policy, opt_state, loss


def train(
    env_config: EnvConfig,
    rules: Rules,
    policies: tuple[Policy, ...],
    trainees: list[Trainee],
    key,
    cfg: PPOConfig,
):
    """Main PPO training loop.

    Args:
        policies: one Policy per team, in team order.
        trainees: list of Trainee specifying which teams to train and with what reward.
            Teams not listed in trainees are used as-is (frozen opponents).
    """
    # Set up per-trainee optimizers
    optimizers = []
    opt_states = []
    for t in trainees:
        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.lr),
        )
        optimizers.append(optimizer)
        opt_states.append(optimizer.init(eqx.filter(policies[t.team_idx], eqx.is_array)))

    for i in range(cfg.n_iters):
        key, rollout_key = jax.random.split(key)

        # Collect rollout for all teams simultaneously
        per_team = collect_rollout(
            env_config, rules, policies, trainees, rollout_key, cfg.n_arenas,
        )

        # Update each trainee's policy
        losses = []
        for ti, t in enumerate(trainees):
            (obs, actions, log_probs, values, rewards, dones, _scores), last_values = per_team[ti]

            policies_list = list(policies)
            policies_list[t.team_idx], opt_states[ti], loss = _update_policy(
                policies[t.team_idx], opt_states[ti],
                obs, actions, log_probs, values, rewards, dones, last_values,
                optimizers[ti], cfg,
            )
            policies = tuple(policies_list)
            losses.append(loss)

        # Logging
        parts = [f"iter {i:4d}"]
        for ti, t in enumerate(trainees):
            (_obs, _act, _lp, _val, rewards, _dones, scores), _lv = per_team[ti]
            name = rules.teams[t.team_idx].name
            mean_reward = rewards.sum(axis=1).mean()
            mean_score = scores.sum(axis=1).mean()
            parts.append(f"{name}: loss {losses[ti]:.4f} reward {mean_reward:.2f} score {mean_score:.2f}")
        print(" | ".join(parts))

    return policies
