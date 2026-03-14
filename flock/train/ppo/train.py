"""Minimal PPO training loop for predator policy."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from flock.env.types import EnvConfig
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


def _gaussian_log_prob(mean, log_std, actions):
    """Log probability of actions under diagonal Gaussian."""
    var = jnp.exp(2 * log_std)
    return -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + (actions - mean) ** 2 / var).sum(axis=-1)


def collect_rollout(env_config, rules, policy, prey_policy, reward_fn, key, n_arenas):
    """Collect trajectories. Returns flat arrays over (n_arenas, T, n_predators) + last_values for bootstrap."""
    prey_ps = prey_policy.init_state()

    def episode_step(carry, _):
        state, done, key, prey_ps = carry
        keys = jax.random.split(key, 3)
        key = keys[0]

        # Observations
        pred_obs = observe(rules, env_config, state, 0)
        prey_obs = observe(rules, env_config, state, 1)

        # Policy forward pass (predator)
        mean, log_std, value = policy.evaluate(pred_obs)
        pred_actions = mean + jnp.exp(log_std) * jax.random.normal(keys[1], mean.shape)
        log_prob = _gaussian_log_prob(mean, log_std, pred_actions)

        # Prey policy
        prey_actions, new_prey_ps = prey_policy(prey_obs, keys[2], prey_ps)

        # Step
        new_state, info = step(env_config, rules, state, (pred_actions, prey_actions))

        # Reward
        pred_reward = reward_fn(env_config, rules, state, new_state, info, team_idx=0)

        # Mask rewards after done
        alive_mask = (1 - done.astype(jnp.float32))
        pred_reward = pred_reward * alive_mask

        # Freeze on done
        done = done | info.done
        state = jax.tree.map(
            lambda old, new: jnp.where(done, old, new),
            state, new_state,
        )

        return (state, done, key, new_prey_ps), (pred_obs, pred_actions, log_prob, value, pred_reward, done, info.scores[0] * alive_mask)

    def single_episode(key):
        key, reset_key = jax.random.split(key)
        init_state = reset(env_config, rules, reset_key)
        init_carry = (init_state, jnp.bool_(False), key, prey_ps)
        (final_state, final_done, _, _), trajectory = jax.lax.scan(
            episode_step, init_carry, None, length=env_config.max_steps,
        )
        # Bootstrap value: V(s_T) for truncated episodes, 0 if truly done
        final_obs = observe(rules, env_config, final_state, 0)
        _, _, last_value = policy.evaluate(final_obs)
        last_value = jnp.where(final_done, 0.0, last_value)
        return trajectory, last_value

    keys = jax.random.split(key, n_arenas)
    # obs, actions, log_probs, values, rewards, dones — each (n_arenas, T, ...)
    trajectories, last_values = jax.vmap(single_episode)(keys)
    return trajectories, last_values


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


def train(
    env_config: EnvConfig,
    rules: Rules,
    policy: ActorCritic,
    prey_policy,
    reward_fn,
    key,
    cfg: PPOConfig,
):
    """Main PPO training loop."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.lr),
    )
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    for i in range(cfg.n_iters):
        key, rollout_key = jax.random.split(key)

        # Collect rollout
        (obs, actions, log_probs, values, rewards, dones, catch_rewards), last_values = collect_rollout(
            env_config, rules, policy, prey_policy, reward_fn, rollout_key, cfg.n_arenas,
        )

        # Flatten obs — (n_arenas, T, n_agents, obs_dim)
        obs_flat = jax.vmap(jax.vmap(flatten_obs))(obs)

        # shapes: (n_arenas, T, n_agents, ...) → flatten to (n_arenas * T * n_agents, ...)
        na, T, n_pred = actions.shape[:3]

        # Compute GAE per arena with bootstrap
        dones_float = dones.astype(jnp.float32)  # (n_arenas, T)
        advantages, returns = jax.vmap(lambda r, v, d, lv: compute_gae(r, v, d, lv, cfg))(
            rewards, values, dones_float, last_values,
        )
        # (n_arenas, T, n_agents)

        # Flatten everything to (batch, ...)
        batch_size = na * T * n_pred
        obs_b = obs_flat.reshape(batch_size, -1)
        act_b = actions.reshape(batch_size, 2)
        lp_b = log_probs.reshape(batch_size)
        adv_b = advantages.reshape(batch_size)
        ret_b = returns.reshape(batch_size)

        for _epoch in range(cfg.n_epochs):
            # Shuffle and split into minibatches
            key, shuffle_key = jax.random.split(key)
            perm = jax.random.permutation(shuffle_key, batch_size)
            for start in range(0, batch_size, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                policy, opt_state, loss = train_step(
                    policy, opt_state,
                    obs_b[idx], act_b[idx], lp_b[idx], adv_b[idx], ret_b[idx],
                    optimizer, cfg,
                )

        mean_reward = rewards.sum(axis=1).mean()
        mean_catches = catch_rewards.sum(axis=1).mean()
        print(f"iter {i:4d} | loss {loss:.4f} | reward {mean_reward:.2f} | catches {mean_catches:.2f}")

    return policy
