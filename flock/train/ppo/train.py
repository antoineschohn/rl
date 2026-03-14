"""Minimal PPO training loop for predator policy."""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from flock.env.types import EnvConfig, Observations
from flock.env.rules import Rules
from flock.env.core import reset, step
from flock.env.obs import observe
from flock.env.physics import pairwise_distances
from flock.train.ppo.policy import ActorCritic, flatten_obs


def _gaussian_log_prob(mean, log_std, actions):
    """Log probability of actions under diagonal Gaussian."""
    var = jnp.exp(2 * log_std)
    return -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + (actions - mean) ** 2 / var).sum(axis=-1)


def collect_rollout(env_config, rules, policy, prey_policy, key, n_arenas, distance_coeff=0.01):
    """Collect trajectories. Returns flat arrays over (n_arenas, T, n_predators)."""
    n_teams = len(rules.teams)
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

        # Dense reward: negative distance to nearest alive prey
        dists = pairwise_distances(new_state.teams[0].pos, new_state.teams[1].pos, env_config.arena_size)
        # Mask dead prey with inf so they don't attract
        dists = jnp.where(new_state.teams[1].alive[None, :], dists, jnp.inf)
        nearest_dist = jnp.minimum(dists.min(axis=1), env_config.arena_size)  # (n_predators,)
        distance_reward = -distance_coeff * nearest_dist

        pred_reward = info.rewards[0] + distance_reward  # (n_predators,)

        # Mask rewards after done
        alive_mask = (1 - done.astype(jnp.float32))
        pred_reward = pred_reward * alive_mask

        # Freeze on done
        done = done | info.done
        state = jax.tree.map(
            lambda old, new: jnp.where(done, old, new),
            state, new_state,
        )

        return (state, done, key, new_prey_ps), (pred_obs, pred_actions, log_prob, value, pred_reward, done, info.rewards[0] * alive_mask)

    def single_episode(key):
        key, reset_key = jax.random.split(key)
        init_state = reset(env_config, rules, reset_key)
        init_carry = (init_state, jnp.bool_(False), key, prey_ps)
        _, trajectory = jax.lax.scan(episode_step, init_carry, None, length=env_config.max_steps)
        return trajectory

    keys = jax.random.split(key, n_arenas)
    # obs, actions, log_probs, values, rewards, dones — each (n_arenas, T, ...)
    return jax.vmap(single_episode)(keys)


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE-lambda. All inputs: (T, n_agents). Returns advantages, returns."""
    T = rewards.shape[0]

    def scan_fn(carry, t):
        next_value, gae = carry
        # Reverse scan: t goes T-1, T-2, ..., 0
        idx = T - 1 - t
        delta = rewards[idx] + gamma * next_value * (1 - dones[idx, None]) - values[idx]
        gae = delta + gamma * lam * (1 - dones[idx, None]) * gae
        return (values[idx], gae), gae

    n_agents = rewards.shape[1]
    _, advantages = jax.lax.scan(scan_fn, (jnp.zeros(n_agents), jnp.zeros(n_agents)), jnp.arange(T))
    # Reverse back to normal time order
    advantages = advantages[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_loss(policy, obs_flat, actions, old_log_probs, advantages, returns, clip_eps=0.2):
    """PPO clipped surrogate loss + value loss. All inputs: (batch, ...)."""
    # Reconstruct Observations for evaluate — obs_flat is already (batch, obs_dim)
    # We need to call trunk + heads directly on flat obs
    h = jax.vmap(policy.trunk)(obs_flat)
    mean = jax.vmap(policy.actor_mean)(h)
    log_std = jnp.broadcast_to(policy.actor_log_std, mean.shape)
    value = jax.vmap(policy.critic)(h).squeeze(-1)

    log_probs = _gaussian_log_prob(mean, log_std, actions)
    ratio = jnp.exp(log_probs - old_log_probs)

    # Clipped surrogate
    adv_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * adv_normalized
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_normalized
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    # Value loss
    value_loss = 0.5 * ((value - returns) ** 2).mean()

    # Entropy bonus
    entropy = (0.5 * jnp.log(2 * jnp.pi * jnp.e) + log_std).sum(axis=-1).mean()

    return policy_loss + 0.5 * value_loss - 0.01 * entropy


@eqx.filter_jit
def train_step(policy, opt_state, obs_flat, actions, old_log_probs, advantages, returns, optimizer):
    """One PPO gradient step."""
    loss, grads = eqx.filter_value_and_grad(ppo_loss)(
        policy, obs_flat, actions, old_log_probs, advantages, returns,
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(policy, eqx.is_array))
    new_policy = eqx.apply_updates(policy, updates)
    return new_policy, new_opt_state, loss


def train(
    env_config: EnvConfig,
    rules: Rules,
    policy: ActorCritic,
    prey_policy,
    key,
    n_iters: int = 100,
    n_arenas: int = 32,
    n_epochs: int = 4,
    lr: float = 3e-4,
):
    """Main PPO training loop."""
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    for i in range(n_iters):
        key, rollout_key = jax.random.split(key)

        # Collect rollout
        obs, actions, log_probs, values, rewards, dones, catch_rewards = collect_rollout(
            env_config, rules, policy, prey_policy, rollout_key, n_arenas,
        )
        # obs is a NamedTuple of (n_arenas, T, n_agents, ...) — flatten obs for loss
        obs_flat = jnp.concatenate([
            obs.own_vel,
            obs.teammates.reshape(*obs.own_vel.shape[:2], obs.own_vel.shape[2], -1),
            obs.opponents.reshape(*obs.own_vel.shape[:2], obs.own_vel.shape[2], -1),
        ], axis=-1)  # (n_arenas, T, n_agents, obs_dim)

        # shapes: (n_arenas, T, n_agents, ...) → flatten to (n_arenas * T * n_agents, ...)
        na, T, n_pred = actions.shape[:3]

        # Compute GAE per arena — vmap over arenas
        dones_float = dones.astype(jnp.float32)  # (n_arenas, T)
        advantages, returns = jax.vmap(compute_gae)(rewards, values, dones_float)
        # (n_arenas, T, n_agents)

        # Flatten everything to (batch, ...)
        batch_size = na * T * n_pred
        obs_b = obs_flat.reshape(batch_size, -1)
        act_b = actions.reshape(batch_size, 2)
        lp_b = log_probs.reshape(batch_size)
        adv_b = advantages.reshape(batch_size)
        ret_b = returns.reshape(batch_size)

        for _epoch in range(n_epochs):
            policy, opt_state, loss = train_step(
                policy, opt_state, obs_b, act_b, lp_b, adv_b, ret_b, optimizer,
            )

        mean_reward = rewards.sum(axis=1).mean()
        mean_catches = catch_rewards.sum(axis=1).mean()
        print(f"iter {i:4d} | loss {loss:.4f} | reward {mean_reward:.2f} | catches {mean_catches:.2f}")

    return policy
