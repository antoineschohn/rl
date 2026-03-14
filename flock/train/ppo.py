"""PPO: rollout collection, GAE, loss, and update step."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from flock.env.types import EnvConfig, Observations, RngKey, StepInfo
from flock.env.core import run_episodes
from flock.env.rules import Rules
from flock.train.policy import MLPPolicy, _flatten_obs


class Rollout(NamedTuple):
    """Training data from one rollout. All arrays have shape (T, n_agents, ...)."""
    obs: jax.Array          # (T, n_agents, obs_dim)
    actions: jax.Array      # (T, n_agents, 2)
    log_probs: jax.Array    # (T, n_agents)
    values: jax.Array       # (T, n_agents)
    rewards: jax.Array      # (T, n_agents)
    dones: jax.Array        # (T,)


class TrainConfig(NamedTuple):
    """PPO hyperparameters."""
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    n_minibatches: int = 4
    n_arenas: int = 16
    n_train_steps: int = 100


def _gaussian_log_prob(actions: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log probability of actions under diagonal Gaussian. Returns (n_agents,)."""
    var = jnp.exp(2 * log_std)
    log_p = -0.5 * (((actions - mean) ** 2) / var + 2 * log_std + jnp.log(2 * jnp.pi))
    return log_p.sum(axis=-1)


def make_training_hook(policy: MLPPolicy, team_idx: int):
    """Create a step_hook that collects training data for one team."""
    def hook(obs_per_team, actions_per_team, state, info):
        our_obs = obs_per_team[team_idx]
        our_actions = actions_per_team[team_idx]
        rewards = info.rewards[team_idx]

        obs_flat = _flatten_obs(our_obs)
        action_mean, action_log_std, values = jax.vmap(policy.forward_one)(obs_flat)
        log_probs = _gaussian_log_prob(our_actions, action_mean, action_log_std)

        return Rollout(
            obs=obs_flat,
            actions=our_actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=info.done,
        )
    return hook


def collect_rollouts(
    env_config: EnvConfig,
    rules: Rules,
    policies: tuple,
    key: RngKey,
    team_idx: int,
    n_arenas: int,
) -> Rollout:
    """Collect rollouts for one team across n_arenas, using run_episodes + hook."""
    hook = make_training_hook(policies[team_idx], team_idx)
    sim = run_episodes(env_config, rules, key, policies, n_arenas, step_hook=hook)
    return sim.extras


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute GAE advantages and returns via reverse scan."""
    T = rewards.shape[0]

    def scan_fn(gae, t):
        idx = T - 1 - t
        next_idx = jnp.minimum(idx + 1, T - 1)
        next_value = values[next_idx]
        not_done = 1.0 - dones[idx].astype(jnp.float32)
        delta = rewards[idx] + gamma * next_value * not_done - values[idx]
        gae = delta + gamma * gae_lambda * not_done * gae
        return gae, gae

    _, advantages_reversed = jax.lax.scan(
        scan_fn, jnp.zeros_like(values[0]), jnp.arange(T),
    )
    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    policy: MLPPolicy,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple[jax.Array, dict]:
    """PPO clipped objective + value loss + entropy bonus."""
    action_mean, action_log_std, values = jax.vmap(policy.forward_one)(obs)
    new_log_probs = _gaussian_log_prob(actions, action_mean, action_log_std)

    ratio = jnp.exp(new_log_probs - old_log_probs)
    normed_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * normed_advantages
    policy_loss = -jnp.minimum(ratio * normed_advantages, clipped).mean()

    value_loss = 0.5 * ((values - returns) ** 2).mean()

    entropy = 0.5 * (1 + jnp.log(2 * jnp.pi) + 2 * action_log_std).sum(axis=-1).mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    metrics = {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "total_loss": total_loss,
    }
    return total_loss, metrics


def train_step(
    policy: MLPPolicy,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    rollout: Rollout,
    train_config: TrainConfig,
    key: RngKey,
) -> tuple[MLPPolicy, optax.OptState, dict]:
    """One PPO update: compute GAE, then K epochs of minibatch updates."""
    advantages, returns = compute_gae(
        rollout.rewards, rollout.values, rollout.dones,
        train_config.gamma, train_config.gae_lambda,
    )

    T, n_agents = rollout.obs.shape[:2]
    flat_obs = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    flat_actions = rollout.actions.reshape(-1, 2)
    flat_log_probs = rollout.log_probs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)

    batch_size = T * n_agents
    minibatch_size = batch_size // train_config.n_minibatches

    metrics_acc = None

    for epoch in range(train_config.n_epochs):
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, batch_size)

        for mb in range(train_config.n_minibatches):
            idx = perm[mb * minibatch_size: (mb + 1) * minibatch_size]

            loss_fn = lambda p: ppo_loss(
                p, flat_obs[idx], flat_actions[idx], flat_log_probs[idx],
                flat_advantages[idx], flat_returns[idx],
                train_config.clip_eps, train_config.entropy_coef, train_config.value_coef,
            )
            (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(policy)

            updates, opt_state = optimizer.update(
                eqx.filter(grads, eqx.is_array), opt_state,
                eqx.filter(policy, eqx.is_array),
            )
            policy = eqx.apply_updates(policy, updates)

            if metrics_acc is None:
                metrics_acc = metrics
            else:
                metrics_acc = {k: metrics_acc[k] + v for k, v in metrics.items()}

    n_updates = train_config.n_epochs * train_config.n_minibatches
    metrics_avg = {k: v / n_updates for k, v in metrics_acc.items()}

    return policy, opt_state, metrics_avg
