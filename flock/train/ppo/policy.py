"""Learned policy: Gaussian MLP actor-critic."""

import jax
import jax.numpy as jnp
import equinox as eqx

from flock.env.types import Observations, Policy, PolicyState, RngKey


def obs_dim(k_teammates: int, k_opponents: int) -> int:
    """Flat observation size: own_vel(2) + teammates(k*4) + opponents(k*4)."""
    return 2 + k_teammates * 4 + k_opponents * 4


def flatten_obs(obs: Observations) -> jnp.ndarray:
    """Flatten Observations into (n_agents, obs_dim)."""
    n = obs.own_vel.shape[0]
    return jnp.concatenate([
        obs.own_vel,                          # (n, 2)
        obs.teammates.reshape(n, -1),         # (n, k_t * 4)
        obs.opponents.reshape(n, -1),         # (n, k_o * 4)
    ], axis=-1)


class ActorCritic(Policy):
    """Gaussian MLP actor-critic."""
    trunk: eqx.nn.MLP
    actor_mean: eqx.nn.Linear
    actor_log_std: jax.Array  # learnable (2,) parameter
    critic: eqx.nn.Linear

    def __init__(self, in_dim: int, hidden: int, *, key: RngKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.MLP(in_dim, hidden, width_size=hidden, depth=2, key=k1)
        self.actor_mean = eqx.nn.Linear(hidden, 2, key=k2)
        self.actor_log_std = jnp.zeros(2)
        self.critic = eqx.nn.Linear(hidden, 1, key=k3)

    def evaluate(self, obs: Observations) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass without sampling. Returns (mean, log_std, value) per agent."""
        x = flatten_obs(obs)  # (n_agents, obs_dim)
        h = jax.vmap(self.trunk)(x)  # (n_agents, hidden)
        mean = jax.vmap(self.actor_mean)(h)  # (n_agents, 2)
        log_std = jnp.broadcast_to(self.actor_log_std, mean.shape)  # (n_agents, 2)
        value = jax.vmap(self.critic)(h).squeeze(-1)  # (n_agents,)
        return mean, log_std, value

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        mean, log_std, _ = self.evaluate(obs)
        actions = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return actions, state


def make_policy(k_teammates: int, k_opponents: int, *, hidden: int = 64, key: RngKey) -> ActorCritic:
    """Create an ActorCritic policy for the given observation shape."""
    return ActorCritic(obs_dim(k_teammates, k_opponents), hidden, key=key)
