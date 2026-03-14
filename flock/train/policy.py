"""Neural network policy for PPO training."""

import equinox as eqx
import jax
import jax.numpy as jnp

from flock.env.types import Observations, Policy, PolicyState, RngKey


def _flatten_obs(obs: Observations) -> jax.Array:
    """Flatten batched observations into (n_agents, obs_dim) for the network."""
    n = obs.own_vel.shape[0]
    return jnp.concatenate([
        obs.own_vel,                              # (n, 2)
        obs.teammates.reshape(n, -1),             # (n, k_teammates * 4)
        obs.opponents.reshape(n, -1),             # (n, k_opponents * 4)
    ], axis=-1)


class MLPPolicy(Policy):
    """MLP policy with Gaussian action head and value head. Shared trunk."""
    trunk: eqx.nn.MLP
    action_mean: eqx.nn.Linear
    action_log_std: jax.Array  # learnable, (2,)
    value_head: eqx.nn.Linear

    def __init__(self, obs_dim: int, hidden_dim: int = 128, n_layers: int = 2, *, key: RngKey):
        k1, k2, k3 = jax.random.split(key, 3)
        self.trunk = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=n_layers,
            activation=jax.nn.relu,
            key=k1,
        )
        self.action_mean = eqx.nn.Linear(hidden_dim, 2, key=k2)
        self.action_log_std = jnp.zeros(2)
        self.value_head = eqx.nn.Linear(hidden_dim, 1, key=k3)

    def forward_one(self, obs_flat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass for a single agent. Returns (action_mean, action_log_std, value)."""
        h = self.trunk(obs_flat)
        action_mean = self.action_mean(h)
        value = self.value_head(h).squeeze(-1)
        return action_mean, self.action_log_std, value

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        """Sample actions for all agents. Returns (actions, state)."""
        obs_flat = _flatten_obs(obs)  # (n_agents, obs_dim)
        # vmap forward over agents
        action_mean, action_log_std, _value = jax.vmap(self.forward_one)(obs_flat)
        # Sample from Gaussian
        action_std = jnp.exp(action_log_std)
        actions = action_mean + action_std * jax.random.normal(key, action_mean.shape)
        return actions, state

    def evaluate(self, obs: Observations) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute action_mean, action_log_std, value for all agents (no sampling).

        Used by PPO to compute log_probs and value estimates.
        Returns: (action_mean, action_log_std, value) each (n_agents, ...).
        """
        obs_flat = _flatten_obs(obs)
        return jax.vmap(self.forward_one)(obs_flat)


def obs_dim(k_teammates: int, k_opponents: int) -> int:
    """Compute the flat observation dimension from config parameters."""
    return 2 + k_teammates * 4 + k_opponents * 4


def make_policy(k_teammates: int, k_opponents: int, *, key: RngKey, hidden_dim: int = 128, n_layers: int = 2) -> MLPPolicy:
    """Create an MLPPolicy with the right input size."""
    return MLPPolicy(
        obs_dim=obs_dim(k_teammates, k_opponents),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        key=key,
    )
