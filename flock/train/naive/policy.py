"""Naive hand-coded policies: chase nearest opponent / flee nearest opponent."""

import jax
import jax.numpy as jnp

from flock.env.types import Observations, Policy, PolicyState, RngKey, TeamConfig


class ChasePrey(Policy):
    """Predator policy: accelerate toward the nearest opponent (prey)."""
    n_agents: int
    max_accel: float

    def __init__(self, team_config: TeamConfig):
        self.n_agents = team_config.n_agents
        self.max_accel = team_config.max_accel

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        # obs.opponents: (n_agents, k_opponents, 4) — (dx, dy, dvx, dvy)
        rel_pos = obs.opponents[:, :, :2]  # (n_agents, k_opponents, 2)
        dists = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, k_opponents)
        nearest = jnp.argmin(dists, axis=-1)  # (n_agents,)
        # Gather the displacement to nearest opponent
        target = rel_pos[jnp.arange(self.n_agents), nearest]  # (n_agents, 2)
        # Normalize and scale to max acceleration
        dist = jnp.linalg.norm(target, axis=-1, keepdims=True)
        direction = target / jnp.maximum(dist, 1e-6)
        actions = direction * self.max_accel
        return actions, state


class FleePredators(Policy):
    """Prey policy: accelerate away from the nearest opponent (predator)."""
    n_agents: int
    max_accel: float

    def __init__(self, team_config: TeamConfig):
        self.n_agents = team_config.n_agents
        self.max_accel = team_config.max_accel

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        rel_pos = obs.opponents[:, :, :2]  # (n_agents, k_opponents, 2)
        dists = jnp.linalg.norm(rel_pos, axis=-1)  # (n_agents, k_opponents)
        nearest = jnp.argmin(dists, axis=-1)  # (n_agents,)
        target = rel_pos[jnp.arange(self.n_agents), nearest]  # (n_agents, 2)
        dist = jnp.linalg.norm(target, axis=-1, keepdims=True)
        direction = target / jnp.maximum(dist, 1e-6)
        # Flee: accelerate in the opposite direction
        actions = -direction * self.max_accel
        return actions, state
