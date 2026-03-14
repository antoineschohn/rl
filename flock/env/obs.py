"""Observation extraction — the information bottleneck between state and policy."""

import jax
import jax.numpy as jnp

from flock.env.types import Agents, EnvConfig, EnvState, Observations
from flock.env.physics import pairwise_deltas


def _k_nearest(deltas: jnp.ndarray, rel_vel: jnp.ndarray, alive: jnp.ndarray, k: int) -> jnp.ndarray:
    """For one agent, find k-nearest others and return relative (dx, dy, dvx, dvy).

    Args:
        deltas: (n_others, 2) — displacement vectors to others
        rel_vel: (n_others, 2) — relative velocities of others
        alive: (n_others,) — alive mask
        k: number of slots

    Returns:
        (k, 4) — relative pos + vel of k-nearest, zero-padded
    """
    dists = jnp.linalg.norm(deltas, axis=-1)  # (n_others,)
    # Dead/masked agents get infinite distance
    dists = jnp.where(alive, dists, jnp.inf)
    nearest_idx = jnp.argsort(dists)[:k]
    rel_pos = deltas[nearest_idx]          # (k, 2)
    rel_v = rel_vel[nearest_idx]           # (k, 2)
    result = jnp.concatenate([rel_pos, rel_v], axis=-1)  # (k, 4)
    # Zero out invalid slots
    valid = alive[nearest_idx][:, None]    # (k, 1)
    return result * valid


def _observe_one_agent(
    own_vel: jnp.ndarray,
    teammate_deltas: jnp.ndarray,
    teammate_rel_vel: jnp.ndarray,
    teammate_alive: jnp.ndarray,
    opponent_deltas: jnp.ndarray,
    opponent_rel_vel: jnp.ndarray,
    opponent_alive: jnp.ndarray,
    k_teammates: int,
    k_opponents: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build observation for a single agent. Designed to be vmapped."""
    teammates = _k_nearest(teammate_deltas, teammate_rel_vel, teammate_alive, k_teammates)
    opponents = _k_nearest(opponent_deltas, opponent_rel_vel, opponent_alive, k_opponents)
    return own_vel, teammates, opponents


def observe(config: EnvConfig, state: EnvState, team: str) -> Observations:
    """Extract observations for a team. Policies only see this, never EnvState.

    Args:
        team: "predators" or "prey"
    """
    if team == "predators":
        us = state.predators
        them = state.prey
    else:
        us = state.prey
        them = state.predators

    n_us = us.pos.shape[0]

    # (n_us, n_us, 2) — from each agent to each teammate
    teammate_deltas = pairwise_deltas(us.pos, us.pos, config.arena_size)
    # Relative velocities: teammate_vel - own_vel
    # (n_us, 1, 2) broadcast with (1, n_us, 2) -> (n_us, n_us, 2)
    teammate_rel_vel = us.vel[None, :, :] - us.vel[:, None, :]

    # Exclude self: mask diagonal as not-alive
    self_mask = ~jnp.eye(n_us, dtype=jnp.bool_)  # (n_us, n_us) True=valid
    teammate_alive = us.alive[None, :] & self_mask  # (n_us, n_us)

    # (n_us, n_them, 2) — from each of us to each opponent
    # pairwise_deltas(a, b) = a[:, None] - b[None, :], so we want them in rows, us in cols
    opponent_deltas = pairwise_deltas(them.pos, us.pos, config.arena_size)  # (n_them, n_us, 2)
    opponent_deltas = jnp.transpose(opponent_deltas, (1, 0, 2))  # (n_us, n_them, 2)
    # Relative velocities: opponent_vel - own_vel
    opponent_rel_vel = them.vel[None, :, :] - us.vel[:, None, :]  # (n_us, n_them, 2)
    # (n_us, n_them) — broadcast alive mask
    opponent_alive = jnp.broadcast_to(them.alive[None, :], (n_us, them.pos.shape[0]))

    # vmap over agents in the team (axis 0 of all per-agent arrays)
    own_vel, teammates, opponents = jax.vmap(
        _observe_one_agent,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
    )(
        us.vel,
        teammate_deltas,
        teammate_rel_vel,
        teammate_alive,
        opponent_deltas,
        opponent_rel_vel,
        opponent_alive,
        config.k_teammates,
        config.k_opponents,
    )

    return Observations(own_vel=own_vel, teammates=teammates, opponents=opponents)
