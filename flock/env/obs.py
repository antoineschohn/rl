"""Observation extraction — the information bottleneck between state and policy."""

import jax
import jax.numpy as jnp

from flock.env.types import Agents, EnvConfig, EnvState, Observations
from flock.env.rules import Rules
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
    dists = jnp.linalg.norm(deltas, axis=-1)
    dists = jnp.where(alive, dists, jnp.inf)
    nearest_idx = jnp.argsort(dists)[:k]
    rel_pos = deltas[nearest_idx]
    rel_v = rel_vel[nearest_idx]
    result = jnp.concatenate([rel_pos, rel_v], axis=-1)
    valid = alive[nearest_idx][:, None]
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


def observe(rules: Rules, env_config: EnvConfig, state: EnvState, team_idx: int) -> Observations:
    """Extract observations for a team. Policies only see this, never EnvState."""
    tc = rules.teams[team_idx]
    us = state.teams[team_idx]
    n_us = us.pos.shape[0]

    # Teammates: same team
    teammate_deltas = pairwise_deltas(us.pos, us.pos, env_config.arena_size)
    teammate_rel_vel = us.vel[None, :, :] - us.vel[:, None, :]
    self_mask = ~jnp.eye(n_us, dtype=jnp.bool_)
    teammate_alive = us.alive[None, :] & self_mask

    # Opponents: all other teams concatenated
    other_indices = [j for j in range(len(rules.teams)) if j != team_idx]
    others_pos = jnp.concatenate([state.teams[j].pos for j in other_indices], axis=0)
    others_vel = jnp.concatenate([state.teams[j].vel for j in other_indices], axis=0)
    others_alive = jnp.concatenate([state.teams[j].alive for j in other_indices], axis=0)

    opponent_deltas_raw = pairwise_deltas(others_pos, us.pos, env_config.arena_size)  # (n_others, n_us, 2)
    opponent_deltas = jnp.transpose(opponent_deltas_raw, (1, 0, 2))  # (n_us, n_others, 2)
    opponent_rel_vel = others_vel[None, :, :] - us.vel[:, None, :]  # (n_us, n_others, 2)
    n_others = others_pos.shape[0]
    opponent_alive = jnp.broadcast_to(others_alive[None, :], (n_us, n_others))

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
        tc.k_teammates,
        tc.k_opponents,
    )

    return Observations(own_vel=own_vel, teammates=teammates, opponents=opponents)
