"""Observation extraction — the information bottleneck between state and policy."""

import jax
import jax.numpy as jnp

from flock.env.types import Bush, EnvConfig, EnvState, Observations
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
    n_others = deltas.shape[0]
    width = deltas.shape[-1] + rel_vel.shape[-1]
    if k == 0:
        return jnp.zeros((0, width), dtype=deltas.dtype)

    if n_others == 0:
        return jnp.zeros((k, width), dtype=deltas.dtype)

    dists = jnp.linalg.norm(deltas, axis=-1)
    dists = jnp.where(alive, dists, jnp.inf)
    nearest_idx = jnp.argsort(dists)
    take = min(k, n_others)
    nearest_idx = nearest_idx[:take]

    rel_pos = deltas[nearest_idx]
    rel_v = rel_vel[nearest_idx]
    result = jnp.concatenate([rel_pos, rel_v], axis=-1)
    valid = alive[nearest_idx][:, None]
    result = result * valid

    pad = k - take
    if pad > 0:
        result = jnp.pad(result, ((0, pad), (0, 0)))
    return result


def _observe_one_agent(
    teammate_deltas: jnp.ndarray,
    teammate_rel_vel: jnp.ndarray,
    teammate_alive: jnp.ndarray,
    opponent_deltas: jnp.ndarray,
    opponent_rel_vel: jnp.ndarray,
    opponent_alive: jnp.ndarray,
    k_teammates: int,
    k_opponents: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build observation for a single agent. Designed to be vmapped."""
    teammates = _k_nearest(teammate_deltas, teammate_rel_vel, teammate_alive, k_teammates)
    opponents = _k_nearest(opponent_deltas, opponent_rel_vel, opponent_alive, k_opponents)
    return teammates, opponents


def _bush_arrays(env_config: EnvConfig, dtype: jnp.dtype) -> tuple[jnp.ndarray, jnp.ndarray]:
    bushes: tuple[Bush, ...] = env_config.bushes
    if not bushes:
        return jnp.zeros((0, 2), dtype=dtype), jnp.zeros((0,), dtype=dtype)

    centers = jnp.asarray([(b.x, b.y) for b in bushes], dtype=dtype)
    radii = jnp.asarray([b.radius for b in bushes], dtype=dtype)
    return centers, radii


def _agents_in_bushes(pos: jnp.ndarray, env_config: EnvConfig) -> jnp.ndarray:
    """Return a boolean membership matrix of shape (n_agents, n_bushes)."""
    centers, radii = _bush_arrays(env_config, pos.dtype)
    if centers.shape[0] == 0:
        return jnp.zeros((pos.shape[0], 0), dtype=jnp.bool_)

    deltas = pairwise_deltas(pos, centers, env_config.arena_size)
    distances = jnp.linalg.norm(deltas, axis=-1)
    return distances <= radii[None, :]


def _visibility_mask(observer_bushes: jnp.ndarray, target_bushes: jnp.ndarray) -> jnp.ndarray:
    """Visibility between observers and targets.

    Targets outside bushes are always visible. Targets in a bush are only visible
    to observers that share at least one bush with them.
    """
    if target_bushes.shape[1] == 0:
        return jnp.ones((observer_bushes.shape[0], target_bushes.shape[0]), dtype=jnp.bool_)

    targets_hidden = jnp.any(target_bushes, axis=-1)
    shared_bush = jnp.any(observer_bushes[:, None, :] & target_bushes[None, :, :], axis=-1)
    return (~targets_hidden)[None, :] | shared_bush


def observe(rules: Rules, env_config: EnvConfig, state: EnvState, team_idx: int) -> Observations:
    """Extract observations for a team. Policies only see this, never EnvState."""
    tc = rules.teams[team_idx]
    us = state.teams[team_idx]
    n_us = us.pos.shape[0]
    us_bushes = _agents_in_bushes(us.pos, env_config)

    # Teammates: same team
    teammate_deltas = pairwise_deltas(us.pos, us.pos, env_config.arena_size)
    teammate_rel_vel = us.vel[None, :, :] - us.vel[:, None, :]
    self_mask = ~jnp.eye(n_us, dtype=jnp.bool_)
    teammate_visible = _visibility_mask(us_bushes, us_bushes)
    teammate_alive = us.alive[None, :] & self_mask & teammate_visible

    # Opponents: all other teams concatenated
    other_indices = [j for j in range(len(rules.teams)) if j != team_idx]
    others_pos = jnp.concatenate([state.teams[j].pos for j in other_indices], axis=0)
    others_vel = jnp.concatenate([state.teams[j].vel for j in other_indices], axis=0)
    others_alive = jnp.concatenate([state.teams[j].alive for j in other_indices], axis=0)
    others_bushes = jnp.concatenate([
        _agents_in_bushes(state.teams[j].pos, env_config) for j in other_indices
    ], axis=0)

    opponent_deltas_raw = pairwise_deltas(others_pos, us.pos, env_config.arena_size)  # (n_others, n_us, 2)
    opponent_deltas = jnp.transpose(opponent_deltas_raw, (1, 0, 2))  # (n_us, n_others, 2)
    opponent_rel_vel = others_vel[None, :, :] - us.vel[:, None, :]  # (n_us, n_others, 2)
    n_others = others_pos.shape[0]
    opponent_visible = _visibility_mask(us_bushes, others_bushes)
    opponent_alive = jnp.broadcast_to(others_alive[None, :], (n_us, n_others)) & opponent_visible

    teammates, opponents = jax.vmap(
        _observe_one_agent,
        in_axes=(0, 0, 0, 0, 0, 0, None, None),
    )(
        teammate_deltas,
        teammate_rel_vel,
        teammate_alive,
        opponent_deltas,
        opponent_rel_vel,
        opponent_alive,
        tc.k_teammates,
        tc.k_opponents,
    )

    in_bush = jnp.any(us_bushes, axis=-1, keepdims=True).astype(us.pos.dtype)
    return Observations(
        own_pos=us.pos,
        own_vel=us.vel,
        in_bush=in_bush,
        teammates=teammates,
        opponents=opponents,
    )
