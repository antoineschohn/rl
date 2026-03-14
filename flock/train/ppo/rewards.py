"""Reward modules for PPO training.

Each is an eqx.Module callable with signature:
    (env_config, rules, state, new_state, info, team_idx) -> (n_agents,)
"""

import equinox as eqx
import jax.numpy as jnp

from flock.env.physics import pairwise_distances


class EnvReward(eqx.Module):
    """Passthrough info.scores[team_idx]."""

    def __call__(self, env_config, rules, state, new_state, info, team_idx):
        return info.scores[team_idx]


class DistanceReward(eqx.Module):
    """Negative nearest-opponent distance, normalized by max_steps."""
    coeff: float = 1.0

    def __call__(self, env_config, rules, state, new_state, info, team_idx):
        opp_idx = 1 - team_idx
        team = new_state.teams[team_idx]
        opp = new_state.teams[opp_idx]
        dists = pairwise_distances(team.pos, opp.pos, env_config.arena_size)
        # Mask dead opponents with inf so they don't attract
        dists = jnp.where(opp.alive[None, :], dists, jnp.inf)
        nearest_dist = jnp.minimum(dists.min(axis=1), env_config.arena_size)
        return -self.coeff * nearest_dist / env_config.max_steps


class PredatorReward(eqx.Module):
    """Catch score + distance shaping. Reproduces the original hardcoded behavior."""
    distance_coeff: float = 1.0

    def __call__(self, env_config, rules, state, new_state, info, team_idx):
        catch = EnvReward()(env_config, rules, state, new_state, info, team_idx)
        distance = DistanceReward(coeff=self.distance_coeff)(
            env_config, rules, state, new_state, info, team_idx,
        )
        return catch + distance
