"""Game rules: teams, catches, rewards, done condition."""

import equinox as eqx
import jax
import jax.numpy as jnp

from flock.env.types import Agents, EnvConfig, TeamConfig
from flock.env.physics import pairwise_distances


class Rules(eqx.Module):
    """Base class for game rules. Defines teams and game mechanics."""
    teams: tuple[TeamConfig, ...]

    def interact(self, env_config: EnvConfig, teams: tuple[Agents, ...]) -> tuple[tuple[Agents, ...], tuple[jax.Array, ...]]:
        """Apply rules after physics. Returns (updated_teams, rewards_per_team)."""
        raise NotImplementedError

    def is_done(self, teams: tuple[Agents, ...]) -> jax.Array:
        """Check if the episode should end (beyond max_steps)."""
        raise NotImplementedError


class PredatorPrey(Rules):
    """Standard predator-prey: team 0 (predators) catches team 1 (prey)."""
    catch_radius: float = 0.3

    def __init__(
        self,
        n_predators: int = 5,
        n_prey: int = 20,
        max_speed_predator: float = 1.0,
        max_speed_prey: float = 1.5,
        max_accel_predator: float = 8.0,
        max_accel_prey: float = 12.0,
        catch_radius: float = 0.3,
        k_teammates: int = 5,
        k_opponents: int = 5,
    ):
        self.teams = (
            TeamConfig("predators", n_predators, max_speed_predator, max_accel_predator, k_teammates, k_opponents),
            TeamConfig("prey", n_prey, max_speed_prey, max_accel_prey, k_teammates, k_opponents),
        )
        self.catch_radius = catch_radius

    def interact(self, env_config: EnvConfig, teams: tuple[Agents, ...]) -> tuple[tuple[Agents, ...], tuple[jax.Array, ...]]:
        predators, prey = teams

        dists = pairwise_distances(predators.pos, prey.pos, env_config.arena_size)
        in_range = dists < self.catch_radius
        catch_mask = in_range & prey.alive[None, :]
        caught = jnp.any(catch_mask, axis=0)
        new_prey_alive = prey.alive & ~caught

        # +1 for each predator within catch_radius of a caught prey (not split)
        pred_reward = catch_mask.sum(axis=1).astype(jnp.float32)
        # +1 per step alive, 0 when dead
        prey_reward = prey.alive.astype(jnp.float32)

        new_prey = prey._replace(alive=new_prey_alive)
        return (predators, new_prey), (pred_reward, prey_reward)

    def is_done(self, teams: tuple[Agents, ...]) -> jax.Array:
        """Done when all prey are dead."""
        prey = teams[1]
        return ~prey.alive.any()
