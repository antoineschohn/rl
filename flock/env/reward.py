import jax
import jax.numpy as jnp

from flock.env.types import Agents
from flock.env.physics import pairwise_distances


def compute_catches(
    predators: Agents,
    prey: Agents,
    catch_radius: float,
    arena_size: float,
) -> tuple[jax.Array, jax.Array]:
    """Determine which prey are caught this step.

    Returns:
        new_prey_alive: (n_prey,) bool — updated alive mask
        catch_mask: (n_pred, n_prey) bool — which predator caught which prey
    """
    dists = pairwise_distances(predators.pos, prey.pos, arena_size)  # (n_pred, n_prey)
    in_range = dists < catch_radius  # (n_pred, n_prey)
    catch_mask = in_range & prey.alive[None, :]
    caught = jnp.any(catch_mask, axis=0)  # (n_prey,)
    new_prey_alive = prey.alive & ~caught
    return new_prey_alive, catch_mask


def predator_reward(catch_mask: jax.Array) -> jax.Array:
    """Reward predators for successful catches."""
    # +1 for each predator within catch_radius of a caught prey
    # note: potentiellement trop généreux ?  
    return catch_mask.sum(axis=1).astype(jnp.float32)  # (n_pred,)


def prey_reward(prey_alive: jax.Array) -> jax.Array:
    """Reward prey for survival."""
    # +1 per step alive, 0 when dead
    return prey_alive.astype(jnp.float32)  # (n_prey,)
