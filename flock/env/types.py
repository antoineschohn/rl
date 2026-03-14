"""State and configuration types for the flock environment."""

from typing import NamedTuple

import jax.numpy as jnp


class Agent(NamedTuple):
    """
    Single agent state. 

    By default, you should use `Agents` in performance critical paths instead. """
    pos: jnp.ndarray   # (2,)
    vel: jnp.ndarray   # (2,)
    alive: jnp.ndarray  # scalar bool


class Agents(NamedTuple):
    """Batched agent state."""
    pos: jnp.ndarray   # (n_agents, 2)
    vel: jnp.ndarray   # (n_agents, 2)
    alive: jnp.ndarray  # (n_agents,) bool


class EnvState(NamedTuple):
    """Full environment state at one timestep."""
    predators: Agents
    prey: Agents
    step_id: jnp.ndarray  # scalar int


class EnvConfig(NamedTuple):
    """Environment parameters. Immutable across an episode."""
    arena_size: float = 10.0
    dt: float = 0.05
    max_speed_predator: float = 3.0
    max_speed_prey: float = 4.0 
    max_accel: float = 10.0
    catch_radius: float = 0.3
    n_predators: int = 5
    n_prey: int = 20
    max_steps: int = 500
