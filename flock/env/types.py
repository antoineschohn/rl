"""State and configuration types for the flock environment."""

from typing import NamedTuple

import equinox as eqx
import jax
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


class Observation(NamedTuple):
    """What one agent sees."""
    own_vel: jnp.ndarray        # (2,)
    teammates: jnp.ndarray      # (k_teammates, 4) — relative (dx, dy, dvx, dvy)
    opponents: jnp.ndarray      # (k_opponents, 4) — relative (dx, dy, dvx, dvy)


class Observations(NamedTuple):
    """What a whole team sees (batched)."""
    own_vel: jnp.ndarray        # (n_agents, 2)
    teammates: jnp.ndarray      # (n_agents, k_teammates, 4)
    opponents: jnp.ndarray      # (n_agents, k_opponents, 4)


RngKey = jax.Array  # alias for readability — JAX PRNG state, create with jax.random.key(seed)
PolicyState = jax.Array | None  # arbitrary pytree carried across steps, None for stateless


class Policy(eqx.Module):
    """Base class for policies. Subclass and implement __call__ and init_state."""

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        """Return (actions, new_state). Actions shape: (n_agents, 2)."""
        raise NotImplementedError

    def init_state(self) -> PolicyState:
        """Return initial policy state. None for stateless policies."""
        return None

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
    k_teammates: int = 5
    k_opponents: int = 5


class EnvState(NamedTuple):
    """Full environment state at one timestep."""
    predators: Agents
    prey: Agents
    step_id: jnp.ndarray  # scalar int


class EnvStates(NamedTuple):
    """Time-stacked environment states (one episode)."""
    predators: Agents   # pos: (T, n_pred, 2), etc.
    prey: Agents         # pos: (T, n_prey, 2), etc.
    step_id: jnp.ndarray  # (T,)


class Simulation(NamedTuple):
    """A complete simulation: config + recorded states."""
    config: EnvConfig
    states: EnvStates
