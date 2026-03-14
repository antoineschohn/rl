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


class TeamConfig(NamedTuple):
    """Configuration for one team."""
    name: str
    n_agents: int
    max_speed: float
    max_accel: float = 10.0
    k_teammates: int = 5
    k_opponents: int = 5


class EnvConfig(NamedTuple):
    """Arena and physics parameters."""
    arena_size: float = 10.0
    dt: float = 0.05
    max_steps: int = 500


class EnvState(NamedTuple):
    """Full environment state at one timestep."""
    teams: tuple[Agents, ...]  # one per team
    step_id: jnp.ndarray       # scalar int


class EnvStates(NamedTuple):
    """Time-stacked environment states (one episode)."""
    teams: tuple[Agents, ...]  # each has pos: (T, n_i, 2), etc.
    step_id: jnp.ndarray       # (T,)


class StepInfo(NamedTuple):
    """Output of a single environment step."""
    rewards: tuple[jax.Array, ...]  # rewards[i] shape (n_agents_i,)
    done: jnp.ndarray               # scalar bool


class Simulation(NamedTuple):
    """A single simulation run."""
    env_config: EnvConfig
    rules: object            # the Rules module used
    states: EnvStates        # (T+1, ...) — includes initial state at t=0
    infos: StepInfo          # (T, ...) — infos[t] = step(states[t]).info
    extras: object = None    # arbitrary pytree from step_hook, stacked over time (T, ...)


class Simulations(NamedTuple):
    """Batched simulation runs. All data arrays have a leading (n_arenas,) dimension."""
    env_config: EnvConfig
    rules: object
    states: EnvStates        # (n_arenas, T+1, ...)
    infos: StepInfo          # (n_arenas, T, ...)
    extras: object = None    # (n_arenas, T, ...)


def unbatch(sims: Simulations) -> list[Simulation]:
    """Split a batched Simulations into a list of Simulation."""
    n = sims.states.step_id.shape[0]
    return [
        Simulation(
            env_config=sims.env_config,
            rules=sims.rules,
            states=jax.tree.map(lambda x: x[i], sims.states),
            infos=jax.tree.map(lambda x: x[i], sims.infos),
            extras=jax.tree.map(lambda x: x[i], sims.extras) if sims.extras is not None else None,
        )
        for i in range(n)
    ]


def batch(sims: list[Simulation]) -> Simulations:
    """Stack a list of Simulation into a batched Simulations."""
    return Simulations(
        env_config=sims[0].env_config,
        rules=sims[0].rules,
        states=jax.tree.map(lambda *xs: jnp.stack(xs), *[s.states for s in sims]),
        infos=jax.tree.map(lambda *xs: jnp.stack(xs), *[s.infos for s in sims]),
        extras=jax.tree.map(lambda *xs: jnp.stack(xs), *[s.extras for s in sims]) if sims[0].extras is not None else None,
    )
