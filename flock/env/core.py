import jax
import jax.numpy as jnp

from typing import Callable

from flock.env.types import Agents, EnvConfig, EnvState, EnvStates, Observations, Policy, PolicyState, RngKey, Simulation, Simulations, StepInfo, TeamConfig
from flock.env.rules import Rules
from flock.env.physics import integrate, wrap_position, clamp_magnitude
from flock.env.obs import observe

# Hook called each step: (obs_per_team, actions_per_team, state, info) -> pytree
StepHook = Callable | None


def reset(env_config: EnvConfig, rules: Rules, key: RngKey) -> EnvState:
    """Initialize a fresh environment with random positions and zero velocity."""
    keys = jax.random.split(key, len(rules.teams))
    teams = tuple(
        Agents(
            pos=jax.random.uniform(keys[i], (tc.n_agents, 2)) * env_config.arena_size,
            vel=jnp.zeros((tc.n_agents, 2)),
            alive=jnp.ones(tc.n_agents, dtype=jnp.bool_),
        )
        for i, tc in enumerate(rules.teams)
    )
    return EnvState(teams=teams, step_id=jnp.int32(0))


def step(
    env_config: EnvConfig,
    rules: Rules,
    state: EnvState,
    actions: tuple[jax.Array, ...],
) -> tuple[EnvState, StepInfo]:
    """One environment step. Pure function."""
    n_teams = len(rules.teams)

    # Clamp accelerations and mask dead agents
    clamped = tuple(
        clamp_magnitude(actions[i], rules.teams[i].max_accel) * state.teams[i].alive[:, None]
        for i in range(n_teams)
    )

    # Integrate physics per team
    new_teams = []
    for i, tc in enumerate(rules.teams):
        pos, vel = integrate(
            state.teams[i].pos, state.teams[i].vel, clamped[i],
            env_config.dt, tc.max_speed,
        )
        pos = wrap_position(pos, env_config.arena_size)
        new_teams.append(Agents(pos=pos, vel=vel, alive=state.teams[i].alive))
    new_teams = tuple(new_teams)

    # Apply rules (catches, rewards)
    new_teams, rewards = rules.interact(env_config, new_teams)

    # Done condition
    new_step = state.step_id + 1
    done = (new_step >= env_config.max_steps) | rules.is_done(new_teams)

    new_state = EnvState(teams=new_teams, step_id=new_step)
    return new_state, StepInfo(rewards=rewards, done=done)


class RandomPolicy(Policy):
    """Stateless policy that applies random accelerations scaled to max_accel."""
    n_agents: int
    max_accel: float

    def __init__(self, team_config: TeamConfig):
        self.n_agents = team_config.n_agents
        self.max_accel = team_config.max_accel

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        return jax.random.normal(key, (self.n_agents, 2)) * self.max_accel, state


def run_episodes(
    env_config: EnvConfig,
    rules: Rules,
    key: RngKey,
    policies: tuple[Policy, ...],
    n_arenas: int,
    step_hook: StepHook = None,
) -> Simulations:
    """Run n_arenas episodes in parallel via vmap.

    Args:
        policies: one Policy per team, in team order.
        step_hook: optional callback (obs_per_team, actions_per_team, state, info) -> pytree.

    Returns a Simulation with batch dimension prepended.
    """
    n_teams = len(rules.teams)
    ps_inits = tuple(p.init_state() for p in policies)

    def single_episode(key):
        key, reset_key = jax.random.split(key)
        init_state = reset(env_config, rules, reset_key)

        def scan_fn(carry, _):
            state, done, key, ps = carry
            keys = jax.random.split(key, n_teams + 1)
            key = keys[0]

            # Observe and act for each team
            obs_all = tuple(observe(rules, env_config, state, i) for i in range(n_teams))
            actions_and_ps = tuple(
                policies[i](obs_all[i], keys[i + 1], ps[i])
                for i in range(n_teams)
            )
            team_actions = tuple(a for a, _ in actions_and_ps)
            new_ps = tuple(s for _, s in actions_and_ps)

            new_state, info = step(env_config, rules, state, team_actions)

            # Freeze state once done
            done = done | info.done
            state = jax.tree.map(
                lambda old, new: jnp.where(done, old, new),
                state, new_state,
            )

            extras = step_hook(obs_all, team_actions, state, info) if step_hook else None
            return (state, done, key, new_ps), (state, info, extras)

        init_carry = (init_state, jnp.bool_(False), key, ps_inits)
        _, (states, infos, extras) = jax.lax.scan(scan_fn, init_carry, None, length=env_config.max_steps)

        # Prepend initial state
        all_states = jax.tree.map(
            lambda init, scanned: jnp.concatenate([init[None], scanned], axis=0),
            init_state, states,
        )

        return (
            EnvStates(teams=all_states.teams, step_id=all_states.step_id),
            infos,
            extras,
        )

    keys = jax.random.split(key, n_arenas)
    states, infos, extras = jax.vmap(single_episode)(keys)
    return Simulations(
        env_config=env_config,
        rules=rules,
        states=states,
        infos=infos,
        extras=extras,
    )


def run_episode(
    env_config: EnvConfig,
    rules: Rules,
    key: RngKey,
    policies: tuple[Policy, ...],
) -> Simulation:
    """Run a single episode. Convenience wrapper around run_episodes."""
    sims = run_episodes(env_config, rules, key, policies, n_arenas=1)
    squeeze = lambda x: x[0]
    return Simulation(
        env_config=sims.env_config,
        rules=sims.rules,
        states=jax.tree.map(squeeze, sims.states),
        infos=jax.tree.map(squeeze, sims.infos),
        extras=jax.tree.map(squeeze, sims.extras) if sims.extras is not None else None,
    )
