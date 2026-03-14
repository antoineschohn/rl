import jax
import jax.numpy as jnp


from flock.env.types import Agent, Agents, EnvConfig, EnvState, EnvStates, Observations, Policy, PolicyState, RngKey, Simulation, StepInfo
from flock.env.physics import integrate, wrap_position, clamp_magnitude
from flock.env.reward import compute_catches, predator_reward, prey_reward
from flock.env.obs import observe


def reset(config: EnvConfig, key: RngKey) -> EnvState:
    """Initialize a fresh environment with random positions and zero velocity. (key is jax' rng-state)"""
    k1, k2 = jax.random.split(key)
    pred_pos = jax.random.uniform(k1, (config.n_predators, 2)) * config.arena_size
    prey_pos = jax.random.uniform(k2, (config.n_prey, 2)) * config.arena_size
    return EnvState(
        predators=Agents(
            pos=pred_pos,
            vel=jnp.zeros((config.n_predators, 2)),
            alive=jnp.ones(config.n_predators, dtype=jnp.bool_),
        ),
        prey=Agents(
            pos=prey_pos,
            vel=jnp.zeros((config.n_prey, 2)),
            alive=jnp.ones(config.n_prey, dtype=jnp.bool_),
        ),
        step_id=jnp.int32(0),
    )



def step(config: EnvConfig, state: EnvState, pred_actions: jax.Array, prey_actions: jax.Array) -> tuple[EnvState, StepInfo]:
    """One environment step. Pure function: (config, state, actions) -> (state, info)."""

    pred_accel = clamp_magnitude(pred_actions, config.max_accel)
    prey_accel = clamp_magnitude(prey_actions, config.max_accel)

    # Dead prey don't move
    prey_accel = prey_accel * state.prey.alive[:, None]

    # Integrate physics
    pred_pos, pred_vel = integrate(
        state.predators.pos, state.predators.vel, pred_accel,
        config.dt, config.max_speed_predator,
    )
    prey_pos, prey_vel = integrate(
        state.prey.pos, state.prey.vel, prey_accel,
        config.dt, config.max_speed_prey,
    )

    # Wrap positions (toroidal)
    pred_pos = wrap_position(pred_pos, config.arena_size)
    prey_pos = wrap_position(prey_pos, config.arena_size)

    # Compute catches
    preds = Agents(pos=pred_pos, vel=pred_vel, alive=state.predators.alive)
    preys = Agents(pos=prey_pos, vel=prey_vel, alive=state.prey.alive)
    new_prey_alive, catch_mask = compute_catches(
        preds, preys, config.catch_radius, config.arena_size,
    )

    # Rewards
    pred_r = predator_reward(catch_mask)
    prey_r = prey_reward(state.prey.alive)

    # Build new state
    new_step = state.step_id + 1
    new_state = EnvState(
        predators=Agents(pos=pred_pos, vel=pred_vel, alive=state.predators.alive),
        prey=Agents(pos=prey_pos, vel=prey_vel, alive=new_prey_alive),
        step_id=new_step,
    )

    done = (new_step >= config.max_steps) | (~new_prey_alive.any())

    return new_state, StepInfo(pred_reward=pred_r, prey_reward=prey_r, done=done)


class RandomPolicy(Policy):
    """Stateless policy that applies random accelerations."""
    n_agents: int

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        return jax.random.normal(key, (self.n_agents, 2)) * 2.0, state


def run_episodes(
    config: EnvConfig,
    key: RngKey,
    pred_policy: Policy,
    prey_policy: Policy,
    n_arenas: int,
) -> Simulation:
    """Run n_arenas episodes in parallel via vmap. Same config and policies, different seeds.

    Returns a Simulation with batch dimension prepended:
    e.g. states.predators.pos has shape (n_arenas, T, n_pred, 2).

    Once done, state freezes — the scan runs for max_steps unconditionally
    but stops mutating state after the episode ends.
    """
    pred_ps_init = pred_policy.init_state()
    prey_ps_init = prey_policy.init_state()

    def single_episode(key):
        key, reset_key = jax.random.split(key)
        init_state = reset(config, reset_key)

        def scan_fn(carry, _):
            state, done, key, ps_pred, ps_prey = carry
            key, k1, k2 = jax.random.split(key, 3)

            pred_obs = observe(config, state, "predators")
            prey_obs = observe(config, state, "prey")

            pred_a, ps_pred = pred_policy(pred_obs, k1, ps_pred)
            prey_a, ps_prey = prey_policy(prey_obs, k2, ps_prey)

            new_state, info = step(config, state, pred_a, prey_a)

            # Freeze state once done
            done = done | info.done
            state = jax.tree.map(
                lambda old, new: jnp.where(done, old, new),
                state, new_state,
            )

            return (state, done, key, ps_pred, ps_prey), (state, info)

        init_carry = (init_state, jnp.bool_(False), key, pred_ps_init, prey_ps_init)
        _, (states, infos) = jax.lax.scan(scan_fn, init_carry, None, length=config.max_steps)

        # Prepend initial state (no info for t=0)
        all_states = jax.tree.map(
            lambda init, scanned: jnp.concatenate([init[None], scanned], axis=0),
            init_state, states,
        )

        return Simulation(
            config=config,
            states=EnvStates(
                predators=Agents(
                    pos=all_states.predators.pos,
                    vel=all_states.predators.vel,
                    alive=all_states.predators.alive,
                ),
                prey=Agents(
                    pos=all_states.prey.pos,
                    vel=all_states.prey.vel,
                    alive=all_states.prey.alive,
                ),
                step_id=all_states.step_id,
            ),
            infos=infos,  # StepInfo stacked over time: (T, ...)
        )

    keys = jax.random.split(key, n_arenas)
    return jax.vmap(single_episode)(keys)


def run_episode(
    config: EnvConfig,
    key: RngKey,
    pred_policy: Policy,
    prey_policy: Policy,
) -> Simulation:
    """Run a single episode. Convenience wrapper around run_episodes."""
    sim = run_episodes(config, key, pred_policy, prey_policy, n_arenas=1)
    return jax.tree.map(lambda x: x[0], sim)
