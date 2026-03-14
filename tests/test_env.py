import jax
import jax.numpy as jnp
from flock.env.types import EnvConfig
from flock.env.rules import PredatorPrey
from flock.env.core import reset, step


def _setup(**overrides):
    pp_defaults = dict(n_predators=3, n_prey=5)
    pp_defaults.update({k: v for k, v in overrides.items() if k in ("n_predators", "n_prey", "catch_radius")})
    env_defaults = dict(max_steps=100)
    env_defaults.update({k: v for k, v in overrides.items() if k in ("arena_size", "max_steps")})
    return EnvConfig(**env_defaults), PredatorPrey(**pp_defaults)


def test_reset_shapes():
    cfg, inter = _setup()
    state = reset(cfg, inter, jax.random.key(0))
    assert state.teams[0].pos.shape == (3, 2)
    assert state.teams[1].pos.shape == (5, 2)
    assert state.teams[0].alive.all()
    assert state.teams[1].alive.all()
    assert state.step_id == 0


def test_reset_positions_in_arena():
    cfg, inter = _setup()
    state = reset(cfg, inter, jax.random.key(42))
    for team in state.teams:
        assert (team.pos >= 0).all() and (team.pos <= cfg.arena_size).all()


def test_step_zero_actions():
    cfg, inter = _setup()
    state = reset(cfg, inter, jax.random.key(0))
    actions = tuple(jnp.zeros((tc.n_agents, 2)) for tc in inter.teams)
    new_state, info = step(cfg, inter, state, actions)
    assert jnp.allclose(new_state.teams[0].pos, state.teams[0].pos)
    assert new_state.step_id == 1


def test_predator_catches_prey():
    cfg, inter = _setup(catch_radius=1.0)
    state = reset(cfg, inter, jax.random.key(0))
    # Place first predator on top of first prey
    pred_pos = state.teams[0].pos.at[0].set(jnp.array([5.0, 5.0]))
    prey_pos = state.teams[1].pos.at[0].set(jnp.array([5.0, 5.0]))
    state = state._replace(teams=(
        state.teams[0]._replace(pos=pred_pos),
        state.teams[1]._replace(pos=prey_pos),
    ))
    actions = tuple(jnp.zeros((tc.n_agents, 2)) for tc in inter.teams)
    new_state, info = step(cfg, inter, state, actions)
    assert not new_state.teams[1].alive[0]
    assert info.rewards[0][0] > 0


def test_done_at_max_steps():
    cfg, inter = _setup(max_steps=2)
    state = reset(cfg, inter, jax.random.key(0))
    actions = tuple(jnp.zeros((tc.n_agents, 2)) for tc in inter.teams)
    state, info = step(cfg, inter, state, actions)
    assert not info.done
    state, info = step(cfg, inter, state, actions)
    assert info.done
