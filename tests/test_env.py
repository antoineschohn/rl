import jax
import jax.numpy as jnp
from flock.env.types import EnvConfig
from flock.env.core import reset, step


def _small_config(**overrides):
    defaults = dict(n_predators=3, n_prey=5, arena_size=10.0, max_steps=100)
    defaults.update(overrides)
    return EnvConfig(**defaults)


def test_reset_shapes():
    cfg = _small_config()
    state = reset(cfg, jax.random.key(0))
    assert state.predators.pos.shape == (3, 2)
    assert state.prey.pos.shape == (5, 2)
    assert state.predators.alive.all()
    assert state.prey.alive.all()
    assert state.step_id == 0


def test_reset_positions_in_arena():
    cfg = _small_config()
    state = reset(cfg, jax.random.key(42))
    assert (state.predators.pos >= 0).all() and (state.predators.pos <= cfg.arena_size).all()
    assert (state.prey.pos >= 0).all() and (state.prey.pos <= cfg.arena_size).all()


def test_step_zero_actions():
    cfg = _small_config()
    state = reset(cfg, jax.random.key(0))
    pred_a = jnp.zeros((cfg.n_predators, 2))
    prey_a = jnp.zeros((cfg.n_prey, 2))
    new_state, info = step(cfg, state, pred_a, prey_a)
    # Zero velocity + zero accel => positions unchanged
    assert jnp.allclose(new_state.predators.pos, state.predators.pos)
    assert new_state.step_id == 1


def test_predator_catches_prey():
    cfg = _small_config(catch_radius=1.0)
    state = reset(cfg, jax.random.key(0))
    # Place first predator on top of first prey
    pred_pos = state.predators.pos.at[0].set(jnp.array([5.0, 5.0]))
    prey_pos = state.prey.pos.at[0].set(jnp.array([5.0, 5.0]))
    state = state._replace(
        predators=state.predators._replace(pos=pred_pos),
        prey=state.prey._replace(pos=prey_pos),
    )
    pred_a = jnp.zeros((cfg.n_predators, 2))
    prey_a = jnp.zeros((cfg.n_prey, 2))
    new_state, info = step(cfg, state, pred_a, prey_a)
    # First prey should be dead
    assert not new_state.prey.alive[0]
    assert info.pred_reward[0] > 0


def test_done_at_max_steps():
    cfg = _small_config(max_steps=2)
    state = reset(cfg, jax.random.key(0))
    pred_a = jnp.zeros((cfg.n_predators, 2))
    prey_a = jnp.zeros((cfg.n_prey, 2))
    state, info = step(cfg, state, pred_a, prey_a)
    assert not info.done
    state, info = step(cfg, state, pred_a, prey_a)
    assert info.done
