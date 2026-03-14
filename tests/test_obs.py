import jax
import jax.numpy as jnp
from flock.env.types import EnvConfig
from flock.env.rules import PredatorPrey
from flock.env.core import reset
from flock.env.obs import observe


def _setup(**overrides):
    pp_defaults = dict(n_predators=3, n_prey=5, k_teammates=2, k_opponents=3)
    pp_defaults.update(overrides)
    return EnvConfig(max_steps=100), PredatorPrey(**pp_defaults)


def test_observe_predators_shape():
    cfg, inter = _setup()
    state = reset(cfg, inter, jax.random.key(0))
    obs = observe(inter, cfg, state, 0)
    assert obs.own_vel.shape == (3, 2)
    assert obs.teammates.shape == (3, 2, 4)
    assert obs.opponents.shape == (3, 3, 4)


def test_observe_prey_shape():
    cfg, inter = _setup()
    state = reset(cfg, inter, jax.random.key(0))
    obs = observe(inter, cfg, state, 1)
    assert obs.own_vel.shape == (5, 2)
    assert obs.teammates.shape == (5, 2, 4)
    assert obs.opponents.shape == (5, 3, 4)


def test_observe_excludes_self():
    cfg, inter = _setup(n_predators=2, k_teammates=1)
    state = reset(cfg, inter, jax.random.key(0))
    obs = observe(inter, cfg, state, 0)
    for i in range(2):
        assert not jnp.allclose(obs.teammates[i, 0, :2], jnp.zeros(2))


def test_observe_dead_agents_zeroed():
    cfg, inter = _setup(n_prey=2, k_opponents=2)
    state = reset(cfg, inter, jax.random.key(0))
    dead_prey = state.teams[1]._replace(alive=jnp.zeros(2, dtype=jnp.bool_))
    state = state._replace(teams=(state.teams[0], dead_prey))
    obs = observe(inter, cfg, state, 0)
    assert jnp.allclose(obs.opponents, 0.0)
