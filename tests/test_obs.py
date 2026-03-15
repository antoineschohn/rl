import jax
import jax.numpy as jnp
from flock.env.types import Bush, EnvConfig
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


def test_observe_hides_opponents_inside_bush_from_outside_agents():
    cfg, inter = _setup(n_predators=1, n_prey=1, k_opponents=1)
    cfg = cfg._replace(bushes=(Bush(x=5.0, y=5.0, radius=1.0),))
    state = reset(cfg, inter, jax.random.key(0))
    state = state._replace(teams=(
        state.teams[0]._replace(pos=jnp.array([[1.0, 1.0]])),
        state.teams[1]._replace(pos=jnp.array([[5.0, 5.0]])),
    ))

    obs = observe(inter, cfg, state, 0)
    assert jnp.allclose(obs.opponents, 0.0)


def test_observe_shows_agents_sharing_same_bush():
    cfg, inter = _setup(n_predators=1, n_prey=1, k_opponents=1)
    cfg = cfg._replace(bushes=(Bush(x=5.0, y=5.0, radius=1.5),))
    state = reset(cfg, inter, jax.random.key(0))
    state = state._replace(teams=(
        state.teams[0]._replace(pos=jnp.array([[5.0, 5.5]])),
        state.teams[1]._replace(pos=jnp.array([[5.5, 5.0]])),
    ))

    obs = observe(inter, cfg, state, 0)
    assert not jnp.allclose(obs.opponents, 0.0)


def test_observe_hides_teammates_inside_bush_from_outside_agents():
    cfg, inter = _setup(n_predators=2, n_prey=1, k_teammates=1)
    cfg = cfg._replace(bushes=(Bush(x=5.0, y=5.0, radius=1.0),))
    state = reset(cfg, inter, jax.random.key(0))
    predators = state.teams[0]._replace(pos=jnp.array([[1.0, 1.0], [5.0, 5.0]]))
    state = state._replace(teams=(predators, state.teams[1]))

    obs = observe(inter, cfg, state, 0)
    assert jnp.allclose(obs.teammates[0], 0.0)
    assert not jnp.allclose(obs.teammates[1], 0.0)
