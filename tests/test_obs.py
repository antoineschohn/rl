import jax
import jax.numpy as jnp
from flock.env.types import EnvConfig
from flock.env.core import reset
from flock.env.obs import observe


def _small_config(**overrides):
    defaults = dict(n_predators=3, n_prey=5, arena_size=10.0, max_steps=100,
                    k_teammates=2, k_opponents=3)
    defaults.update(overrides)
    return EnvConfig(**defaults)


def test_observe_predators_shape():
    cfg = _small_config()
    state = reset(cfg, jax.random.key(0))
    obs = observe(cfg, state, "predators")
    assert obs.own_vel.shape == (3, 2)
    assert obs.teammates.shape == (3, 2, 4)   # k_teammates=2
    assert obs.opponents.shape == (3, 3, 4)   # k_opponents=3


def test_observe_prey_shape():
    cfg = _small_config()
    state = reset(cfg, jax.random.key(0))
    obs = observe(cfg, state, "prey")
    assert obs.own_vel.shape == (5, 2)
    assert obs.teammates.shape == (5, 2, 4)
    assert obs.opponents.shape == (5, 3, 4)


def test_observe_excludes_self():
    """With 2 predators and k_teammates=1, each should see the other, not itself."""
    cfg = _small_config(n_predators=2, k_teammates=1)
    state = reset(cfg, jax.random.key(0))
    obs = observe(cfg, state, "predators")
    # Teammate deltas should be nonzero (not self)
    for i in range(2):
        assert not jnp.allclose(obs.teammates[i, 0, :2], jnp.zeros(2))


def test_observe_dead_agents_zeroed():
    """Dead opponents should not appear in observations."""
    cfg = _small_config(n_prey=2, k_opponents=2)
    state = reset(cfg, jax.random.key(0))
    # Kill all prey
    dead_prey = state.prey._replace(alive=jnp.zeros(2, dtype=jnp.bool_))
    state = state._replace(prey=dead_prey)
    obs = observe(cfg, state, "predators")
    # All opponent slots should be zero
    assert jnp.allclose(obs.opponents, 0.0)
