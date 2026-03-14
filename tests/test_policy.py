import jax
import jax.numpy as jnp

from flock.env import EnvConfig, run_episode, RandomPolicy
from flock.env.obs import observe
from flock.env.core import reset
from flock.train.policy import MLPPolicy, make_policy, obs_dim


def _cfg(**overrides):
    defaults = dict(n_predators=3, n_prey=5, k_teammates=2, k_opponents=3, max_steps=10)
    defaults.update(overrides)
    return EnvConfig(**defaults)


def test_obs_dim():
    assert obs_dim(k_teammates=2, k_opponents=3) == 2 + 2 * 4 + 3 * 4  # 22


def test_policy_shapes():
    cfg = _cfg()
    policy = make_policy(cfg.k_teammates, cfg.k_opponents, key=jax.random.key(0))
    state = reset(cfg, jax.random.key(1))
    obs = observe(cfg, state, "predators")

    actions, ps = policy(obs, jax.random.key(2), policy.init_state())
    assert actions.shape == (3, 2)
    assert ps is None


def test_policy_evaluate():
    cfg = _cfg()
    policy = make_policy(cfg.k_teammates, cfg.k_opponents, key=jax.random.key(0))
    state = reset(cfg, jax.random.key(1))
    obs = observe(cfg, state, "predators")

    mean, log_std, value = policy.evaluate(obs)
    assert mean.shape == (3, 2)
    assert log_std.shape == (3, 2)  # broadcast from (2,) by vmap
    assert value.shape == (3,)


def test_policy_in_episode():
    """MLPPolicy plugs into run_episode without errors."""
    cfg = _cfg()
    pred_policy = make_policy(cfg.k_teammates, cfg.k_opponents, key=jax.random.key(0))
    prey_policy = RandomPolicy(cfg.n_prey)
    sim = run_episode(cfg, jax.random.key(1), pred_policy, prey_policy)
    assert sim.states.predators.pos.shape == (cfg.max_steps + 1, cfg.n_predators, 2)
