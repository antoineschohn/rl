import jax
import jax.numpy as jnp

from flock.env import EnvConfig, RandomPolicy, run_episode
from flock.env.rules import PredatorPrey
from flock.env.obs import observe
from flock.env.core import reset
from flock.train.policy import make_policy, obs_dim


def _setup():
    env_cfg = EnvConfig(max_steps=10)
    inter = PredatorPrey(n_predators=3, n_prey=5, k_teammates=2, k_opponents=3)
    return env_cfg, inter


def test_obs_dim():
    assert obs_dim(k_teammates=2, k_opponents=3) == 2 + 2 * 4 + 3 * 4


def test_policy_shapes():
    cfg, inter = _setup()
    tc = inter.teams[0]
    policy = make_policy(tc.k_teammates, tc.k_opponents, key=jax.random.key(0))
    state = reset(cfg, inter, jax.random.key(1))
    obs = observe(inter, cfg, state, 0)

    actions, ps = policy(obs, jax.random.key(2), policy.init_state())
    assert actions.shape == (3, 2)
    assert ps is None


def test_policy_evaluate():
    cfg, inter = _setup()
    tc = inter.teams[0]
    policy = make_policy(tc.k_teammates, tc.k_opponents, key=jax.random.key(0))
    state = reset(cfg, inter, jax.random.key(1))
    obs = observe(inter, cfg, state, 0)

    mean, log_std, value = policy.evaluate(obs)
    assert mean.shape == (3, 2)
    assert log_std.shape == (3, 2)
    assert value.shape == (3,)


def test_policy_in_episode():
    cfg, inter = _setup()
    pred_policy = make_policy(inter.teams[0].k_teammates, inter.teams[0].k_opponents, key=jax.random.key(0))
    prey_policy = RandomPolicy(inter.teams[1])
    policies = (pred_policy, prey_policy)
    sim = run_episode(cfg, inter, jax.random.key(1), policies)
    assert sim.states.teams[0].pos.shape == (cfg.max_steps + 1, inter.teams[0].n_agents, 2)
