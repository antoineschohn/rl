"""Performance benchmarks. Run with: uv run pytest tests/test_perf.py --benchmark-only"""

import jax

from flock.env import EnvConfig, RandomPolicy, run_episode, run_episodes
from flock.env.rules import PredatorPrey
from flock.train.ppo import PPOConfig, Trainee, PredatorReward, make_policy, train
from flock.train.naive import FleePredators


def _make_runner(env_config: EnvConfig, interaction: PredatorPrey, n_arenas: int = 1):
    """Return a callable that runs episode(s) (pre-warmed)."""
    policies = tuple(RandomPolicy(tc) for tc in interaction.teams)
    if n_arenas == 1:
        _ = run_episode(env_config, interaction, jax.random.key(0), policies)
        def run():
            run_episode(env_config, interaction, jax.random.key(1), policies)
    else:
        _ = run_episodes(env_config, interaction, jax.random.key(0), policies, n_arenas)
        def run():
            run_episodes(env_config, interaction, jax.random.key(1), policies, n_arenas)
    return run


def test_perf_small(benchmark):
    """5 predators, 20 prey, 200 steps."""
    benchmark(_make_runner(EnvConfig(max_steps=200), PredatorPrey()))


def test_perf_medium(benchmark):
    """10 predators, 50 prey, 500 steps."""
    benchmark(_make_runner(
        EnvConfig(max_steps=500),
        PredatorPrey(n_predators=10, n_prey=50),
    ))


def test_perf_large(benchmark):
    """20 predators, 100 prey, 500 steps."""
    benchmark(_make_runner(
        EnvConfig(max_steps=500),
        PredatorPrey(n_predators=20, n_prey=100, k_teammates=10, k_opponents=10),
    ))


def test_perf_batched_64(benchmark):
    """5v20, 200 steps, 64 parallel arenas."""
    benchmark(_make_runner(EnvConfig(max_steps=200), PredatorPrey(), n_arenas=64))


def test_perf_batched_256(benchmark):
    """5v20, 200 steps, 256 parallel arenas."""
    benchmark(_make_runner(EnvConfig(max_steps=200), PredatorPrey(), n_arenas=256))


def _make_ppo_runner(n_iters=2, n_arenas=32):
    """Return a callable that runs a short PPO training loop (pre-warmed)."""
    rules = PredatorPrey()
    env_cfg = EnvConfig(max_steps=100)
    cfg = PPOConfig(n_iters=n_iters, n_arenas=n_arenas, n_epochs=1)

    key = jax.random.key(0)
    policy = make_policy(rules.teams[0].k_teammates, rules.teams[0].k_opponents, key=key)
    prey_policy = FleePredators(rules.teams[1])
    trainees = (Trainee(team_idx=0, reward_fn=PredatorReward()),)

    # Warmup: 1 iteration to JIT-compile everything
    _ = train(env_cfg, rules, (policy, prey_policy), trainees, jax.random.key(99),
              cfg=cfg._replace(n_iters=1))

    def run():
        train(env_cfg, rules, (policy, prey_policy), trainees, jax.random.key(1), cfg=cfg)
    return run


def test_perf_ppo_iter(benchmark):
    """PPO: 2 iters, 32 arenas, 100 steps — measures per-iteration cost."""
    benchmark(_make_ppo_runner())
