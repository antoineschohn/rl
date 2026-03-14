"""Performance benchmarks. Run with: uv run pytest tests/test_perf.py --benchmark-only"""

import jax

from flock.env import EnvConfig, RandomPolicy, run_episode, run_episodes


def _make_runner(config: EnvConfig, n_arenas: int = 1):
    """Return a callable that runs episode(s) (pre-warmed)."""
    pred_pol = RandomPolicy(config.n_predators)
    prey_pol = RandomPolicy(config.n_prey)
    if n_arenas == 1:
        _ = run_episode(config, jax.random.key(0), pred_pol, prey_pol)
        def run():
            run_episode(config, jax.random.key(1), pred_pol, prey_pol)
    else:
        _ = run_episodes(config, jax.random.key(0), pred_pol, prey_pol, n_arenas)
        def run():
            run_episodes(config, jax.random.key(1), pred_pol, prey_pol, n_arenas)
    return run


def test_perf_small(benchmark):
    """5 predators, 20 prey, 200 steps."""
    cfg = EnvConfig(n_predators=5, n_prey=20, max_steps=200)
    benchmark(_make_runner(cfg))


def test_perf_medium(benchmark):
    """10 predators, 50 prey, 500 steps."""
    cfg = EnvConfig(n_predators=10, n_prey=50, max_steps=500)
    benchmark(_make_runner(cfg))


def test_perf_large(benchmark):
    """20 predators, 100 prey, 500 steps."""
    cfg = EnvConfig(n_predators=20, n_prey=100, max_steps=500, k_teammates=10, k_opponents=10)
    benchmark(_make_runner(cfg))


def test_perf_batched_64(benchmark):
    """5v20, 200 steps, 64 parallel arenas."""
    cfg = EnvConfig(n_predators=5, n_prey=20, max_steps=200)
    benchmark(_make_runner(cfg, n_arenas=64))


def test_perf_batched_256(benchmark):
    """5v20, 200 steps, 256 parallel arenas."""
    cfg = EnvConfig(n_predators=5, n_prey=20, max_steps=200)
    benchmark(_make_runner(cfg, n_arenas=256))
