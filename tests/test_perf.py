"""Performance benchmarks. Run with: uv run pytest tests/test_perf.py --benchmark-only"""

import jax

from flock.env import EnvConfig, run_episode, random_policy


def _make_runner(config: EnvConfig):
    """Return a callable that runs one episode (pre-warmed)."""
    pred_pol = random_policy(config.n_predators)
    prey_pol = random_policy(config.n_prey)
    # Warmup JIT
    _ = run_episode(config, jax.random.key(0), pred_pol, prey_pol)

    def run():
        run_episode(config, jax.random.key(1), pred_pol, prey_pol)

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
