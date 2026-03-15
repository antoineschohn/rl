import jax
import jax.numpy as jnp

from flock.env import Bush, EnvConfig, RandomPolicy, run_episode, save, load
from flock.env.rules import PredatorPrey


def test_save_load_roundtrip(tmp_path):
    rules = PredatorPrey(n_predators=3, n_prey=5)
    sim = run_episode(
        EnvConfig(max_steps=10, bushes=(Bush(x=2.0, y=3.0, radius=1.0),)),
        rules,
        jax.random.key(0),
        (RandomPolicy(rules.teams[0]), RandomPolicy(rules.teams[1])),
    )

    path = tmp_path / "sim.npz"
    save(sim, path)
    sim2 = load(path)

    assert sim2.env_config == sim.env_config
    assert sim2.rules is None
    assert len(sim2.states.teams) == len(sim.states.teams)
    for orig, loaded in zip(sim.states.teams, sim2.states.teams):
        assert jnp.allclose(orig.pos, loaded.pos)
        assert jnp.allclose(orig.vel, loaded.vel)
        assert (orig.alive == loaded.alive).all()
    assert (sim.states.step_id == sim2.states.step_id).all()
    assert jnp.allclose(sim.infos.scores[0], sim2.infos.scores[0])
    assert (sim.infos.done == sim2.infos.done).all()
