"""Save and load simulations to disk.

Stores states and infos as numpy arrays + env_config as a plain dict.
Does not serialize Rules (reconstruct from your game config instead).
"""

import json
from pathlib import Path

import jax
import numpy as np

from flock.env.types import Agents, EnvConfig, EnvStates, StepInfo


def save(sim, path):
    """Save a Simulation or Simulations to a .npz file."""
    path = Path(path)
    arrays = {}

    # Flatten states
    for i, team in enumerate(sim.states.teams):
        arrays[f"states/teams/{i}/pos"] = np.asarray(team.pos)
        arrays[f"states/teams/{i}/vel"] = np.asarray(team.vel)
        arrays[f"states/teams/{i}/alive"] = np.asarray(team.alive)
    arrays["states/step_id"] = np.asarray(sim.states.step_id)

    # Flatten infos
    for i, r in enumerate(sim.infos.rewards):
        arrays[f"infos/rewards/{i}"] = np.asarray(r)
    arrays["infos/done"] = np.asarray(sim.infos.done)

    # Save env_config as JSON string in a separate sidecar
    meta = {"env_config": sim.env_config._asdict(), "n_teams": len(sim.states.teams)}
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta))

    np.savez_compressed(path, **arrays)


def load(path):
    """Load states, infos, and env_config from disk.

    Returns (env_config, states, infos). Caller reconstructs Rules separately.
    """
    from flock.env.types import Simulation

    path = Path(path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    env_config = EnvConfig(**meta["env_config"])
    n_teams = meta["n_teams"]

    data = np.load(path, allow_pickle=False)

    teams = tuple(
        Agents(
            pos=data[f"states/teams/{i}/pos"],
            vel=data[f"states/teams/{i}/vel"],
            alive=data[f"states/teams/{i}/alive"],
        )
        for i in range(n_teams)
    )
    states = EnvStates(teams=teams, step_id=data["states/step_id"])

    rewards = tuple(data[f"infos/rewards/{i}"] for i in range(n_teams))
    infos = StepInfo(rewards=rewards, done=data["infos/done"])

    return Simulation(env_config=env_config, rules=None, states=states, infos=infos)
