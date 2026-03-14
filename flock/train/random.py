import jax

from flock.env.types import Observations, Policy, PolicyState, RngKey, TeamConfig


class RandomPolicy(Policy):
    """Stateless policy that applies random accelerations scaled to max_accel."""
    n_agents: int
    max_accel: float

    def __init__(self, team_config: TeamConfig):
        self.n_agents = team_config.n_agents
        self.max_accel = team_config.max_accel

    def __call__(self, obs: Observations, key: RngKey, state: PolicyState) -> tuple[jax.Array, PolicyState]:
        return jax.random.normal(key, (self.n_agents, 2)) * self.max_accel, state
