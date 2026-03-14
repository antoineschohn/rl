# (ugly-but-useful)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flock.env.types import Simulation

TEAM_COLORS = ["red", "dodgerblue", "green", "orange", "purple", "cyan", "magenta", "yellow"]


def animate(sim: Simulation, interval: int = 50):
    """Matplotlib animation of a simulation. One color per team."""
    env_config = sim.env_config
    rules = sim.rules
    traj = sim.states

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, env_config.arena_size)
    ax.set_ylim(0, env_config.arena_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    scatters = []
    for i, tc in enumerate(rules.teams):
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        s = ax.scatter([], [], c=color, s=30, label=tc.name)
        scatters.append(s)

    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        color="white", fontsize=10, va="top",
    )
    ax.legend(loc="upper right", fontsize=8)

    n_frames = len(traj.step_id)

    def update(frame):
        info_parts = []
        for i, (sc, tc) in enumerate(zip(scatters, rules.teams)):
            team = traj.teams[i]
            sc.set_offsets(team.pos[frame])
            sc.set_alpha(team.alive[frame].astype(float))
            alive = int(team.alive[frame].sum())
            info_parts.append(f"{tc.name}: {alive}")

        step_text.set_text(f"step {frame}/{n_frames - 1}  |  " + "  ".join(info_parts))
        return (*scatters, step_text)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True,
    )
    return ani


def plot_stats(sim: Simulation):
    """Plot alive counts and cumulative rewards per team over time."""
    rules = sim.rules
    infos = sim.infos
    n_teams = len(rules.teams)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    # Alive counts
    for i, tc in enumerate(rules.teams):
        alive = sim.states.teams[i].alive[1:]  # skip t=0 to align with infos
        count = alive.sum(axis=-1)
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        axes[0].plot(count, color=color, label=tc.name)
    axes[0].set_ylabel("alive")
    axes[0].set_title("alive count over time")
    axes[0].legend()

    # Cumulative rewards
    for i, tc in enumerate(rules.teams):
        reward_per_step = infos.scores[i].sum(axis=-1)  # sum across agents
        cumulative = jnp.cumsum(reward_per_step)
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        axes[1].plot(cumulative, color=color, label=tc.name)
    axes[1].set_ylabel("cumulative reward")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].set_title("cumulative reward over time")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import jax
    from flock.env.core import run_episode, RandomPolicy
    from flock.env.types import EnvConfig
    from flock.env.rules import PredatorPrey

    env_config = EnvConfig()
    rules = PredatorPrey()
    policies = tuple(RandomPolicy(tc) for tc in rules.teams)
    sim = run_episode(env_config, rules, jax.random.key(42), policies)
    animate(sim)
    plt.show()
