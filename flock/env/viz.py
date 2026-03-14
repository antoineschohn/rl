"""Ugly-but-useful matplotlib replay for physics sanity checking."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flock.env.types import Simulation


def animate(sim: Simulation, interval: int = 50):
    """Matplotlib animation of a simulation. Red = predators, blue = prey."""
    config = sim.config
    traj = sim.states

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlim(0, config.arena_size)
    ax.set_ylim(0, config.arena_size)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")

    pred_scatter = ax.scatter([], [], c="red", s=40, label="predators")
    prey_scatter = ax.scatter([], [], c="dodgerblue", s=20, label="prey")
    step_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes,
        color="white", fontsize=10, va="top",
    )
    ax.legend(loc="upper right", fontsize=8)

    n_frames = len(traj.step_id)

    def update(frame):
        pred_scatter.set_offsets(traj.predators.pos[frame])
        pred_scatter.set_alpha(traj.predators.alive[frame].astype(float))

        prey_scatter.set_offsets(traj.prey.pos[frame])
        prey_scatter.set_alpha(traj.prey.alive[frame].astype(float))

        alive_prey = int(traj.prey.alive[frame].sum())
        step_text.set_text(f"step {frame}/{n_frames - 1}  |  prey alive: {alive_prey}")
        return pred_scatter, prey_scatter, step_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval, blit=True,
    )
    plt.show()
    return ani


if __name__ == "__main__":
    import jax
    from flock.env.core import run_episode, random_policy
    from flock.env.types import EnvConfig

    config = EnvConfig()
    sim = run_episode(
        config, jax.random.key(42),
        pred_policy=random_policy(config.n_predators),
        prey_policy=random_policy(config.n_prey),
    )
    animate(sim)
