# (ugly-but-useful)

import jax.numpy as jnp
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
    return ani


def plot_stats(sim: Simulation):
    """Plot prey count and cumulative rewards over time."""
    infos = sim.infos
    T = len(infos.done)

    prey_alive = sim.states.prey.alive[1:]  # skip t=0 to align with infos
    prey_count = prey_alive.sum(axis=-1)    # (T,)

    pred_reward_per_step = infos.pred_reward.sum(axis=-1)  # (T,) total across predators
    prey_reward_per_step = infos.prey_reward.sum(axis=-1)  # (T,) total across prey

    pred_cumulative = jnp.cumsum(pred_reward_per_step)
    prey_cumulative = jnp.cumsum(prey_reward_per_step)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(prey_count, color="dodgerblue")
    axes[0].set_ylabel("prey alive")
    axes[0].set_title("prey count over time")

    axes[1].plot(pred_cumulative, color="red", label="predators")
    axes[1].plot(prey_cumulative, color="dodgerblue", label="prey")
    axes[1].set_ylabel("cumulative reward")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].set_title("cumulative reward over time")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import jax
    from flock.env.core import run_episode
    from flock.env.types import EnvConfig

    from flock.env.core import RandomPolicy
    config = EnvConfig()
    sim = run_episode(
        config, jax.random.key(42),
        pred_policy=RandomPolicy(config.n_predators),
        prey_policy=RandomPolicy(config.n_prey),
    )
    animate(sim)
    plt.show()
