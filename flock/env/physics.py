import jax.numpy as jnp


def clamp_magnitude(v, max_mag):
    """Clamp vector magnitude, preserving direction."""
    mag = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.where(mag > max_mag, v / mag * max_mag, v)


def integrate(pos, vel, accel, dt, max_speed):
    """Euler integration with velocity clamping."""
    vel = clamp_magnitude(vel + accel * dt, max_speed)
    pos = pos + vel * dt
    return pos, vel


def wrap_position(pos, arena_size):
    """Toroidal wrapping — agents exiting one side appear on the other."""
    return pos % arena_size


def pairwise_distances(pos_a, pos_b, arena_size):
    """Toroidal pairwise distances between two sets of agents.

    Returns distances of shape (n_a, n_b), accounting for wrap-around.
    """
    # (n_a, 1, 2) - (1, n_b, 2) -> (n_a, n_b, 2)
    delta = pos_a[:, None, :] - pos_b[None, :, :]
    # Shortest distance on torus
    delta = delta - arena_size * jnp.round(delta / arena_size)
    return jnp.linalg.norm(delta, axis=-1)


def pairwise_deltas(pos_a, pos_b, arena_size):
    """Toroidal pairwise displacement vectors from pos_b to pos_a.

    Returns (n_a, n_b, 2) — the direction from each b to each a,
    accounting for wrap-around.
    """
    delta = pos_a[:, None, :] - pos_b[None, :, :]
    return delta - arena_size * jnp.round(delta / arena_size)
