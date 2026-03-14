import jax.numpy as jnp
from flock.env.physics import integrate, wrap_position, pairwise_distances


def test_stationary_agent():
    pos = jnp.array([[5.0, 5.0]])
    vel = jnp.zeros((1, 2))
    accel = jnp.zeros((1, 2))
    new_pos, new_vel = integrate(pos, vel, accel, dt=0.05, max_speed=3.0)
    assert jnp.allclose(new_pos, pos)
    assert jnp.allclose(new_vel, vel)


def test_constant_acceleration():
    pos = jnp.array([[0.0, 0.0]])
    vel = jnp.zeros((1, 2))
    accel = jnp.array([[1.0, 0.0]])
    new_pos, new_vel = integrate(pos, vel, accel, dt=0.1, max_speed=10.0)
    assert jnp.allclose(new_vel, jnp.array([[0.1, 0.0]]))
    assert jnp.allclose(new_pos, jnp.array([[0.01, 0.0]]))


def test_speed_clipping():
    pos = jnp.array([[0.0, 0.0]])
    vel = jnp.array([[3.0, 0.0]])
    accel = jnp.array([[100.0, 0.0]])  # huge accel
    _, new_vel = integrate(pos, vel, accel, dt=0.1, max_speed=5.0)
    speed = jnp.linalg.norm(new_vel)
    assert speed <= 5.0 + 1e-5


def test_wrap_position():
    pos = jnp.array([[11.0, -1.0]])
    wrapped = wrap_position(pos, arena_size=10.0)
    assert jnp.allclose(wrapped, jnp.array([[1.0, 9.0]]))


def test_pairwise_distances_shape():
    a = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    b = jnp.array([[5.0, 5.0], [6.0, 6.0]])
    dists = pairwise_distances(a, b, arena_size=10.0)
    assert dists.shape == (3, 2)


def test_pairwise_distances_toroidal():
    """Agent at 0.5 and agent at 9.5 should be distance 1.0 on a size-10 torus."""
    a = jnp.array([[0.5, 5.0]])
    b = jnp.array([[9.5, 5.0]])
    dists = pairwise_distances(a, b, arena_size=10.0)
    assert jnp.allclose(dists, jnp.array([[1.0]]))


def test_pairwise_distances_self():
    a = jnp.array([[3.0, 4.0]])
    dists = pairwise_distances(a, a, arena_size=10.0)
    assert jnp.allclose(dists, jnp.array([[0.0]]))
