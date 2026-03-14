"""Training loop: collect rollouts across parallel arenas, PPO update, repeat."""

import equinox as eqx
import jax
import optax

from flock.env.types import EnvConfig, RngKey
from flock.env.core import RandomPolicy
from flock.train.policy import make_policy
from flock.train.ppo import TrainConfig, collect_rollouts, train_step


def train(
    env_config: EnvConfig,
    train_config: TrainConfig = TrainConfig(),
    key: RngKey = jax.random.key(0),
):
    """Train predator and prey policies via PPO co-training.

    Both teams train simultaneously, each treating the other as the environment.
    """
    key, k_pred, k_prey = jax.random.split(key, 3)

    pred_policy = make_policy(env_config.k_teammates, env_config.k_opponents, key=k_pred)
    prey_policy = make_policy(env_config.k_teammates, env_config.k_opponents, key=k_prey)

    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adam(train_config.lr),
    )
    pred_opt_state = optimizer.init(eqx.filter(pred_policy, eqx.is_array))
    prey_opt_state = optimizer.init(eqx.filter(prey_policy, eqx.is_array))

    for step_i in range(train_config.n_train_steps):
        key, k_rollout_pred, k_rollout_prey, k_update_pred, k_update_prey = jax.random.split(key, 5)

        # Collect rollouts via run_episodes + hook
        pred_rollouts = collect_rollouts(
            env_config, pred_policy, prey_policy, k_rollout_pred,
            "predators", train_config.n_arenas,
        )
        prey_rollouts = collect_rollouts(
            env_config, prey_policy, pred_policy, k_rollout_prey,
            "prey", train_config.n_arenas,
        )

        # Merge arenas: (n_arenas, T, n_agents, ...) -> (T * n_arenas, n_agents, ...)
        def merge_arenas(rollout):
            return jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]) if x.ndim > 2 else x.reshape(-1, *x.shape[2:]),
                rollout,
            )

        pred_rollout_merged = merge_arenas(pred_rollouts)
        prey_rollout_merged = merge_arenas(prey_rollouts)

        # PPO updates
        pred_policy, pred_opt_state, pred_metrics = train_step(
            pred_policy, pred_opt_state, optimizer, pred_rollout_merged,
            train_config, k_update_pred,
        )
        prey_policy, prey_opt_state, prey_metrics = train_step(
            prey_policy, prey_opt_state, optimizer, prey_rollout_merged,
            train_config, k_update_prey,
        )

        if step_i % 10 == 0:
            print(f"step {step_i:4d} | "
                  f"pred loss: {pred_metrics['total_loss']:.4f}  "
                  f"prey loss: {prey_metrics['total_loss']:.4f}  "
                  f"pred entropy: {pred_metrics['entropy']:.4f}  "
                  f"prey entropy: {prey_metrics['entropy']:.4f}")

    return pred_policy, prey_policy
