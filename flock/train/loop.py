"""Training loop: collect rollouts across parallel arenas, PPO update, repeat."""

import equinox as eqx
import jax
import optax

from flock.env.types import EnvConfig, RngKey
from flock.env.rules import Rules
from flock.train.policy import make_policy
from flock.train.ppo import TrainConfig, collect_rollouts, train_step


def train(
    env_config: EnvConfig,
    rules: Rules,
    train_config: TrainConfig = TrainConfig(),
    key: RngKey = jax.random.key(0),
) -> tuple:
    """Train all team policies via PPO co-training.

    Returns tuple of trained policies, one per team.
    """
    n_teams = len(rules.teams)
    keys = jax.random.split(key, n_teams + 1)
    key = keys[0]

    from flock.env.core import RandomPolicy
    policies = tuple(
        make_policy(tc.k_teammates, tc.k_opponents, key=keys[i + 1])
        for i, tc in enumerate(rules.teams)
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adam(train_config.lr),
    )
    opt_states = tuple(
        optimizer.init(eqx.filter(p, eqx.is_array)) for p in policies
    )

    for step_i in range(train_config.n_train_steps):
        keys = jax.random.split(key, 2 * n_teams + 1)
        key = keys[0]

        # Collect rollouts and update each team
        new_policies = list(policies)
        new_opt_states = list(opt_states)
        all_metrics = []

        for i, tc in enumerate(rules.teams):
            rollouts = collect_rollouts(
                env_config, rules, policies,
                keys[i + 1], i, train_config.n_arenas,
            )
            # Merge arenas: (n_arenas, T, n_agents, ...) -> (T * n_arenas, n_agents, ...)
            merged = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]),
                rollouts,
            )
            p, os, m = train_step(
                policies[i], opt_states[i], optimizer, merged,
                train_config, keys[n_teams + 1 + i],
            )
            new_policies[i] = p
            new_opt_states[i] = os
            all_metrics.append(m)

        policies = tuple(new_policies)
        opt_states = tuple(new_opt_states)

        if step_i % 10 == 0:
            parts = []
            for i, tc in enumerate(rules.teams):
                m = all_metrics[i]
                parts.append(f"{tc.name} loss: {m['total_loss']:.4f} ent: {m['entropy']:.4f}")
            print(f"step {step_i:4d} | " + "  |  ".join(parts))

    return policies
