from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_fast.jax_parity_backend import JAXParityOrbitWarsSimulator
from orbit_wars_fast.rl_adapter import evaluate_policy_seeds

Array = jax.Array
PolicyApply = Callable[[Any, Array, Array], dict[str, Array]]


def make_parity_simulator_factory(
    num_agents: int = 2,
    max_planets: int = 64,
    max_fleets: int = 1024,
) -> Callable[[int], JAXParityOrbitWarsSimulator]:
    def _factory(seed: int) -> JAXParityOrbitWarsSimulator:
        return JAXParityOrbitWarsSimulator(
            num_agents=num_agents,
            seed=seed,
            max_planets=max_planets,
            max_fleets=max_fleets,
        )

    return _factory


def evaluate_candidate_parity(
    params: Any,
    policy_apply: PolicyApply,
    seeds: list[int],
    base_key: Array,
    simulator_factory: Callable[[int], JAXParityOrbitWarsSimulator] | None = None,
    max_steps: int = 500,
    fitness_player_index: int = 0,
) -> dict[str, Any]:
    """Evaluate a single candidate policy on parity backend across seeds."""
    if simulator_factory is None:
        simulator_factory = make_parity_simulator_factory()

    result = evaluate_policy_seeds(
        make_simulator=simulator_factory,
        policy_apply=policy_apply,
        params=params,
        seeds=seeds,
        base_key=base_key,
        max_steps=max_steps,
    )

    mean_per_player = np.asarray(result["mean_episode_reward_per_player"], dtype=np.float32)
    fitness = float(mean_per_player[fitness_player_index])

    return {
        "fitness": fitness,
        "mean_episode_reward_per_player": mean_per_player,
        "num_episodes": int(result["num_episodes"]),
    }


def evaluate_population_parity(
    params_population: Any,
    policy_apply: PolicyApply,
    seeds: list[int],
    base_key: Array,
    simulator_factory: Callable[[int], JAXParityOrbitWarsSimulator] | None = None,
    max_steps: int = 500,
    fitness_player_index: int = 0,
) -> dict[str, Any]:
    """Evaluate population of candidates (evosax-style outer loop helper).

    `params_population` is any pytree where axis 0 is population.
    """
    if simulator_factory is None:
        simulator_factory = make_parity_simulator_factory()

    leaves = jax.tree_util.tree_leaves(params_population)
    if not leaves:
        raise ValueError("params_population must contain at least one leaf")
    pop_size = int(leaves[0].shape[0])

    fitnesses = np.zeros((pop_size,), dtype=np.float32)
    mean_rewards = []

    key = base_key
    for i in range(pop_size):
        key, eval_key = jax.random.split(key)
        params_i = jax.tree_util.tree_map(lambda x: x[i], params_population)
        out = evaluate_candidate_parity(
            params=params_i,
            policy_apply=policy_apply,
            seeds=seeds,
            base_key=eval_key,
            simulator_factory=simulator_factory,
            max_steps=max_steps,
            fitness_player_index=fitness_player_index,
        )
        fitnesses[i] = out["fitness"]
        mean_rewards.append(out["mean_episode_reward_per_player"])

    return {
        "fitnesses": fitnesses,
        "mean_episode_reward_per_player": np.stack(mean_rewards, axis=0),
    }


def make_qdax_scoring_fn(
    policy_apply: PolicyApply,
    seeds: list[int],
    simulator_factory: Callable[[int], JAXParityOrbitWarsSimulator] | None = None,
    max_steps: int = 500,
    fitness_player_index: int = 0,
):
    """Create a QDax-compatible scoring function.

    Returns function with signature:
    `(genotypes, random_key) -> (fitnesses, descriptors, extra_scores, random_key)`
    """

    def scoring_fn(genotypes: Any, random_key: Array):
        if simulator_factory is None:
            sim_factory = make_parity_simulator_factory()
        else:
            sim_factory = simulator_factory

        pop_eval = evaluate_population_parity(
            params_population=genotypes,
            policy_apply=policy_apply,
            seeds=seeds,
            base_key=random_key,
            simulator_factory=sim_factory,
            max_steps=max_steps,
            fitness_player_index=fitness_player_index,
        )

        fitnesses = jnp.asarray(pop_eval["fitnesses"], dtype=jnp.float32)
        mean_rewards = jnp.asarray(
            pop_eval["mean_episode_reward_per_player"], dtype=jnp.float32
        )

        # Simple descriptor: first two player means (pad if needed).
        if mean_rewards.shape[1] == 1:
            descriptors = jnp.concatenate([mean_rewards, jnp.zeros_like(mean_rewards)], axis=1)
        else:
            descriptors = mean_rewards[:, :2]

        extra_scores = {"mean_episode_reward_per_player": mean_rewards}
        random_key, new_key = jax.random.split(random_key)
        return fitnesses, descriptors, extra_scores, new_key

    return scoring_fn
