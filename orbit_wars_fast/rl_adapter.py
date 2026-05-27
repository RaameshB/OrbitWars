from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from orbit_wars_fast.jax_parity_backend import JAXParityOrbitWarsSimulator


Array = jax.Array


@dataclass
class RolloutResult:
    episode_rewards: np.ndarray
    steps: int
    done: bool
    final_snapshot: dict[str, Any]


def flatten_observations_jax(obs_jax: dict[str, Any]) -> Array:
    """Flatten `JAXParityOrbitWarsSimulator.observations_jax()` into policy features.

    Returns shape: [num_agents, feature_dim]
    """
    planets = obs_jax["planets"]
    fleets = obs_jax["fleets"]

    planet_features = jnp.concatenate(
        [
            planets["owner"].astype(jnp.float32)[..., None],
            planets["x"][..., None] / 100.0,
            planets["y"][..., None] / 100.0,
            planets["radius"][..., None] / 10.0,
            planets["ships"][..., None] / 200.0,
            planets["production"][..., None] / 5.0,
            planets["alive"].astype(jnp.float32)[..., None],
        ],
        axis=-1,
    )

    fleet_features = jnp.concatenate(
        [
            fleets["owner"].astype(jnp.float32)[..., None],
            fleets["x"][..., None] / 100.0,
            fleets["y"][..., None] / 100.0,
            jnp.sin(fleets["angle"])[..., None],
            jnp.cos(fleets["angle"])[..., None],
            fleets["ships"][..., None] / 200.0,
            fleets["alive"].astype(jnp.float32)[..., None],
        ],
        axis=-1,
    )

    num_agents = planet_features.shape[0]
    flat_planets = planet_features.reshape(num_agents, -1)
    flat_fleets = fleet_features.reshape(num_agents, -1)

    scalars = jnp.stack(
        [
            obs_jax["player"].astype(jnp.float32),
            obs_jax["angular_velocity"].astype(jnp.float32),
            obs_jax["next_fleet_id"].astype(jnp.float32) / 5000.0,
            obs_jax["step"].astype(jnp.float32) / 500.0,
            obs_jax["remainingOverageTime"].astype(jnp.float32) / 60.0,
        ],
        axis=-1,
    )

    return jnp.concatenate([flat_planets, flat_fleets, scalars], axis=-1)


def decode_actions(
    observations: list[dict[str, Any]],
    source_logits: np.ndarray,
    angle: np.ndarray,
    send_fraction: np.ndarray,
) -> list[list[list[Any]]]:
    """Decode policy tensors into Orbit Wars action lists.

    Expected policy output shapes:
    - source_logits: [num_agents, max_planets]
    - angle: [num_agents]
    - send_fraction: [num_agents]
    """
    num_agents = len(observations)
    actions: list[list[list[Any]]] = [[] for _ in range(num_agents)]

    for player in range(num_agents):
        obs = observations[player]
        planets = obs["planets"]
        if len(planets) == 0:
            continue

        logits = source_logits[player]
        frac = float(np.clip(send_fraction[player], 0.0, 1.0))
        theta = float(angle[player])

        valid_indices = []
        valid_scores = []
        for idx, p in enumerate(planets):
            if p[1] == player and p[5] > 0:
                valid_indices.append(idx)
                valid_scores.append(float(logits[idx]) if idx < logits.shape[0] else -1e9)

        if not valid_indices:
            continue

        best_local = int(np.argmax(np.asarray(valid_scores)))
        chosen_planet = planets[valid_indices[best_local]]

        ships_available = int(chosen_planet[5])
        ships_to_send = int(np.floor(ships_available * frac))
        if ships_to_send <= 0 and ships_available > 0:
            ships_to_send = 1
        ships_to_send = min(ships_to_send, ships_available)

        if ships_to_send > 0:
            actions[player].append([int(chosen_planet[0]), theta, ships_to_send])

    return actions


def rollout_episode(
    simulator: JAXParityOrbitWarsSimulator,
    policy_apply: Callable[[Any, Array, Array], dict[str, Array]],
    params: Any,
    rng_key: Array,
    max_steps: int | None = None,
) -> RolloutResult:
    """Run one episode on parity backend using a JAX policy.

    `policy_apply(params, obs_features, key)` must return dict with keys:
    - `source_logits`: [num_agents, max_planets]
    - `angle`: [num_agents]
    - `send_fraction`: [num_agents]
    """
    if max_steps is None:
        max_steps = 500

    simulator.reset()
    cumulative = np.zeros((simulator.num_agents,), dtype=np.float32)
    done = simulator.done
    steps = 0

    while (not done) and steps < max_steps:
        obs_raw = simulator.observations()
        obs_jax = simulator.observations_jax()
        obs_features = flatten_observations_jax(obs_jax)

        rng_key, policy_key = jax.random.split(rng_key)
        outputs = policy_apply(params, obs_features, policy_key)

        source_logits = np.asarray(outputs["source_logits"], dtype=np.float32)
        angle = np.asarray(outputs["angle"], dtype=np.float32)
        send_fraction = np.asarray(outputs["send_fraction"], dtype=np.float32)

        actions = decode_actions(obs_raw, source_logits, angle, send_fraction)
        _, done, reward = simulator.step(actions)
        cumulative += np.asarray(reward, dtype=np.float32)
        steps += 1

    return RolloutResult(
        episode_rewards=cumulative,
        steps=steps,
        done=done,
        final_snapshot=simulator.snapshot(),
    )


def evaluate_policy_seeds(
    make_simulator: Callable[[int], JAXParityOrbitWarsSimulator],
    policy_apply: Callable[[Any, Array, Array], dict[str, Array]],
    params: Any,
    seeds: list[int],
    base_key: Array,
    max_steps: int = 500,
) -> dict[str, Any]:
    """Evaluate one policy across multiple seeds.

    Returns mean per-player reward and scalar fitness (player 0 mean reward).
    """
    episode_rewards = []
    key = base_key

    for seed in seeds:
        sim = make_simulator(seed)
        key, ep_key = jax.random.split(key)
        result = rollout_episode(
            simulator=sim,
            policy_apply=policy_apply,
            params=params,
            rng_key=ep_key,
            max_steps=max_steps,
        )
        episode_rewards.append(result.episode_rewards)

    rewards = np.stack(episode_rewards, axis=0)
    mean_per_player = rewards.mean(axis=0)

    return {
        "mean_episode_reward_per_player": mean_per_player,
        "fitness": float(mean_per_player[0]),
        "num_episodes": len(seeds),
    }
