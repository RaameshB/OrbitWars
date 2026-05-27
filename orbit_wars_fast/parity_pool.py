from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from orbit_wars_fast.fast_sim import FastOrbitWarsSimulator, random_agent_action


def _rollout_one(
    seed: int,
    num_agents: int,
    max_steps: int,
    use_numba: bool,
) -> dict[str, Any]:
    import random

    sim = FastOrbitWarsSimulator(
        num_agents=num_agents,
        seed=seed,
        use_numba=use_numba,
    )
    obs = sim.observations()
    rng = random.Random(seed * 1_000_003 + 17)

    steps = 0
    while (not sim.done) and steps < max_steps:
        actions = [random_agent_action(obs[p], rng) for p in range(num_agents)]
        obs, _, _ = sim.step(actions)
        steps += 1

    return {
        "seed": seed,
        "steps": steps,
        "reward": np.asarray(sim.reward, dtype=np.float32),
    }


def _rollout_batch(
    seeds: list[int],
    num_agents: int,
    max_steps: int,
    use_numba: bool,
) -> list[dict[str, Any]]:
    return [_rollout_one(int(seed), num_agents, max_steps, use_numba) for seed in seeds]


def _chunk_list(values: list[int], chunks: int) -> list[list[int]]:
    if chunks <= 1:
        return [values]
    out: list[list[int]] = [[] for _ in range(chunks)]
    for i, v in enumerate(values):
        out[i % chunks].append(v)
    return [x for x in out if x]


def _default_start_method() -> str:
    methods = mp.get_all_start_methods()
    # In Linux/Colab notebooks, "fork" avoids spawn re-import overhead.
    if "fork" in methods:
        return "fork"
    return "spawn"


def run_parallel_rollouts(
    seeds: list[int],
    num_agents: int = 2,
    max_steps: int = 500,
    workers: int | None = None,
    start_method: str | None = None,
    use_numba: bool = False,
) -> dict[str, Any]:
    """Run parity rollouts in parallel processes.

    This preserves parity behavior while scaling throughput on CPU cores.
    """
    if workers is None:
        workers = max(1, (mp.cpu_count() or 2) - 1)
    if start_method is None:
        start_method = _default_start_method()

    workers = max(1, min(workers, len(seeds) if seeds else 1))
    ctx = mp.get_context(start_method)
    results: list[dict[str, Any]] = []

    seed_chunks = _chunk_list([int(s) for s in seeds], workers)

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [
            ex.submit(_rollout_batch, chunk, num_agents, max_steps, use_numba)
            for chunk in seed_chunks
        ]
        for fut in futs:
            results.extend(fut.result())

    results = sorted(results, key=lambda x: x["seed"])
    rewards = np.stack([r["reward"] for r in results], axis=0) if results else np.zeros((0, num_agents), dtype=np.float32)
    steps = np.asarray([r["steps"] for r in results], dtype=np.int32)

    return {
        "num_episodes": len(results),
        "mean_reward_per_player": rewards.mean(axis=0) if len(results) else np.zeros((num_agents,), dtype=np.float32),
        "mean_steps": float(steps.mean()) if len(steps) else 0.0,
        "raw": results,
        "start_method": start_method,
        "workers": workers,
    }
