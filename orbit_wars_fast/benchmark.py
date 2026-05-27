from __future__ import annotations

import argparse
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
from kaggle_environments import make

from orbit_wars_fast.fast_sim import FastOrbitWarsSimulator, random_agent_action
from orbit_wars_fast.jax_core import default_config, reset as jax_reset, step as jax_step
from orbit_wars_fast.jax_parity_backend import JAXParityOrbitWarsSimulator
from orbit_wars_fast.parity_pool import run_parallel_rollouts
from orbit_wars_fast.reference_adapter import ReferenceOrbitWarsSimulator


def _time_kaggle_env_run(num_games: int, seed: int) -> tuple[float, int]:
    total_time = 0.0
    total_steps = 0
    for i in range(num_games):
        env = make("orbit_wars", configuration={"seed": seed + i}, debug=False)
        t0 = time.perf_counter()
        env.run(["random", "random"])
        total_time += time.perf_counter() - t0
        total_steps += max(0, len(env.steps) - 1)
    return total_time, total_steps


def _generate_actions(
    obs: list[dict[str, Any]], num_agents: int, rng_seed: int
) -> list[list[list[Any]]]:
    import random

    rng = random.Random(rng_seed)
    return [random_agent_action(obs[p], rng) for p in range(num_agents)]


def _time_step_sim(
    sim_cls,
    num_games: int,
    max_steps: int,
    seed: int,
    num_agents: int = 2,
    sim_kwargs: dict[str, Any] | None = None,
) -> tuple[float, int]:
    total_time = 0.0
    total_steps = 0
    kwargs = sim_kwargs or {}

    for game_idx in range(num_games):
        sim = sim_cls(num_agents=num_agents, seed=seed + game_idx, **kwargs)
        obs = sim.observations()
        for step in range(max_steps):
            if sim.done:
                break
            actions = _generate_actions(
                obs,
                num_agents=num_agents,
                rng_seed=(seed * 100000 + game_idx * 1000 + step),
            )
            t0 = time.perf_counter()
            obs, _, _ = sim.step(actions)
            total_time += time.perf_counter() - t0
            total_steps += 1
        if hasattr(sim, "close"):
            sim.close()
    return total_time, total_steps


def _time_parallel_parity(
    num_games: int,
    max_steps: int,
    seed: int,
    workers: int,
    num_agents: int = 2,
    use_numba: bool = False,
) -> tuple[float, int]:
    seeds = [seed + i for i in range(num_games)]
    t0 = time.perf_counter()
    out = run_parallel_rollouts(
        seeds=seeds,
        num_agents=num_agents,
        max_steps=max_steps,
        workers=workers,
        start_method=None,
        use_numba=use_numba,
    )
    elapsed = time.perf_counter() - t0
    steps = int(sum(int(r["steps"]) for r in out["raw"]))
    return elapsed, steps


def _time_jax_core(
    jax_num_envs: int,
    jax_steps: int,
    seed: int,
    num_players: int = 2,
    warmup_steps: int = 10,
    max_fleets: int = 512,
    use_swept_collision: bool = False,
    use_segment_sun_check: bool = False,
) -> tuple[float, int, dict[str, float]]:
    """Benchmark approximation-first JAX core throughput.

    Measures fully device-synchronized step time over `jax_steps` calls.
    """
    cfg = default_config(
        num_envs=jax_num_envs,
        num_players=num_players,
        max_fleets=max_fleets,
        use_swept_collision=use_swept_collision,
        use_segment_sun_check=use_segment_sun_check,
    )
    key = jax.random.PRNGKey(seed)
    state, _ = jax_reset(key, cfg)

    action = {
        "source_logits": jnp.zeros(
            (cfg.num_envs, cfg.num_players, cfg.max_planets), dtype=jnp.float32
        ),
        "angle": jnp.zeros((cfg.num_envs, cfg.num_players), dtype=jnp.float32),
        "send_fraction": 0.5
        * jnp.ones((cfg.num_envs, cfg.num_players), dtype=jnp.float32),
    }

    # Compile/warmup
    for _ in range(warmup_steps):
        state, _, reward, _, info = jax_step(state, action, cfg)
    _ = jax.block_until_ready(reward)

    t0 = time.perf_counter()
    steps_done = 0
    for _ in range(jax_steps):
        state, _, reward, _, info = jax_step(state, action, cfg)
        steps_done += 1
    _ = jax.block_until_ready(reward)
    elapsed = time.perf_counter() - t0

    info_host = jax.device_get(info)
    overflow_attempts = float(jnp.sum(info_host["fleet_overflow_attempts"]))
    overflow_steps = float(jnp.sum(info_host["fleet_overflow_steps"]))
    fleet_active_max = float(jnp.max(info_host["fleet_active_max"]))
    fleet_capacity = float(info_host["fleet_capacity"][0])

    extra = {
        "overflow_attempts_total": overflow_attempts,
        "overflow_steps_total": overflow_steps,
        "fleet_active_max": fleet_active_max,
        "fleet_capacity": fleet_capacity,
        "overflow_step_rate": overflow_steps / max(float(jax_num_envs * jax_steps), 1.0),
        "fleet_active_max_utilization": fleet_active_max / max(fleet_capacity, 1.0),
    }
    return elapsed, steps_done, extra


def run_benchmark(
    num_games: int,
    max_steps: int,
    seed: int,
    include_kaggle_runtime: bool = True,
    jax_num_envs: int = 256,
    jax_steps: int = 1000,
    include_numba_variant: bool = True,
    include_parallel_parity: bool = True,
    parallel_workers: int = 4,
    include_jax_toggle_sweep: bool = True,
    jax_max_fleets: int = 512,
) -> dict[str, dict[str, float]]:
    """Run benchmark suite and return machine-readable results.

    This function is notebook-friendly and can be called directly from Colab.
    """
    results: dict[str, dict[str, float]] = {}

    if include_kaggle_runtime:
        kaggle_time, kaggle_steps = _time_kaggle_env_run(num_games=num_games, seed=seed)
        results["kaggle env.run"] = {
            "elapsed_s": kaggle_time,
            "steps": float(kaggle_steps),
            "ms_per_step": (kaggle_time / max(kaggle_steps, 1)) * 1e3,
            "env_steps_per_s": float(kaggle_steps) / max(kaggle_time, 1e-12),
        }

    ref_time, ref_steps = _time_step_sim(
        sim_cls=ReferenceOrbitWarsSimulator,
        num_games=num_games,
        max_steps=max_steps,
        seed=seed,
    )
    fast_time, fast_steps = _time_step_sim(
        sim_cls=FastOrbitWarsSimulator,
        num_games=num_games,
        max_steps=max_steps,
        seed=seed,
    )
    jax_parity_time, jax_parity_steps = _time_step_sim(
        sim_cls=JAXParityOrbitWarsSimulator,
        num_games=num_games,
        max_steps=max_steps,
        seed=seed,
    )

    results["reference adapter"] = {
        "elapsed_s": ref_time,
        "steps": float(ref_steps),
        "ms_per_step": (ref_time / max(ref_steps, 1)) * 1e3,
        "env_steps_per_s": float(ref_steps) / max(ref_time, 1e-12),
    }
    results["fast simulator"] = {
        "elapsed_s": fast_time,
        "steps": float(fast_steps),
        "ms_per_step": (fast_time / max(fast_steps, 1)) * 1e3,
        "env_steps_per_s": float(fast_steps) / max(fast_time, 1e-12),
    }
    results["jax parity backend"] = {
        "elapsed_s": jax_parity_time,
        "steps": float(jax_parity_steps),
        "ms_per_step": (jax_parity_time / max(jax_parity_steps, 1)) * 1e3,
        "env_steps_per_s": float(jax_parity_steps) / max(jax_parity_time, 1e-12),
    }

    if include_numba_variant:
        numba_time, numba_steps = _time_step_sim(
            sim_cls=FastOrbitWarsSimulator,
            num_games=num_games,
            max_steps=max_steps,
            seed=seed,
            sim_kwargs={"use_numba": True},
        )
        results["fast simulator (numba)"] = {
            "elapsed_s": numba_time,
            "steps": float(numba_steps),
            "ms_per_step": (numba_time / max(numba_steps, 1)) * 1e3,
            "env_steps_per_s": float(numba_steps) / max(numba_time, 1e-12),
        }
        

    if include_parallel_parity:
        pp_time, pp_steps = _time_parallel_parity(
            num_games=num_games,
            max_steps=max_steps,
            seed=seed,
            workers=parallel_workers,
            num_agents=2,
            use_numba=False,
        )
        results[f"parallel parity ({parallel_workers}w)"] = {
            "elapsed_s": pp_time,
            "steps": float(pp_steps),
            "ms_per_step": (pp_time / max(pp_steps, 1)) * 1e3,
            "env_steps_per_s": float(pp_steps) / max(pp_time, 1e-12),
        }
        if include_numba_variant:
            ppn_time, ppn_steps = _time_parallel_parity(
                num_games=num_games,
                max_steps=max_steps,
                seed=seed,
                workers=parallel_workers,
                num_agents=2,
                use_numba=True,
            )
            results[f"parallel parity numba ({parallel_workers}w)"] = {
                "elapsed_s": ppn_time,
                "steps": float(ppn_steps),
                "ms_per_step": (ppn_time / max(ppn_steps, 1)) * 1e3,
                "env_steps_per_s": float(ppn_steps) / max(ppn_time, 1e-12),
            }

            jax_parity_parallel_time, jax_parity_parallel_steps = _time_step_sim(
                sim_cls=JAXParityOrbitWarsSimulator,
                num_games=num_games,
                max_steps=max_steps,
                seed=seed,
                sim_kwargs={"use_parallel_parity_numba": True},
            )
            results["jax parity backend (parallel numba)"] = {
                "elapsed_s": jax_parity_parallel_time,
                "steps": float(jax_parity_parallel_steps),
                "ms_per_step": (jax_parity_parallel_time / max(jax_parity_parallel_steps, 1)) * 1e3,
                "env_steps_per_s": float(jax_parity_parallel_steps) / max(jax_parity_parallel_time, 1e-12),
            }

    jax_core_time, jax_core_steps, jax_core_extra = _time_jax_core(
        jax_num_envs=jax_num_envs,
        jax_steps=jax_steps,
        seed=seed,
        num_players=2,
        max_fleets=jax_max_fleets,
        use_swept_collision=False,
        use_segment_sun_check=False,
    )
    results["jax core (approx)"] = {
        "elapsed_s": jax_core_time,
        "steps": float(jax_core_steps),
        "ms_per_step": (jax_core_time / max(jax_core_steps, 1)) * 1e3,
        "env_steps_per_s": float(jax_num_envs * jax_core_steps) / max(jax_core_time, 1e-12),
        "num_envs": float(jax_num_envs),
        "max_fleets": float(jax_max_fleets),
        **jax_core_extra,
    }

    if include_jax_toggle_sweep:
        jt1_time, jt1_steps, jt1_extra = _time_jax_core(
            jax_num_envs=jax_num_envs,
            jax_steps=jax_steps,
            seed=seed,
            num_players=2,
            max_fleets=jax_max_fleets,
            use_swept_collision=True,
            use_segment_sun_check=False,
        )
        results["jax core (swept)"] = {
            "elapsed_s": jt1_time,
            "steps": float(jt1_steps),
            "ms_per_step": (jt1_time / max(jt1_steps, 1)) * 1e3,
            "env_steps_per_s": float(jax_num_envs * jt1_steps) / max(jt1_time, 1e-12),
            "num_envs": float(jax_num_envs),
            "max_fleets": float(jax_max_fleets),
            **jt1_extra,
        }

        jt2_time, jt2_steps, jt2_extra = _time_jax_core(
            jax_num_envs=jax_num_envs,
            jax_steps=jax_steps,
            seed=seed,
            num_players=2,
            max_fleets=jax_max_fleets,
            use_swept_collision=False,
            use_segment_sun_check=True,
        )
        results["jax core (segment sun)"] = {
            "elapsed_s": jt2_time,
            "steps": float(jt2_steps),
            "ms_per_step": (jt2_time / max(jt2_steps, 1)) * 1e3,
            "env_steps_per_s": float(jax_num_envs * jt2_steps) / max(jt2_time, 1e-12),
            "num_envs": float(jax_num_envs),
            "max_fleets": float(jax_max_fleets),
            **jt2_extra,
        }

        jt3_time, jt3_steps, jt3_extra = _time_jax_core(
            jax_num_envs=jax_num_envs,
            jax_steps=jax_steps,
            seed=seed,
            num_players=2,
            max_fleets=jax_max_fleets,
            use_swept_collision=True,
            use_segment_sun_check=True,
        )
        results["jax core (swept+sun)"] = {
            "elapsed_s": jt3_time,
            "steps": float(jt3_steps),
            "ms_per_step": (jt3_time / max(jt3_steps, 1)) * 1e3,
            "env_steps_per_s": float(jax_num_envs * jt3_steps) / max(jt3_time, 1e-12),
            "num_envs": float(jax_num_envs),
            "max_fleets": float(jax_max_fleets),
            **jt3_extra,
        }

    if include_kaggle_runtime and "kaggle env.run" in results:
        kaggle_ms = results["kaggle env.run"]["ms_per_step"]
        for k in [
            "fast simulator",
            "jax parity backend",
            "fast simulator (numba)",
            f"parallel parity ({parallel_workers}w)",
            f"parallel parity numba ({parallel_workers}w)",
            "jax parity backend (parallel numba)",
        ]:
            if k in results:
                results[k]["speedup_vs_kaggle"] = kaggle_ms / max(
                    results[k]["ms_per_step"], 1e-12
                )

    # Helpful relative metrics for Colab GPU runs.
    py_ms = results["fast simulator"]["ms_per_step"]
    jax_ms = results["jax core (approx)"]["ms_per_step"]
    results["jax core (approx)"]["speedup_vs_fast_step_latency"] = py_ms / max(
        jax_ms, 1e-12
    )

    if "fast simulator (numba)" in results:
        results["fast simulator (numba)"]["speedup_vs_fast"] = py_ms / max(
            results["fast simulator (numba)"]["ms_per_step"], 1e-12
        )

    if f"parallel parity ({parallel_workers}w)" in results:
        results[f"parallel parity ({parallel_workers}w)"]["speedup_vs_fast"] = py_ms / max(
            results[f"parallel parity ({parallel_workers}w)"]["ms_per_step"], 1e-12
        )
    if f"parallel parity numba ({parallel_workers}w)" in results:
        results[f"parallel parity numba ({parallel_workers}w)"]["speedup_vs_fast"] = py_ms / max(
            results[f"parallel parity numba ({parallel_workers}w)"]["ms_per_step"], 1e-12
        )

    return results


def print_benchmark(results: dict[str, dict[str, float]]) -> None:
    print("Benchmarking Orbit Wars backends")
    for name, row in results.items():
        elapsed = row.get("elapsed_s", 0.0)
        steps = int(row.get("steps", 0.0))
        ms = row.get("ms_per_step", 0.0)
        print(
            f"{name:24s} total={elapsed:8.4f}s | steps={steps:6d} | {ms:8.4f} ms/step"
        )

        if "speedup_vs_kaggle" in row:
            print(f"  speedup_vs_kaggle={row['speedup_vs_kaggle']:.2f}x")
        if "speedup_vs_fast" in row:
            print(f"  speedup_vs_fast={row['speedup_vs_fast']:.2f}x")
        if "env_steps_per_s" in row:
            print(f"  env_steps_per_s={row['env_steps_per_s']:.1f}")
        if "overflow_step_rate" in row:
            print(
                "  overflow_step_rate="
                f"{row['overflow_step_rate']:.6f}, "
                f"fleet_active_max_utilization={row.get('fleet_active_max_utilization', 0.0):.3f}"
            )
        if "speedup_vs_fast_step_latency" in row:
            print(
                "  speedup_vs_fast_step_latency="
                f"{row['speedup_vs_fast_step_latency']:.2f}x"
            )


def benchmark_dataframe(results: dict[str, dict[str, float]]):
    """Optional helper for notebooks: convert to pandas DataFrame."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for name, metrics in results.items():
        row: dict[str, Any] = {"backend": name}
        row.update(dict(metrics))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("backend").reset_index(drop=True)


def run_fleet_cap_sweep(
    caps: list[int],
    jax_num_envs: int,
    jax_steps: int,
    seed: int,
    use_swept_collision: bool = False,
    use_segment_sun_check: bool = False,
) -> list[dict[str, float]]:
    """Run JAX-core sweep across max_fleets caps and collect overflow telemetry."""
    rows: list[dict[str, float]] = []
    for cap in caps:
        elapsed, steps, extra = _time_jax_core(
            jax_num_envs=jax_num_envs,
            jax_steps=jax_steps,
            seed=seed,
            num_players=2,
            max_fleets=int(cap),
            use_swept_collision=use_swept_collision,
            use_segment_sun_check=use_segment_sun_check,
        )
        rows.append(
            {
                "max_fleets": float(cap),
                "elapsed_s": float(elapsed),
                "steps": float(steps),
                "ms_per_step": (float(elapsed) / max(float(steps), 1.0)) * 1e3,
                "env_steps_per_s": float(jax_num_envs * steps) / max(float(elapsed), 1e-12),
                **extra,
            }
        )
    return rows


def recommend_fleet_cap(
    sweep_rows: list[dict[str, float]],
    utilization_target: float = 0.70,
    overflow_step_rate_target: float = 0.0,
    safety_margin: float = 1.30,
) -> dict[str, float]:
    """Recommend cap from sweep rows using overflow + utilization constraints.

    Strategy:
    1) Pick smallest cap meeting overflow and utilization targets.
    2) If none satisfy, estimate from observed max active fleets with safety margin.
    """
    if not sweep_rows:
        raise ValueError("sweep_rows is empty")

    rows = sorted(sweep_rows, key=lambda r: r["max_fleets"])
    feasible = [
        r
        for r in rows
        if r.get("overflow_step_rate", 1.0) <= overflow_step_rate_target
        and r.get("fleet_active_max_utilization", 1.0) <= utilization_target
    ]

    if feasible:
        chosen = feasible[0]
        reason = 1.0
    else:
        max_active = max(r.get("fleet_active_max", 0.0) for r in rows)
        estimated = max(1, int(math.ceil(max_active * safety_margin)))
        # round up to nearest multiple of 64 for tensor-friendly sizing
        rounded = int(math.ceil(estimated / 64.0) * 64)
        closest = min(rows, key=lambda r: abs(r["max_fleets"] - rounded))
        chosen = closest
        reason = 0.0

    return {
        "recommended_max_fleets": float(chosen["max_fleets"]),
        "estimated_from_constraints": reason,
        "overflow_step_rate": float(chosen.get("overflow_step_rate", 0.0)),
        "fleet_active_max_utilization": float(chosen.get("fleet_active_max_utilization", 0.0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Orbit Wars simulation backends")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-kaggle-runtime", action="store_true")
    parser.add_argument("--jax-num-envs", type=int, default=256)
    parser.add_argument("--jax-steps", type=int, default=1000)
    parser.add_argument("--skip-numba", action="store_true")
    parser.add_argument("--skip-parallel-parity", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--skip-jax-toggle-sweep", action="store_true")
    parser.add_argument("--jax-max-fleets", type=int, default=512)
    args = parser.parse_args()

    results = run_benchmark(
        num_games=args.games,
        max_steps=args.steps,
        seed=args.seed,
        include_kaggle_runtime=not args.skip_kaggle_runtime,
        jax_num_envs=args.jax_num_envs,
        jax_steps=args.jax_steps,
        include_numba_variant=not args.skip_numba,
        include_parallel_parity=not args.skip_parallel_parity,
        parallel_workers=args.parallel_workers,
        include_jax_toggle_sweep=not args.skip_jax_toggle_sweep,
        jax_max_fleets=args.jax_max_fleets,
    )
    print_benchmark(results)
