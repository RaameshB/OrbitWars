from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any

from orbit_wars_fast.fast_sim import FastOrbitWarsSimulator, random_agent_action
from orbit_wars_fast.jax_parity_backend import JAXParityOrbitWarsSimulator
from orbit_wars_fast.reference_adapter import ReferenceOrbitWarsSimulator


def _canonical_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(
        snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _generate_actions(
    observations: list[dict[str, Any]], num_agents: int, rng: random.Random
) -> list[list[list[Any]]]:
    return [random_agent_action(observations[i], rng) for i in range(num_agents)]


def run_byte_accuracy_test(
    num_games: int,
    num_agents: int,
    max_steps: int,
    base_seed: int,
    include_jax_parity: bool,
) -> None:
    print(
        f"Running byte-accuracy test: games={num_games}, agents={num_agents}, max_steps={max_steps}, seed={base_seed}"
    )

    ref_time = 0.0
    fast_time = 0.0
    total_steps = 0

    for game_idx in range(num_games):
        game_seed = base_seed + game_idx
        policy_rng = random.Random((base_seed * 10_000) + game_idx)

        ref = ReferenceOrbitWarsSimulator(num_agents=num_agents, seed=game_seed)
        fast = FastOrbitWarsSimulator(num_agents=num_agents, seed=game_seed)
        jax_parity = (
            JAXParityOrbitWarsSimulator(num_agents=num_agents, seed=game_seed)
            if include_jax_parity
            else None
        )
        jax_parity_parallel_numba = (
            JAXParityOrbitWarsSimulator(
                num_agents=num_agents, seed=game_seed, use_parallel_parity_numba=True
            )
            if include_jax_parity
            else None
        )

        ref_obs = ref.observations()
        fast_obs = fast.observations()

        ref0 = _canonical_bytes(ref.snapshot())
        fast0 = _canonical_bytes(fast.snapshot())
        if ref0 != fast0:
            raise AssertionError(
                f"Initial snapshot mismatch (reference vs fast) for game={game_idx}, seed={game_seed}"
            )
        if jax_parity is not None:
            jax0 = _canonical_bytes(jax_parity.snapshot())
            if ref0 != jax0:
                raise AssertionError(
                    f"Initial snapshot mismatch (reference vs jax_parity) for game={game_idx}, seed={game_seed}"
                )
            if jax_parity_parallel_numba is not None:
                jax_fn0 = _canonical_bytes(jax_parity_parallel_numba.snapshot())
                if ref0 != jax_fn0:
                    raise AssertionError(
                        f"Initial snapshot mismatch (reference vs jax_parity_parallel_numba) for game={game_idx}, seed={game_seed}"
                    )

        for step in range(max_steps):
            if ref.done or fast.done:
                if ref.done != fast.done:
                    raise AssertionError(
                        f"Done-flag mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref.done}, fast={fast.done}"
                    )
                break

            actions = _generate_actions(ref_obs, num_agents, policy_rng)

            t0 = time.perf_counter()
            ref_obs, ref_done, ref_reward = ref.step(actions)
            ref_time += time.perf_counter() - t0

            t1 = time.perf_counter()
            fast_obs, fast_done, fast_reward = fast.step(actions)
            fast_time += time.perf_counter() - t1

            if jax_parity is not None:
                jax_obs, jax_done, jax_reward = jax_parity.step(actions)
                _ = jax_obs
                if jax_parity_parallel_numba is not None:
                    jax_fn_obs, jax_fn_done, jax_fn_reward = jax_parity_parallel_numba.step(actions)
                    _ = jax_fn_obs
                else:
                    jax_fn_done = ref_done
                    jax_fn_reward = ref_reward
            else:
                jax_done = ref_done
                jax_reward = ref_reward
                jax_fn_done = ref_done
                jax_fn_reward = ref_reward

            total_steps += 1

            if ref_done != fast_done:
                raise AssertionError(
                    f"Done mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_done}, fast={fast_done}"
                )
            if ref_reward != fast_reward:
                raise AssertionError(
                    f"Reward mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_reward}, fast={fast_reward}"
                )
            if jax_parity is not None and ref_done != jax_done:
                raise AssertionError(
                    f"Done mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_done}, jax_parity={jax_done}"
                )
            if jax_parity is not None and ref_reward != jax_reward:
                raise AssertionError(
                    f"Reward mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_reward}, jax_parity={jax_reward}"
                )
            if jax_parity_parallel_numba is not None and ref_done != jax_fn_done:
                raise AssertionError(
                    f"Done mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_done}, jax_parity_parallel_numba={jax_fn_done}"
                )
            if jax_parity_parallel_numba is not None and ref_reward != jax_fn_reward:
                raise AssertionError(
                    f"Reward mismatch at game={game_idx}, step={step}, seed={game_seed}: ref={ref_reward}, jax_parity_parallel_numba={jax_fn_reward}"
                )

            ref_bytes = _canonical_bytes(ref.snapshot())
            fast_bytes = _canonical_bytes(fast.snapshot())
            jax_bytes = _canonical_bytes(jax_parity.snapshot()) if jax_parity is not None else ref_bytes
            jax_fn_bytes = _canonical_bytes(jax_parity_parallel_numba.snapshot()) if jax_parity_parallel_numba is not None else ref_bytes
            
            if ref_bytes != fast_bytes or ref_bytes != jax_bytes or ref_bytes != jax_fn_bytes:
                debug_path = (
                    f"orbit_wars_fast/mismatch_game_{game_idx}_step_{step}.json"
                )
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "game_idx": game_idx,
                            "step": step,
                            "seed": game_seed,
                            "actions": actions,
                            "reference": ref.snapshot(),
                            "fast": fast.snapshot(),
                            "jax_parity": jax_parity.snapshot() if jax_parity is not None else None,
                            "jax_parity_parallel_numba": jax_parity_parallel_numba.snapshot() if jax_parity_parallel_numba is not None else None,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                raise AssertionError(
                    f"Snapshot mismatch at game={game_idx}, step={step}, seed={game_seed}. Debug written to {debug_path}"
                )

        if jax_parity_parallel_numba is not None:
            jax_parity_parallel_numba.close()

    print(f"✅ Byte-accuracy PASSED over {num_games} games and {total_steps} steps.")
    if total_steps > 0:
        print(
            f"Reference step time total: {ref_time:.6f}s ({1e3 * ref_time / total_steps:.3f} ms/step)"
        )
        print(
            f"Fast sim step time total:  {fast_time:.6f}s ({1e3 * fast_time / total_steps:.3f} ms/step)"
        )
        print(f"Speedup: {ref_time / max(fast_time, 1e-12):.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Byte-accuracy checker for fast Orbit Wars simulator"
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--agents", type=int, default=2, choices=[2, 4])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--skip-jax-parity", action="store_true")
    args = parser.parse_args()

    run_byte_accuracy_test(
        num_games=args.games,
        num_agents=args.agents,
        max_steps=args.steps,
        base_seed=args.seed,
        include_jax_parity=not args.skip_jax_parity,
    )


if __name__ == "__main__":
    main()
