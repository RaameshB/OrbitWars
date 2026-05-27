# RL Integration (evosax + QDax)

This guide explains how to use JAX-compatible backends in evolutionary/RL loops.

## Parity-first recommendation

For Kaggle-faithful training/evaluation loops, prefer:

- `JAXParityOrbitWarsSimulator` from `jax_parity_backend.py`

It provides:

- Official transition dynamics (via reference interpreter)
- JAX-friendly tensor observations via `observations_jax()`

## Throughput-first option

For approximation-first pretraining, use:

- `jax_core.py`

It provides:

- Pure JAX arrays (PyTree-compatible)
- Batched environments (`num_envs`)
- JIT-compatible step function
- Fixed-size observations/actions

## Typical rollout structure (parity-first path)

1. Create `JAXParityOrbitWarsSimulator`
2. Get tensor obs using `observations_jax()`
3. Flatten via `flatten_observations_jax(...)`
4. Policy emits (`source_logits`, `angle`, `send_fraction`)
5. Decode via `decode_actions(...)`
6. Step with simulator `step(actions)`
7. Accumulate fitness via `rollout_episode(...)` or `evaluate_policy_seeds(...)`

## Action head design

Given observation embedding `h`, produce:

- `source_logits`: dense layer -> `[Players, MaxPlanets]`
- `angle`: dense layer -> `[Players]` (map to `[-pi, pi]` if desired)
- `send_fraction`: sigmoid head -> `[Players]`

For population evaluation:

- `vmap` policy over population axis
- `vmap`/reshape over environment axis depending on evaluator design

## Fitness choices

Common scalar objectives:

- Mean terminal reward per player
- Win rate against random/starter opponents
- Average normalized score difference using `info["scores"]`

For multi-agent self-play setups:

- Track per-player fitness separately
- Or aggregate team/role-specific fitness depending on algorithm

## Recommended training setup

- Start with 2-player mode (`num_players=2`)
- Moderate capacities (`max_planets=48`, `max_fleets=512`)
- Batch 64-512 envs depending on hardware
- Use fixed episode length for stable compilation behavior

## Adapter utility module

Use `orbit_wars_fast/rl_adapter.py` for ready-to-use helpers:

- `flatten_observations_jax`
- `decode_actions`
- `rollout_episode`
- `evaluate_policy_seeds`

## Important caveat

If you use approximation-first `jax_core.py`, periodically validate transfer in parity backends and official Kaggle runtime.
