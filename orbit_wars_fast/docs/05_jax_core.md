# JAX Core Guide

File: `orbit_wars_fast/jax_core.py`

## Purpose

The JAX core is a batched, fixed-shape simulator intended for RL/evolution throughput.

- JIT-compiled `step`
- Vectorized state tensors
- Works well with population/batch evaluation loops

## Fidelity warning

Current JAX core is **not byte-accurate** to Kaggle.

Use byte-accurate Python/parity backends when exact parity matters.

### Exact simplifications/approximations in current JAX core

1. **No comet lifecycle**
   - No comet spawn/move/remove mechanics from official interpreter.
2. **Collision approximation**
   - Uses endpoint planet-distance hit test (`new_fleet_pos` vs `planet_pos`).
   - Does not use official swept continuous collision (`swept_pair_hit`) between moving fleet segment and moving planet segment.
3. **Sun collision approximation**
   - Uses endpoint-in-sun check only.
   - Official path uses segment-to-sun distance (`point_to_segment_distance`) which can detect crossing between endpoints.
4. **Launch/action model simplification**
   - Fixed action schema with at most one launch per player per env step.
   - Official env accepts list of moves per player per step.
5. **Fleet capacity truncation model**
   - Uses fixed `max_fleets` tensor capacity and first-empty insertion.
   - Official env has dynamic Python list.
6. **Numeric precision and storage**
   - JAX core primarily uses `float32`; official path uses Python float behavior.
7. **Reset path host-side generation**
   - Planet generation/reset uses Python loops and host logic before tensors are created.

These choices are intentional for high-throughput JIT/vectorized execution.

## API

- `JAXOrbitWarsConfig`
- `default_config(num_envs=..., num_players=...)`
- `reset(key, cfg) -> (state, obs)`
- `observe(state, cfg) -> obs`
- `step(state, action, cfg) -> (state, obs, reward, done, info)`

## Config fields

- `num_envs`: parallel environments (batch size)
- `num_players`: players per environment (2 or 4)
- `max_planets`: static capacity for planet tensors
- `max_fleets`: static capacity for fleet tensors
- `episode_steps`: horizon
- `ship_speed`: max fleet speed

## Action schema

The action is a dictionary of fixed-shape tensors:

- `source_logits`: `[B, Players, MaxPlanets]`
- `angle`: `[B, Players]`
- `send_fraction`: `[B, Players]`, clipped to `[0, 1]`

Semantics:

- Each player picks source planet by argmax over masked logits.
- Fleet direction from `angle`.
- Ships sent = floor(source_ships * send_fraction), minimum 1 when valid.

## Observation schema

`obs` is a dictionary:

- `planets`: `[B, MaxPlanets, 7]`
- `fleets`: `[B, MaxFleets, 6]`
- `done`: `[B]`
- `step`: `[B]`

## Reward and done

- `reward`: shape `[B, Players]`, winner(s) `+1`, non-winner `-1`
- `done`: shape `[B]`, true on timeout or when ≤1 player remains alive

## JIT behavior

`step` is already wrapped with `jax.jit(..., static_argnames=("cfg",))`.

Best practices:

- Keep `cfg` static across training loops to avoid recompilation.
- Keep tensor dtypes/shapes constant.
- Prefer pre-allocated fixed-shape action tensors.
