# Quickstart

## 1) Validate byte accuracy

Run:

- `python -m orbit_wars_fast.test_byte_accuracy --games 20 --agents 2 --steps 200 --seed 1337`

What this does:

- Runs the official interpreter and local `FastOrbitWarsSimulator` side by side.
- Uses identical random actions for both.
- Compares canonical JSON snapshots byte-for-byte on every step.

For 4-player mode, use:

- `python -m orbit_wars_fast.test_byte_accuracy --games 20 --agents 4 --steps 200 --seed 1337`

## 2) Benchmark runtime

Run:

- `python -m orbit_wars_fast.benchmark --games 10 --steps 200 --seed 1337`

This prints per-backend total time and milliseconds per step.

## 3) Use the Python simulator directly

Typical flow:

- Construct `FastOrbitWarsSimulator(num_agents=2, seed=...)`
- Call `observations()`
- Build an action list of length `num_agents`
- Step using `step(actions)` until `done`

Action format per player:

- A list of moves
- Each move is `[from_planet_id, angle_radians, ships]`

## 4) Use the JAX core

Typical flow:

- Create config via `default_config(num_envs=..., num_players=...)`
- Call `reset(key, cfg)` to get `(state, obs)`
- Build fixed-shape action tensors
- Step with `step(state, action, cfg)` (JIT-compiled)

See `docs/05_jax_core.md` for exact tensor shapes.
