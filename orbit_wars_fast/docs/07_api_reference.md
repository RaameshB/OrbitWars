# API Reference

## `orbit_wars_fast.fast_sim`

### `FastOrbitWarsSimulator(num_agents=2, seed=None, use_numba=False)`

Local step simulator with Kaggle-equivalent mechanics used by byte-accuracy tests.

Methods:

- `reset(seed=None) -> observations`
- `observations() -> list[obs_dict]`
- `step(actions) -> (observations, done, reward)`
- `snapshot() -> dict`

Properties:

- `done`

### `random_agent_action(obs, rng)`

Utility random policy for fuzz/parity tests.

## `orbit_wars_fast.reference_adapter`

### `ReferenceOrbitWarsSimulator(num_agents=2, seed=None)`

Thin wrapper over official `orbit_wars.interpreter`.

Methods:

- `reset(seed=None)`
- `observations()`
- `step(actions)`
- `snapshot()`

Properties:

- `done`

## `orbit_wars_fast.jax_parity_backend`

### `JAXParityOrbitWarsSimulator(num_agents=2, seed=None, max_planets=64, max_fleets=1024)`

Parity-first backend for safe training/evaluation, with JAX-friendly tensor observations.

Methods:

- `reset(seed=None)`
- `observations()`
- `observations_jax()`
- `step(actions)`
- `snapshot()`

Properties:

- `done`

## `orbit_wars_fast.jax_core`

### `JAXOrbitWarsConfig`

Dataclass with shape/runtime controls.

### `default_config(num_envs=128, num_players=2)`

Convenience constructor.

### `reset(key, cfg)`

Returns `(state, obs)`.

### `observe(state, cfg)`

Returns observation dictionary.

### `step(state, action, cfg)`

JIT-compiled step.

Returns `(next_state, obs, reward, done, info)`.

## `orbit_wars_fast.rl_adapter`

### `RolloutResult`

Dataclass fields:

- `episode_rewards`
- `steps`
- `done`
- `final_snapshot`

### `flatten_observations_jax(obs_jax)`

Flattens parity backend tensor observation into policy feature matrix.

### `decode_actions(observations, source_logits, angle, send_fraction)`

Converts policy outputs into Orbit Wars action lists.

### `rollout_episode(simulator, policy_apply, params, rng_key, max_steps=None)`

Runs one parity-first episode.

### `evaluate_policy_seeds(make_simulator, policy_apply, params, seeds, base_key, max_steps=500)`

Runs multi-seed evaluation and returns aggregate metrics.

## `orbit_wars_fast.evosax_qdax_adapter`

### `make_parity_simulator_factory(...)`

Builds simulator factory for parity backend.

### `evaluate_candidate_parity(...)`

Evaluate one candidate policy over multiple seeds.

### `evaluate_population_parity(...)`

Evaluate a population (axis-0 pytree) over multiple seeds.

### `make_qdax_scoring_fn(...)`

Returns QDax-compatible scoring function.

## `orbit_wars_fast.parity_pool`

### `run_parallel_rollouts(seeds, num_agents=2, max_steps=500, workers=None, start_method="spawn", use_numba=False)`

Runs parity rollouts in parallel worker processes.

## `orbit_wars_fast.test_byte_accuracy`

CLI entry for parity checks.

## `orbit_wars_fast.benchmark`

CLI and notebook-callable benchmark for backend runtime comparison.

Exports:

- `run_benchmark(...)`
- `print_benchmark(results)`
- `benchmark_dataframe(results)`
- `run_fleet_cap_sweep(...)`
- `recommend_fleet_cap(...)`
