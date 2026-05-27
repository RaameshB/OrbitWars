# orbit_wars_fast

Fast, local Orbit Wars simulation tools:

- **Byte-accurate Python simulator** (`fast_sim.py`) against Kaggle interpreter.
- **Reference adapter** (`reference_adapter.py`) that directly calls the official interpreter.
- **JAX parity backend** (`jax_parity_backend.py`) with parity-first transitions and JAX-friendly tensor observations.
- **Benchmark script** (`benchmark.py`) to compare throughput against Kaggle `env.run`.
- **JAX vectorized core** (`jax_core.py`) for approximation-first batched RL rollout throughput.

## Files

- `fast_sim.py`: byte-accurate step simulator
- `reference_adapter.py`: official interpreter adapter
- `test_byte_accuracy.py`: random-policy parity test (snapshot byte compare)
- `benchmark.py`: backend speed comparison (Kaggle env vs reference adapter vs fast sim vs jax parity backend)
- `jax_parity_backend.py`: parity-first backend with JAX tensor observations
- `jax_core.py`: batched JAX environment core (fixed-shape tensors, not parity)

## Run byte-accuracy test

```bash
python -m orbit_wars_fast.test_byte_accuracy --games 20 --agents 2 --steps 200 --seed 1337
```

Use `--agents 4` to validate 4-player mode.

## Run benchmark

```bash
python -m orbit_wars_fast.benchmark --games 10 --steps 200 --seed 1337
```

## JAX core quickstart

```bash
python - <<'PY'
import jax
import jax.numpy as jnp
from orbit_wars_fast.jax_core import default_config, reset, step

cfg = default_config(num_envs=64, num_players=2)
key = jax.random.PRNGKey(0)
state, obs = reset(key, cfg)

# action schema:
# source_logits: [B, Players, MaxPlanets]
# angle:         [B, Players]
# send_fraction: [B, Players]
action = {
    "source_logits": jnp.zeros((cfg.num_envs, cfg.num_players, cfg.max_planets), dtype=jnp.float32),
    "angle": jnp.zeros((cfg.num_envs, cfg.num_players), dtype=jnp.float32),
    "send_fraction": 0.5 * jnp.ones((cfg.num_envs, cfg.num_players), dtype=jnp.float32),
}

state, obs, reward, done, info = step(state, action, cfg)
print(obs["planets"].shape, obs["fleets"].shape, reward.shape, done.shape)
PY
```

### JAX backend guidance

- `jax_parity_backend.py` is the parity-first backend and should be used when simulator fidelity is critical.
- `jax_core.py` is throughput-first and currently **not byte-accurate** to Kaggle (simplified collisions).

Use `test_byte_accuracy.py` to validate parity-sensitive flows.

## Colab / Jupyter benchmark usage

You can call benchmark functions directly from notebook cells:

- `from orbit_wars_fast.benchmark import run_benchmark, print_benchmark, benchmark_dataframe`
- `results = run_benchmark(num_games=10, max_steps=200, seed=1337, include_kaggle_runtime=True, jax_num_envs=512, jax_steps=2000)`
- `print_benchmark(results)`
- `benchmark_dataframe(results)`

This is the recommended way to measure GPU acceleration of `jax_core` in Colab.
A ready notebook is included at `orbit_wars_fast/colab_benchmark.ipynb`.

## Parallel parity rollouts (CPU scaling)

Use `orbit_wars_fast/parity_pool.py` for process-parallel parity rollouts:

- `from orbit_wars_fast.parity_pool import run_parallel_rollouts`
- `run_parallel_rollouts(seeds=list(range(1000, 1100)), num_agents=2, max_steps=500, workers=8)`
- `run_parallel_rollouts(seeds=list(range(1000, 1100)), num_agents=2, max_steps=500, workers=8, use_numba=True)`

This keeps parity behavior and increases throughput via multi-core CPU execution.

## Documentation

Comprehensive docs live under:

- `orbit_wars_fast/docs/00_index.md`

Quick links:

- `orbit_wars_fast/docs/01_quickstart.md`
- `orbit_wars_fast/docs/04_benchmarking.md`
- `orbit_wars_fast/docs/05_jax_core.md`
- `orbit_wars_fast/docs/06_rl_integration_evosax_qdax.md`
- `orbit_wars_fast/docs/07_api_reference.md`

## RL adapter quickstart (parity-first)

Use `orbit_wars_fast/rl_adapter.py` helpers to run policy rollouts on `JAXParityOrbitWarsSimulator`.

Expected policy output keys:

- `source_logits`: `[num_agents, max_planets]`
- `angle`: `[num_agents]`
- `send_fraction`: `[num_agents]`
