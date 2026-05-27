# Colab / Notebook Benchmarking

This guide is for running the benchmark interactively in Jupyter or Google Colab (including GPU runtimes).

A ready notebook is included:

- `orbit_wars_fast/colab_benchmark.ipynb`

## 1) Setup in notebook

Install dependencies and import helpers:

- `from orbit_wars_fast.benchmark import run_benchmark, print_benchmark, benchmark_dataframe`

## 2) Run benchmark from a notebook cell

Call:

- `results = run_benchmark(num_games=10, max_steps=200, seed=1337, include_kaggle_runtime=True, jax_num_envs=512, jax_steps=2000, include_numba_variant=True, include_parallel_parity=True, parallel_workers=4, include_jax_toggle_sweep=True, jax_max_fleets=1024)`
- `print_benchmark(results)`

For DataFrame display:

- `df = benchmark_dataframe(results)`
- `display(df)`

## 3) Colab GPU comparison tips

To stress JAX GPU throughput:

- Increase `jax_num_envs` (e.g. 512, 1024, 2048 depending on memory)
- Increase `jax_steps` (e.g. 2000+)
- Keep `include_kaggle_runtime=False` for faster iterative profiling loops

Then run occasional full comparisons with `include_kaggle_runtime=True`.

## 4) Interpreting JAX metrics

Backends you can compare include:

- `fast simulator`
- `fast simulator (numba)`
- `parallel parity (Nw)`
- `parallel parity numba (Nw)`
- `jax core (approx)`

Notes:

- Parallel parity uses process-level CPU scaling and usually benefits from Linux/Colab `fork` start method.
- `parallel_workers` should typically match (or slightly under) CPU core count.
- `jax core (approx)` can still underperform at small batch sizes; increase `jax_num_envs` and `jax_steps` for stable GPU utilization.

JAX toggle variants (when enabled):

Overflow/cap telemetry is also reported for JAX variants:

- `overflow_step_rate`
- `fleet_active_max`
- `fleet_active_max_utilization`

Use these to choose `jax_max_fleets` with a safety margin.

The notebook includes cap utilities:

- `run_fleet_cap_sweep(...)`
- `recommend_fleet_cap(...)`

- `jax core (approx)`
- `jax core (swept)`
- `jax core (segment sun)`
- `jax core (swept+sun)`

These let you isolate perf effects of parity-leaning geometry toggles.

`jax core (approx)` includes:

- `ms_per_step`: latency per JAX step call
- `env_steps_per_s`: aggregate environment steps per second (`num_envs * steps / sec`)
- `speedup_vs_fast_step_latency`: latency ratio vs `FastOrbitWarsSimulator`

Important:

- `jax core (approx)` is not parity-accurate.
- For parity-first behavior use `JAXParityOrbitWarsSimulator` and parity tests.
