# Troubleshooting

## 1) Byte-accuracy mismatch appears

Symptoms:

- `test_byte_accuracy` fails with snapshot mismatch.

What to do:

- Open emitted mismatch file `orbit_wars_fast/mismatch_game_<g>_step_<s>.json`.
- Compare first diverging structures (`planets`, `fleets`, `comets`).
- Re-run with fewer games and lower steps to isolate earliest divergence.

## 2) Benchmark seems inconsistent

Symptoms:

- Large run-to-run variance.

What to do:

- Increase games count for averaging (`--games 30+`).
- Keep machine load low during tests.
- Use same seed and environment when comparing revisions.

## 3) JAX recompilation overhead dominates

Symptoms:

- First few steps are very slow.

What to do:

- Ignore first call for throughput measurements (warm-up compile).
- Keep `cfg`, action shapes, and dtypes static.
- Avoid changing `num_envs`, `max_planets`, `max_fleets` frequently.

## 4) JAX OOM / memory pressure

Symptoms:

- Out-of-memory errors during training.

What to do:

- Reduce `num_envs`.
- Reduce `max_fleets` first, then `max_planets`.
- Prefer `float32` (already used) over larger dtypes.

## 5) Policy learns in JAX core but transfers poorly

Symptoms:

- Good reward in JAX core, weak in byte-accurate sim or Kaggle env.

Reason:

- JAX core currently uses simplified collision handling.

What to do:

- Periodically evaluate candidates in `FastOrbitWarsSimulator`.
- Final evaluation in official Kaggle runtime before submission.

## 6) Parity backend slower than JAX core on GPU

This is expected.

- `JAXParityOrbitWarsSimulator` preserves official interpreter transitions (CPU-bound Python path).
- `jax_core` runs vectorized JAX/XLA kernels (GPU-accelerated), but is approximation-first.

Use parity backends for correctness, `jax_core` for throughput pretraining.

## 7) Import issues from notebooks

If running from notebook, ensure project root is in path and import with module names:

- `from orbit_wars_fast.fast_sim import FastOrbitWarsSimulator`
- `from orbit_wars_fast.jax_core import default_config, reset, step`
