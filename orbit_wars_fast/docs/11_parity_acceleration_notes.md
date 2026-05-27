# Parity Acceleration Notes

## Can parity be GPU accelerated with JAX?

Partially, but with strong constraints.

### What *can* be accelerated without parity risk

- Policy network forward pass
- Observation post-processing / feature extraction
- Batched data transforms and logging

### What is hard to accelerate while preserving parity

- Official transition logic (Python interpreter control flow)
- Exact event ordering and tie behavior under all edge cases
- Continuous collision checks in identical operation order

## About tiny ULP differences

If you allow tiny ULP drift but require identical events, you still need:

- Stable operation order
- Consistent branch decisions near boundaries

Near-threshold collisions can flip event outcomes even with tiny numeric changes.

## Why PyTrees do not solve this directly

PyTrees are container abstractions for JAX transformations. They do not make Python list/dict control flow run on GPU.

- JAX can trace/jit array programs.
- Arbitrary Python-side interpreter logic remains CPU/Python.

So PyTrees help organize data, not magically compile non-JAX logic.

## Implemented acceleration options in this repo

1. `FastOrbitWarsSimulator` (NumPy broad-phase)
2. `FastOrbitWarsSimulator(use_numba=True)` (Numba-accelerated narrow-phase collision checks)
3. `run_parallel_rollouts(...)` from `parity_pool.py` (multi-process parity throughput)
4. `run_parallel_rollouts(..., use_numba=True)` (process parallel + numba kernels)

These are benchmarked in `benchmark.py`.

## Practical strategy

1. Keep parity transitions on CPU (`FastOrbitWarsSimulator` / `ReferenceOrbitWarsSimulator` / `JAXParityOrbitWarsSimulator`).
2. Use multiprocessing and/or Numba for parity throughput (`parity_pool.py`, `FastOrbitWarsSimulator(use_numba=True)`).
3. Use GPU for policy inference and approximation-first pretraining.
4. Final selection and fine-tuning on parity backend.
