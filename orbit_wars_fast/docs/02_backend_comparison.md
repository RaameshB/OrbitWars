# Backend Comparison

`orbit_wars_fast` provides multiple backends with different goals.

## A) Kaggle env runtime (`kaggle_environments.make("orbit_wars")`)

Use this when:

- You want a full Kaggle-compatible runtime path.
- You need rendering/replay integration and agent server behavior.

Tradeoff:

- Highest overhead.

## B) Reference adapter (`ReferenceOrbitWarsSimulator`)

File: `orbit_wars_fast/reference_adapter.py`

Use this when:

- You want official mechanics exactly (calls `orbit_wars.interpreter`).
- You need step-level control without full env runtime overhead.

Tradeoff:

- Still Python/interpreter-driven; faster than full Kaggle runtime but not vectorized.

## C) Local fast simulator (`FastOrbitWarsSimulator`)

File: `orbit_wars_fast/fast_sim.py`

Use this when:

- You want local high-speed rollouts with Kaggle-equivalent rules.
- You need byte-accurate state replication checks.

Tradeoff:

- Python loops, so throughput gains are moderate versus reference adapter.

## D) JAX parity backend (`jax_parity_backend.py`)

File: `orbit_wars_fast/jax_parity_backend.py`

Use this when:

- You need parity-first transitions but JAX-friendly tensor observations.
- You want safe training/evaluation behavior relative to Kaggle mechanics.

Tradeoff:

- Transition stepping still goes through official interpreter wrapper.
- Throughput lower than approximation-first JAX core.

## E) JAX vectorized core (`jax_core.py`)

File: `orbit_wars_fast/jax_core.py`

Use this when:

- You need maximum batched rollout throughput for RL/evolution.
- You can tolerate simulator approximation.

Tradeoff:

- Not byte-accurate to Kaggle.
- Uses simplified collision handling for vectorized speed.

## Recommendation matrix

- Strict simulator parity tests: `ReferenceOrbitWarsSimulator` + `FastOrbitWarsSimulator` + `JAXParityOrbitWarsSimulator`.
- Fast local game loop and debugging: `FastOrbitWarsSimulator`.
- Parity-first RL pipelines with JAX tensor IO: `JAXParityOrbitWarsSimulator`.
- Approximation-first high-throughput pretraining: `jax_core.py`.
