# Benchmarking Guide

This project includes a runtime benchmark script:

- `orbit_wars_fast/benchmark.py`

It compares:

1. Kaggle full env runtime (`env.run`)
2. Official interpreter adapter (`ReferenceOrbitWarsSimulator`)
3. Local fast simulator (`FastOrbitWarsSimulator`)
4. JAX parity backend (`JAXParityOrbitWarsSimulator`)

## Command

- `python -m orbit_wars_fast.benchmark --games 10 --steps 200 --seed 1337`

Arguments:

- `--games`: number of episodes
- `--steps`: max steps per episode for adapter/fast backends
- `--seed`: base seed

## Reading output

Each backend prints:

- total elapsed time
- total steps executed
- milliseconds per step

It also prints speedups (fast simulator and jax parity backend) relative to Kaggle `env.run`.

## Fairness notes

- Kaggle `env.run` includes full runtime stack overhead and built-in agent execution path.
- Adapter and fast simulator are direct in-process step loops.
- So this benchmark is useful for practical throughput budgeting, not micro-architectural apples-to-apples only.

## Suggested benchmark suite

- Smoke: `--games 3 --steps 100`
- Medium: `--games 20 --steps 300`
- Heavy: `--games 100 --steps 500`
