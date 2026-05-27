# Orbit Wars Fast Docs

This docs set explains how to use the `orbit_wars_fast` package end to end.

Recommended reading order:

1. `docs/01_quickstart.md`
2. `docs/02_backend_comparison.md`
3. `docs/03_byte_accuracy.md`
4. `docs/04_benchmarking.md`
5. `docs/05_jax_core.md`
6. `docs/06_rl_integration_evosax_qdax.md`
7. `docs/07_api_reference.md`
8. `docs/08_troubleshooting.md`
9. `docs/10_colab_benchmark.md`
10. `docs/11_parity_acceleration_notes.md`

Main package files:

- `orbit_wars_fast/fast_sim.py` (local byte-accurate simulator)
- `orbit_wars_fast/reference_adapter.py` (official interpreter wrapper)
- `orbit_wars_fast/jax_parity_backend.py` (parity-first backend with JAX-friendly tensors)
- `orbit_wars_fast/test_byte_accuracy.py` (parity verifier, includes jax parity backend)
- `orbit_wars_fast/benchmark.py` (speed benchmark)
- `orbit_wars_fast/jax_core.py` (batched JAX core for RL throughput, not parity)
