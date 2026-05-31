# JAX Patterns and Gotchas

This document covers JAX-specific patterns and decisions that appear throughout the codebase. Understanding these is essential for maintenance.

---

## Static vs. Traced Values

JAX traces functions to build XLA computation graphs. Any value that affects the *shape* of an array or the *structure* of the computation must be a Python-level constant (static), not a JAX array (traced).

**Good** (static branching on number of players):
```python
@functools.partial(jax.jit, static_argnames=('num_players',))
def step(state, params, actions, num_players=4):
    if num_players == 2:
        ...  # different code path, compiled separately
```

`num_players` is declared static so JAX compiles separate XLA programs for 2-player and 4-player games. This is fine because we only have 2 variants.

**Bad** (dynamic branching on an array):
```python
# WRONG — you can't do this in JIT
if state.step > 50:
    do_something()
```

Use `jnp.where()` for element-wise conditional selection, or `jax.lax.cond()` for full branch selection (but see the OOM warning below).

---

## `lax.while_loop` vs Python Loops

Python loops in JIT-compiled functions get *unrolled* — JAX traces through every iteration at compile time, creating a massive XLA graph. For short fixed-count loops (e.g., 6 attention layers), this is fine. For data-dependent loops (e.g., "keep trying until a valid planet is found"), use `lax.while_loop`.

`lax.while_loop` compiles once. The loop body is re-entered dynamically at runtime.

**Constraint**: all loop-carried state must have fixed shapes and dtypes. You can't append to a list; you pre-allocate and write into fixed slots.

---

## The Replay Buffer JIT Boundary Problem

`jax.lax.cond` — even when only one branch runs — forces XLA to pre-allocate output buffers for **both** branches. If either branch's pytree contains the replay buffer's `actions` field (`[32768, 60, 72, float32] = 540 MB`), XLA allocates that 540 MB even when taking the "no-op" branch.

The same issue arises with plain JIT boundaries: any pytree passed into or out of a JIT function requires output buffer allocation.

**Solution**: Keep the replay buffer entirely outside JIT. Sample batches eagerly in Python, pass only the `_TrainState` (model + optimizer, ~few MB) through JIT. See `_run_n_train_steps()` in `nnx_pgame.py`.

---

## `jnp.where` vs `.at[].set()`

Prefer `jnp.where(mask, new_value, old_value)` for conditional updates. The `.at[].set()` pattern works but has known edge cases with JAX's out-of-bounds behavior and can behave unexpectedly when indices are non-unique or out of bounds.

**Common pattern for updating specific array positions**:
```python
# If you want to set position i to value v:
result = jnp.where(jnp.arange(N) == i, v, original_array)
```

For sorted/packed writes (like fleet launch), argsort + indexed write is acceptable since the indices are guaranteed unique and in-bounds.

---

## Multi-Device Sharding Protocol

When running on multiple devices, follow this exact protocol for the scoring function call:

```python
# 1. Pre-shard inputs BEFORE calling — JAX validates sharding at JIT entry
g_s = jax.device_put(genotypes, batch_sharding)
r_s = jax.device_put(rep_genotypes, rep_sharding)   # replicate

# 2. Activate mesh BEFORE calling
with jax.set_mesh(mesh):
    result = scoring_fn(g_s, r_s, ...)

# 3. Gather ALL outputs to device 0 AFTER calling
d0 = jax.devices()[0]
fitnesses   = jax.device_put(fitnesses, d0)
transitions = jax.device_put(transitions, d0)
# ...etc for every output array
```

**Why pre-shard?** JAX validates that the input sharding matches the `with_sharding_constraint` inside the JIT at the function boundary, before entering the function body. Passing a device-0 array into a function that expects a batch-sharded array raises `ValueError: incompatible devices`.

**Why gather everything?** Functions downstream of scoring (tell, emit, replay buffer insert) all run on device 0. Any sharded array passed into them causes sharding conflicts.

**Why `jax.set_mesh()`, not `with mesh:`?** `with mesh:` is deprecated as of JAX 0.4.x. Use `with jax.set_mesh(mesh):`.

---

## Checkpoint Device Placement

Orbax `PyTreeCheckpointer` saves raw pytree arrays with no device metadata. When restoring:
- Arrays are loaded to the default device (device 0)
- No re-sharding is needed — device 0 is where everything lives anyway
- `jax.device_put(x, rep_sharding)` is NOT needed after restore; `with_sharding_constraint` inside JIT handles placement at call time

**Warning**: Calling `jax.device_put(repertoire, rep_sharding)` outside JIT before `map_elites.tell()` converts the repertoire's arrays to "replicated sharded" objects. When `tell()` then tries to combine them with plain device-0 descriptors, it raises `ValueError: Received incompatible devices`.

---

## lax.scan for Episode Rollouts

500 game steps are run as a single `lax.scan`:
```python
_, (scores_history, bd_history, transitions) = jax.lax.scan(
    scan_step,
    init_carry,
    None,
    length=500
)
```

`lax.scan` is equivalent to a fixed-length for loop that accumulates outputs into a stacked array. It compiles once, runs on-device, and returns all intermediate outputs (the "scan outputs" stacked along axis 0).

The accumulated `transitions` tensor has shape `[500, B//8, ...]` — 500 steps × (B/8) subsampled games × per-transition shape. This gets flattened to `[-1, ...]` before replay buffer insertion.

---

## bfloat16 for Memory-Intensive Operations

The intercept angle computation uses bfloat16 for the distance tensor:
```python
src_bf16 = source_coords.astype(jnp.bfloat16)
tfc_bf16 = future_coords.astype(jnp.bfloat16)
dists = sqrt((tfc_bf16[...,0] - src_bf16[...,0])**2 + ...).astype(float32) + 1e-8
```

bfloat16 uses half the memory of float32 with the same exponent range (just less mantissa precision). For a `[B, 60, 60, 150, 2]` tensor at batch=128, this saves ~500 MB of HBM. The small loss in distance precision (we only need ~1% accuracy to select the right intercept timestep) is acceptable.

---

## CVT Centroids: CPU-Only

```python
centroids = compute_cvt_centroids(num_descriptors=4, num_init_cvt_samples=100000, ...)
```

This uses sklearn's KMeans under the hood — not JAX-native. It runs on CPU and takes several minutes for 10,000 centroids in 4D space. This is computed once at startup, so the overhead is acceptable. The resulting centroid array is then transferred to JAX and used for the rest of training.

There is no JAX-native KMeans implementation stable enough to substitute without risk (low GitHub star count and unclear JAX version compatibility at time of development).
