# Supporting Scripts

---

## `scripts/generate_site.py`: Website Replay Generation

This script is run by the GitHub Actions workflow on every push to `main`. It:
1. Loads the latest checkpoint from `/tmp/checkpoints_v6/`
2. Picks the top agents from the archive
3. Runs a game simulation
4. Converts the JAX trajectory to HTML
5. Writes `public/index.html`

GitHub Pages then serves the `public/` directory as the live site.

**`--players` flag**: Controls the number of players in the displayed game (default 4). Passed from the GitHub Actions workflow as `${{ github.event.client_payload.players || '4' }}`.

---

## `scripts/jax_visualizer.py`: JAX → Kaggle Renderer

The game replay uses the Kaggle planet-wars visualizer (a third-party JS renderer). This script bridges OrbitWars' JAX state format to the Kaggle renderer's expected JSON format.

**Key design decisions**:

### Comets hardcoded to `[]`
```python
"comets": []
```
The Kaggle renderer expects comets in a different format than our JAX game uses internally. Passing real comet data crashed the JavaScript renderer mid-frame. The simplest fix is to omit comets from the visualization — they're visible as neutral planets anyway (they appear and disappear).

### Trajectory format
The visualizer receives a list of `EnvState` objects (one per step) and `EnvParams`. It converts each state to the Kaggle JSON format:
```json
{
    "planets": [{"x": ..., "y": ..., "owner": ..., "ships": ..., "radius": ...}],
    "fleets": [{"x": ..., "y": ..., "owner": ..., "ships": ..., "angle": ...}],
    "comets": []
}
```

Each frame's state is assembled from the JAX arrays (converted to Python scalars), inactive slots (owner == -1) are filtered out, and the result is serialized to JSON embedded in the self-contained HTML file.

---

## `scripts/migrate_r2_compression.py`: One-Shot R2 Compression

A maintenance script run once to compress old R2 checkpoints (stored as raw directories) into `.tar.zst` archives.

**Why**: Orbax saves checkpoints as directories of `.msgpack` files. After many generations, dozens of these accumulate in R2, taking significant space. ZSTD compression at level 22 (maximum) gives modest savings (~1% for already-compact msgpack), but eliminates the overhead of thousands of small objects in the bucket.

**How it works**:
1. List all `qdax_rep_N` directories in the bucket (not `_hof`)
2. Keep the newest 2 uncompressed (so auto-resume still works)
3. For each older directory: download locally → `tar --zstd -cf archive.tar.zst -C tmp name` → upload archive → delete original directory from R2

**Parallelism**: 4 worker threads (I/O bound with a CPU burst for compression). Each thread makes its own boto3 client (boto3 clients are not thread-safe).

**Idempotent**: Checks for the `.tar.zst` key before downloading. Safe to run multiple times.

Note: The current `deploy_ondemand.yml` workflow looks for uncompressed `qdax_rep_N/` directories with `aws s3 ls | grep -E '^qdax_rep_[0-9]+/$'`. If all directories are compressed, the workflow will find nothing. This script is intended for historical cleanup, not ongoing use.

---

## `scripts/verify_behavior.py`: Behavioral Sanity Check

A standalone test script that runs a 500-step 4-player game with **randomly initialized** (untrained) actors and verifies two invariants:

### Invariant 1: Diagonal mask prevents self-sends
```python
diag_ships = sum(diagonal(actions.ships[:, :, :60]))
assert diag_ships == 0   # EXACTLY zero
```
The `_SELF_TARGET_MASK` in `logits_to_action()` zeros out the diagonal of the 60×60 ship allocation matrix. No ships should ever be launched "to yourself".

### Invariant 2: Deep-space bins are active
```python
ds_ships = sum(actions.ships[:, :, 60:])
assert ds_ships > 0   # at least some deep-space launches
```
The actor outputs 4 deep-space bins. The +1.0 bias in the actor keeps them competitive against 60 planet logits. An untrained network should use them at least occasionally.

### Why these specific tests?

Early during development, the "do nothing" behavior (all ships going to diagonal = self) was a common failure mode. The structural +1.0 diagonal bias (in `Actor.__call__`) was added deliberately to encourage "keep ships" as the default action, but the diagonal mask was supposed to convert this to a no-op rather than an actual self-send.

The verify script catches regressions where:
- The mask is accidentally removed → diagonal ships would be non-zero
- The deep-space logit initialization is wrong → DS ships would be zero, network can't explore free-space

Running it: `uv run python scripts/verify_behavior.py`

---

## `.github/workflows/deploy_ondemand.yml`

The CI/CD workflow that regenerates the website. Triggers on:
- Push to `main` (automatic, for any change)
- `workflow_dispatch` (manual from GitHub UI)
- `repository_dispatch` (triggered programmatically, e.g., from a training run)

Steps:
1. Checkout code
2. Install uv + Python 3.12
3. **Pull latest checkpoint**: list R2 bucket → find highest `qdax_rep_N` dir → sync it (+ `_hof` variant)
4. Install Python dependencies
5. Run `generate_site.py --players N`
6. Upload `public/` to GitHub Pages

**No `aws s3 sync` of the whole bucket**: only the single latest directory is pulled. This avoids the multi-GB sync that would time out the workflow.
