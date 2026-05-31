# Training Pipeline: `scripts/train.py`

This is the top-level orchestration script that ties together the game engine, neural networks, TD3 emitter, and QDax MAP-Elites framework. It runs indefinitely, checkpointing to disk and syncing to Cloudflare R2 in the background.

---

## Startup Sequence

```
1.  Parse args (--hardware, --no-cloud, --no-pull, --no-push)
2.  Detect JAX devices → build mesh + sharding specs (if multi-device)
3.  Set AWS credentials (always, regardless of push/pull flags)
4.  Start R2SyncThread (if not --no-cloud and not --no-push)
5.  Build Actor/DoubleCritic graphs (nnx.split → graph + params)
6.  Configure hyperparameters for selected hardware
7.  Compute CVT centroids (CPU, k-means on 100k points in 4D)
8.  Build scoring_fn, config, dummy_transition, emitter
9.  Initialize init_policy_params (B random actors via vmap)
10. Initialize Hall-of-Fame buffer (50 random actors)
11. map_elites.init() → initial repertoire + emitter_state
12. Pull latest checkpoint from R2 (if not --no-pull)
13. Restore checkpoint → overwrite repertoire, emitter_state
14. Training loop (runs forever)
```

---

## Cloud Flags

| Flag | Pulls from R2? | Pushes to R2? | Use case |
|------|---------------|---------------|----------|
| *(none)* | Yes | Yes | Normal training |
| `--no-push` | Yes | No | Test locally, continue from cloud checkpoint |
| `--no-pull` | No | Yes | Start fresh, still back up to cloud |
| `--no-cloud` | No | No | Fully local debug |

**Credential setup**: The `R2SyncThread.__init__` was historically the only place that set `AWS_*` environment variables. Since `--no-push` skips R2SyncThread entirely, the cloud pull (`aws s3 ls`) had no credentials. The fix: set credentials immediately after `load_dotenv()`, unconditionally:

```python
_ak = os.environ.get("R2_ACCESS_KEY_ID", "")
if _ak:
    os.environ["AWS_ACCESS_KEY_ID"]     = _ak
    os.environ["AWS_SECRET_ACCESS_KEY"] = _sk
    os.environ["AWS_DEFAULT_REGION"]    = "auto"
```

---

## Dual-Format Self-Play Scoring

Every generation, `env_batch` agents are evaluated (128 per device). The batch is split 50/50:

```
policy_params [B]
       ↓
   ┌───┴───┐
  [B/2]   [B/2]
  run_1v1  run_ffa
```

**1v1 half**: Each of the B/2 agents plays a one-on-one match. The opponent is chosen per-agent:
- 50% chance: opponent drawn from archive with **fitness-proportional** probability (pressure to beat the current best)
- 50% chance: opponent drawn from the **Hall of Fame** (robustness against historical agents)

**FFA half**: Each of the B/2 agents plays a 4-player free-for-all:
- opp1: fitness-proportional from archive
- opp2: uniform-random from archive (diversity pressure)
- opp3: random from Hall of Fame

**Why both formats?** 1v1 gives a cleaner signal about head-to-head performance, but FFA rewards strategies that can survive multi-way conflicts and exploit third-party confrontations. Training exclusively on one format produces agents that are brittle in the other.

### Behavioral Descriptors

Each agent's behavioral descriptor is computed as a time-average of four metrics, accumulated only during active (non-terminated) steps:

| BD | Formula | What it measures |
|----|---------|-----------------|
| BD0 | fleet_ships / (fleet_ships + planet_ships) | Aggression: ships deployed vs. hoarded |
| BD1 | my_planets / total_active_planets | Territory fraction |
| BD2 | my_production / (my_planets × 5) | Production efficiency (5 = max prod level) |
| BD3 | my_comet_production / total_comet_production | Comet exploitation |

After the game ends, all accumulators are divided by the number of active steps (clamped to ≥ 1) and clipped to `[0, 1]`.

### Reward Shaping

```python
reward = (score_delta) - my_production
reward = where(done, reward × 100, reward)
```

Raw score delta (`scores[0] - prev_score`) includes both production and combat gains. Subtracting `my_production` removes the steady-state production noise — the reward is now zero when nothing strategically changes, positive when capturing planets, negative when losing ships in combat. The 100× terminal multiplier gives a strong end-of-game signal.

### Transition Subsampling

Inside the scan, every 8th game is used for replay buffer insertion (`[::8]`). With B=128, this gives 16 representative games per step — enough diversity without storing the full batch.

---

## Hall of Fame (HoF)

```
HoF buffer: [50 agent pytrees]
  slots  0..33  — Archive pool (FIFO, 34 slots)
  slots 34..49  — Exploiter pool (FIFO, 16 slots)
```

**Archive snapshots** (every 10 generations):
```python
top_agent = tree_map(lambda x: x[top_idx], repertoire.genotypes)
hof_params = hof_params.at[hof_archive_ptr].set(top_agent)
hof_archive_ptr = (hof_archive_ptr + 1) % HOF_ARCHIVE_SIZE
```

**Exploiter fine-tuning** (every 50 generations): The current TD3 actor is fine-tuned specifically to win against the current archive, then stored in the exploiter pool. The idea is to inject a "known weakness" into the HoF, forcing future agents to learn to handle exploitation attempts.

**`hof_filled`**: A JAX scalar passed to the scoring function as the upper bound for random HoF sampling. Initialized to `HOF_SIZE` (all slots populated with random agents). This avoids the complexity of tracking partial fills — we just overfill with random agents and they get overwritten as real snapshots accumulate.

---

## Multi-Device Sharding

When running on multiple devices (e.g., 2 T4 GPUs or 8 TPU chips), the batch dimension is sharded across all devices using `NamedSharding`:

```python
mesh           = Mesh(np.array(jax.local_devices()), ('batch',))
batch_sharding = NamedSharding(mesh, PartitionSpec('batch'))  # shard on batch dim
rep_sharding   = NamedSharding(mesh, PartitionSpec())         # replicate everywhere
```

**The pattern** for each scoring call:

```python
# Pre-shard inputs before entering JIT
g_s  = device_put(genotypes,            batch_sharding)  # split across devices
rg_s = device_put(rep_genotypes,        rep_sharding)    # replicate to all devices
hof  = device_put(hof_genotypes,        rep_sharding)    # replicate to all devices

# Activate mesh context and call
with jax.set_mesh(mesh):
    fitnesses, descriptors, extra, key = scoring_fn(g_s, rg_s, rf_s, hof, ...)

# Gather all outputs back to device 0
d0 = jax.devices()[0]
fitnesses    = device_put(fitnesses,   d0)
descriptors  = device_put(descriptors, d0)
key          = device_put(key,         d0)
extra_scores = {**extra, 'transitions': device_put(extra['transitions'], d0)}
```

**Why gather everything?** The downstream code (`map_elites.tell()`, `emitter.state_update()`, replay buffer) all live on device 0. Passing sharded arrays into these functions causes shape/sharding conflicts.

**Why not shard the repertoire?** The repertoire has 10,000 cells; accessing arbitrary cells based on fitness sampling indices is a gather operation that doesn't parallelize cleanly across the batch dimension. It's kept replicated — each device has a full copy, so all batch-dimension lookups work locally.

**Checkpoint compatibility**: `NamedSharding` never adds a leading device dimension (unlike `pmap`). Array shapes are identical between 1-device and 8-device runs. An 8-device checkpoint loads correctly on 1 device and vice versa.

---

## Checkpointing

Uses **Orbax** (`ocp.PyTreeCheckpointer`), which serializes arbitrary JAX pytrees to directory-based checkpoints.

**Directory structure** (in `/tmp/checkpoints_v6/` locally, synced to R2):
```
qdax_rep_N/          ← repertoire at generation N
qdax_rep_N_hof/      ← HoF buffer + pointers at generation N
```

**Auto-resume pull** (targeted, not full sync):
```python
# List bucket → find highest qdax_rep_N directory → sync only that + _hof
list_result = subprocess.run(["aws", "s3", "ls", "s3://bucket/v6/", ...])
latest = max(re.findall(r'qdax_rep_(\d+)', list_result.stdout), key=int)
subprocess.run(["aws", "s3", "sync", f"s3://bucket/v6/qdax_rep_{latest}/", ...])
```

Previously, `aws s3 sync` was syncing the entire `/v6/` prefix — potentially 3+ GB of old checkpoints — on every startup. The targeted pull grabs only the single directory that matters, taking seconds instead of minutes.

**R2 Sync**: The `R2SyncThread` daemon thread wakes every 5 minutes and runs `aws s3 sync /tmp/checkpoints_v6/ s3://bucket/v6/`. This uploads new checkpoint directories incrementally (only changed files).

---

## Loop Body (per Generation)

```python
for gen in range(start_gen, ∞):

    # --- Emit candidates ---
    genotypes, extra = map_elites._emitter.emit(repertoire, emitter_state, key)

    # --- Score ---
    if mesh:
        (pre-shard, set_mesh context)
    fitnesses, descriptors, extra_scores, key = scoring_fn(genotypes, ...)
    if mesh:
        (gather to device 0)

    # --- Archive update ---
    repertoire, emitter_state, metrics = map_elites.tell(
        genotypes, fitnesses, descriptors, extra_scores, repertoire, emitter_state)
    if mesh:
        repertoire = device_put(repertoire, d0)

    # --- TD3 training (256 gradient steps) ---
    emitter_state = emitter.state_update(emitter_state, repertoire, ...)
    for _ in range(7):
        emitter_state = emitter.train_only(emitter_state)

    # --- Logging & HoF updates ---
    if gen % 10 == 0:
        (save archive snapshot to HoF)
        (checkpoint to disk)
    if gen % 50 == 0:
        (fine-tune exploiter, save to exploiter pool)

    # --- Memory monitoring (TPU/GPU) ---
    for d in jax.local_devices():
        print(f"{d}: {used:.2f}/{total:.2f} GB ({pct:.1f}%)")
```
