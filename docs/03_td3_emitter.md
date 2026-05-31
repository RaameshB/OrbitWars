# TD3 Emitter: `core/nnx_pgame.py`

This file implements the Policy Gradient part of PGA-ME: a TD3 actor-critic that improves on the best agents in the archive using gradient descent. It plugs into the QDax `Emitter` interface.

---

## Key Abstractions

### `Transition`: The Memory Unit

```python
class Transition(PyTreeNode):
    obs      : (planets [60,7], fleets [256,6])  # current state
    next_obs : (planets [60,7], fleets [256,6])  # state after action
    actions  : [60, 72]                          # actor logits (not ship allocations)
    rewards  : scalar                            # shaped reward signal
    dones    : bool                              # episode terminated?
```

Actions stored are the raw **logits**, not the decoded ship counts. The critic was trained to evaluate logits, so we replay logits. The same logits are passed to both `actor_loss_fn` and the replay.

Note the fleet dimension is `256`, not `1024`. `REPLAY_FLEET_CAP = 256` in `train.py` — this cap reduces the replay buffer's memory footprint by 4× while still covering the observed peak of ~70 active fleets with 3.5× headroom.

### `ReplayBuffer`: Circular Buffer

```
[  slot 0  ][  slot 1  ][  slot 2  ] ... [  slot N-1  ]
                                          ↑
                              current_position (wraps around)
```

`insert()`: writes `batch_size` transitions starting at `current_position`, wrapping around. Uses `.at[indices].set()` — sequential unique indices, so there's no scatter-add non-determinism.

`sample()`: draws `sample_size` random indices uniformly from `[0, current_size)`.

### `NNXPgameEmitterState`: Full Training State

Everything the emitter needs to be serialized in a checkpoint:
```
replay_buffer       ReplayBuffer (32k transitions at ~540MB)
critic_params       DoubleCritic parameters
target_critic_params Polyak-averaged version (EMA of critic_params)
actor_opt_state     Adam optimizer state for actor
critic_opt_state    Adam optimizer state for critic
pg_actor_params     Current TD3 actor (the one being improved by gradients)
target_actor_params Polyak-averaged version (EMA of pg_actor_params)
random_key          PRNG state
steps               total gradient steps taken
```

### `_TrainState`: Compact JIT Boundary State

```python
class _TrainState(PyTreeNode):
    critic_params, target_critic_params
    actor_opt_state, critic_opt_state
    pg_actor_params, target_actor_params
    random_key, steps
    # NOTE: no replay_buffer
```

**The OOM story**: Early implementations passed the full `NNXPgameEmitterState` (including replay buffer) through `jax.lax.cond` or across JIT boundaries. XLA must pre-allocate output buffers for every leaf in the pytree it receives, including the path it's NOT executing in a `cond`. The `actions` field in the replay buffer is `32768 × 60 × 72 × float32 ≈ 540MB`. Passing this through a JIT boundary doubles the peak memory (input + output copies), causing OOMs on TPU HBM.

The fix: extract only the small model+optimizer state into `_TrainState` before entering JIT, and sample the replay buffer eagerly (in Python) before each call. XLA only allocates output buffers for `_TrainState` (~a few MB).

---

## Training Loop

### `emit()`: Generate Candidates

```python
def emit(self, repertoire, emitter_state, random_key):
    # Select N-1 pairs of parents from the archive
    parents1 = repertoire.select(key1, N-1)
    parents2 = repertoire.select(key2, N-1)

    # Isoline variation: mutate along the line between two parents
    mutants = isoline_variation(parents1, parents2, ...)

    # Add the current TD3 actor as the last candidate
    pg_actor = emitter_state.pg_actor_params
    return concat([mutants, pg_actor])   # [B, ...]
```

**Isoline mutation**: given two parent networks P1 and P2, compute:
```
child = P1 + σ_iso × noise + σ_line × (P2 - P1) × u
```
where noise is Gaussian and u is uniform. This moves the child along the P1→P2 line in weight space, plus a small isotropic perturbation. It's a way of exploring the "landscape" between two successful strategies.

### `state_update()`: Insert Experience + Train

```python
def state_update(self, ...):
    # 1. Insert transitions into replay buffer (eager, outside JIT)
    flat_transitions = reshape(extra_scores["transitions"], [-1, ...])
    replay_buffer = replay_buffer.insert(flat_transitions)

    # 2. Python-level warm-up guard
    if int(replay_buffer.current_size) < learning_starts:
        return emitter_state  # don't train yet

    # 3. 32 gradient steps
    return self._run_n_train_steps(emitter_state, 32)
```

The warm-up guard is a Python `if`, not `jax.lax.cond`. This is intentional — `lax.cond` would force XLA to trace both branches, requiring output buffers for both the "train" and "no-train" code paths, which involves the full replay buffer. A Python `if` means only one branch ever runs per Python call.

From `train.py`, `state_update()` is called once (32 steps) and then `train_only()` is called 7 more times (7 × 32 = 224 steps), for 256 total gradient steps per generation.

### `_run_n_train_steps()`: Sample Eagerly, JIT Cheaply

```python
for _ in range(n):
    rng, sample_key = jax.random.split(train_state.random_key)
    batch = rb.sample(sample_key, pg_batch_size)   # ← eager, outside JIT
    train_state = self._jit_train_step(train_state.replace(random_key=rng), batch)
```

Each iteration: sample outside JIT, pass batch into the JIT-compiled step. The replay buffer never enters or exits JIT.

### `_jit_train_step()`: One TD3 Update

```
Input:  _TrainState (small), batch: Transition (128 samples)

Critic update:
  1. Compute target actions:  target_actor(next_obs)
  2. Add clipped noise to target logits:
       logit dims: clip(N(0, 0.2), -0.5, +0.5)
       sin/cos dims: clip(N(0, 0.05), -0.1, +0.1)
     (different scales because logits are unbounded, sin/cos are in [-1,1])
  3. target_Q = min(target_q1, target_q2)(next_obs, noisy_target_actions)
  4. y = reward + γ × target_Q × (1 - done)     ← Bellman target
  5. critic_loss = mean((Q1 - y)² + (Q2 - y)²)
  6. gradient step on critic

Actor update (every step, no delay):
  7. actor_loss = -mean(Q1(obs, actor(obs)))     ← maximize critic's value
  8. gradient step on actor

Polyak averaging (EMA):
  9. target_actor = (1 - τ) × target_actor + τ × actor        τ = 0.005
  10. target_critic = (1 - τ) × target_critic + τ × critic
```

**Why no policy delay?** Standard TD3 updates the actor every 2 critic steps to let the critic stabilize before the actor chases it. In practice here, eliminating this delay removes a `lax.cond` from the XLA graph (which otherwise forced output buffer allocation for both branches of the cond). The performance difference without delay is negligible.

**Target policy noise**: adding noise to target actions in step 2 prevents the critic from exploiting narrow peaks in action space where Q-value is overestimated. The noise is clipped asymmetrically — larger noise for the logit dimensions (wide range), smaller for the sin/cos dimensions (already bounded).

---

## Flax NNX: How Model Parameters Work

The codebase uses Flax's functional (NNX) API, which splits a model into:
- **`actor_graph`**: the structure/topology of the network (static)
- **`params`**: the actual weight arrays (stored as pytrees)

This split allows QDax to treat each agent as a parameter pytree — you can vmap over a batch of different parameter sets and run them all simultaneously:

```python
# Merge graph + params → callable model
actor = nnx.merge(actor_graph, single_params)
logits = actor(planets, fleets)

# vmap version: run B different actors on B different observations
vmap_forward = jax.vmap(lambda p, pl, fl: nnx.merge(actor_graph, p)(pl, fl))
all_logits = vmap_forward(batch_params, batch_planets, batch_fleets)
```

`actor_graph` is defined once at startup and closed over by all the JIT-compiled functions. It never changes.
