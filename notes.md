state format: (planets_array:jnp.Array, num_planets:int, static_groups:int, rng_key:jax.random.PRNGKey, attempts:int)
condition for phase 1: static_groups < MIN_STATIC_GROUPS & attempts < 5000 [ask why bitwise and is used instead of logical]

state format: (planets:jnp.Array, num_planets:int, has_orbiting_planets:bool, rng_key:jax.random.PRNGKey, attempts:int)
condition for phase 2: (needs_more_planets | needs_orbiting_planets) & (attempts < 5000) & (num_planets < max_num_planets)


## Vectorizing Comet Path Generation (Original vs. JAX)

The most significant leap from standard Python to JAX occurs in `generate_comet_paths()`. JAX requires completely static array shapes (no dynamic `.append()`) and forbids standard Python control flow (`if`, `for`) if it depends on data values.

### 1. Re-sampling the Path (Distance Accumulation)

**Original Sequential Code:**
```python
path = [dense[0]]
cum = 0.0
target = comet_speed
for i in range(1, len(dense)):
    cum += distance(dense[i], dense[i - 1])
    if cum >= target:
        path.append(dense[i])
        target += comet_speed
```
*Process:* This code steps through the points sequentially, updating a running total (`cum`), and dynamically expanding a `path` list whenever that total crosses a threshold.

**JAX Vectorized Code:**
```python
dists = distance(dense[1:], dense[:-1])
cumulative_dists = jnp.cumsum(dists)
cumulative_dists = jnp.pad(cumulative_dists, (1,0), constant_values=0.0)

MAX_PATH_LEN = 150 
targets = jnp.arange(MAX_PATH_LEN) * comet_speed
indices = jnp.searchsorted(cumulative_dists, targets) 
safe_idxs = jnp.clip(indices, 0, dense.shape[0] - 1)
path = dense[safe_idxs]
```
*Process Differences:*
1. **Parallel Distance:** Distances between all 5000 points are calculated simultaneously by shifting the array by 1 index (`dense[1:]` and `dense[:-1]`).
2. **Cumulative Sum:** The running total `cum` is replaced by `jnp.cumsum()`. This builds the entire cumulative timeline in parallel.
3. **Static Allocation:** Because JAX arrays cannot grow dynamically, we *must* allocate a fixed size (`MAX_PATH_LEN = 150`).
4. **Binary Search Mapping:** `jnp.searchsorted` replaces the `if cum >= target` check. It takes our fixed array of target thresholds (`[0, 4, 8, 12...]`) and uses binary search to immediately map each threshold to the correct index in `cumulative_dists`.

### 2. Clipping the Path to the Board

**Original Sequential Code:**
```python
board_start = None
board_end = None
for i, (x, y) in enumerate(path):
    if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
        if board_start is None:
            board_start = i
        board_end = i

if board_start is None: continue
visible = path[board_start : board_end + 1]
if not (5 <= len(visible) <= 40): continue
```
*Process:* Iterates through the dynamically sized path array to log the first and last indices that land inside the 100x100 board, then uses Python's `continue` keyword to reject paths that are too long or too short.

**JAX Vectorized Code:**
```python
valid_dist_mask = targets <= cumulative_dists[-1]
on_board_mask = (path[:, 0] >= 0.0) & (path[:, 0] <= BOARD_SIZE) & \
                (path[:, 1] >= 0.0) & (path[:, 1] <= BOARD_SIZE)
valid_mask = valid_dist_mask & on_board_mask

board_start = jnp.argmax(valid_mask)
board_end = MAX_PATH_LEN - 1 - jnp.argmax(valid_mask[::-1])
visible_len = jnp.where(jnp.any(valid_mask), board_end - board_start + 1, 0)

is_valid_comet = (visible_len >= 5) & (visible_len <= 40)
```
*Process Differences:*
1. **Boolean Masking:** Replaces the element-wise `if` checks. Logical comparisons are applied across the entire fixed-size `path` array, resulting in a 150-element array of `True`/`False` values.
2. **Data Padding Logic:** Since `path` is padded out to 150 targets, `valid_dist_mask` is required to "turn off" the padding targets that exceeded the arc's true length.
3. **Argmax for Boundaries:** A clever JAX trick: `jnp.argmax` on a boolean array returns the index of the *first* `True` value. Running it normally gives the `board_start`. Reversing the array (`[::-1]`) and running it again calculates the `board_end`.
4. **No Early Returns:** Instead of `continue`, JAX sets a stateful flag (`is_valid_comet`). A `lax.while_loop` (implemented around this body) evaluates this flag to decide if it should generate a new comet or accept this one.

### 3. Symmetric Mapping (Path Generation)

**Original Sequential Code:**
```python
paths = [
    [[y, x] for x, y in visible],
    [[BOARD_SIZE - x, y] for x, y in visible],
    [[x, BOARD_SIZE - y] for x, y in visible],
    [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
]
```
*Process:* List comprehensions map over the dynamically sized `visible` array to create the 4 mirrored/rotated copies.

**JAX Vectorized Code:**
```python
all_paths = jnp.stack([
    jnp.stack((path[:, 1], path[:, 0]), axis=-1),
    jnp.stack((BOARD_SIZE - path[:, 0], path[:, 1]), axis=-1),
    jnp.stack((path[:, 0], BOARD_SIZE - path[:, 1]), axis=-1),
    jnp.stack((BOARD_SIZE - path[:, 1], BOARD_SIZE - path[:, 0]), axis=-1),
], axis=0)
```
*Process Differences:* Standard JAX array slicing and stacking. It runs the symmetries across the *entire* padded 150-element array, keeping the shape perfectly static at `[4, 150, 2]`.

### 4. Vectorized Collision Detection (Broadcasting)

**Original Sequential Code:**
```python
for k, (cx, cy) in enumerate(visible):
    for planet in orbiting_planets:
        # Calculate future planet position ...
        for sp in sym_pts:
            if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                valid = False
                break
```
*Process:* Triple-nested `for` loop testing each tick `k`, against each `planet`, against each of the 4 symmetric quadrant copies (`sym_pts`). Uses `break` to escape early upon a failure.

**JAX Vectorized Code:**
```python
# dists_to_orbiting calculates distances for [N_planets, 4_paths, 150_points]
dists_to_orbiting = distance(orbiting_pos[:, None, :, :], all_paths[None, :, :, :])
orb_collision = jnp.any(valid_target[:, None, None] & is_orbiting[:, None, None] & valid_mask[None, None, :] & (dists_to_orbiting < req_dist_orb))
```
*Process Differences:*
1. **Dimension Broadcasting (`None`)**: By reshaping the array to `[N, 1, 150, 2]` and subtracting `[1, 4, 150, 2]`, JAX perfectly maps out a 3D grid of distances (`[N, 4, 150]`) calculating *every possible point in time for every planet against every path simultaneously* on the GPU.
2. **Parallel Boolean Reduction**: Instead of `if` checks and `break` statements, JAX constructs a 3D truth table using our logical masks (e.g. `valid_mask` removes padded distances, `valid_target` ignores deactivated planets) and runs a single `jnp.any()` across the entire grid to return a single `True`/`False` collision flag.


## Environment State and Architecture in JAX

Moving from a stateful, Pythonic environment to JAX requires strict functional paradigms and pre-allocation strategies.

### 1. The `EnvState` PyTree

**Original Sequential Code:**
```python
def interpreter(state, env):
    obs0 = state[0].observation
    
    # Initialization buried inside the step function
    if not hasattr(obs0, "planets") or not obs0.planets:
        obs0.planets = generate_planets(init_rng)
        return state
        
    # Normal game step logic follows...
```
*Process:* A monolithic `interpreter` function handles both the initialization and the frame-by-frame stepping, mutating the state in-place using standard Python objects.

**JAX Vectorized Code:**
```python
class EnvState(NamedTuple):
    planets: jnp.ndarray
    step: jnp.ndarray
    # ...

def setup(rng_key: jrandom.PRNGKey) -> EnvState:
    # Pure initialization logic
    return EnvState(planets=..., step=jnp.array(0))
    
def step(state: EnvState, actions) -> EnvState:
    # Pure step logic taking an EnvState and returning a new EnvState
```
*Process Differences:* 
1. **Separation of Concerns:** Initialization is pulled out into a pure `setup()` function, removing the "Is this turn 0?" branch from the core game loop.
2. **PyTree Structures:** The state is packed into a `typing.NamedTuple`. JAX natively understands this "PyTree", automatically flattening the properties into raw tensors for the GPU to compile, and unflattening them back for readability.
3. **Instant Vectorization:** This strict separation makes it trivial to run 10,000 parallel games by simply passing an array of random keys into `jax.vmap(setup)`.

### 2. Joint Planet and Comet Array

**Original Sequential Code:**
```python
# Sometime during the game loop when a comet spawns
planet = [pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION]
obs0.planets.append(planet)
```
*Process:* Comets are treated as a dynamic list that is appended onto the board state mid-game when needed.

**JAX Vectorized Code:**
```python
# Pre-allocated entirely during setup()
base_planets = generate_planets(k2)               # Shape: [40, 7]
comet_tail = jnp.full((MAX_COMETS, 7), -1.0)      # Shape: [4, 7]
planets = jnp.concatenate([base_planets, comet_tail], axis=0) # Shape: [44, 7]
```
*Process Differences:* 
1. **Static Padding:** Since dynamic list appending is banned in JAX, we allocate a fixed static array of size `[44, 7]`. Indices `0-39` hold the base planets, while indices `40-43` act as a buffer for the maximum 4 comets (padded with `-1.0` to hide them until spawn).
2. **Unified Kernels:** This grants massive vectorization benefits. Fleet movement, collision detection, and ship production logic only needs to be written *once*. The XLA compiler runs the operations over all 44 slots simultaneously.

### 3. Parallel Boolean Masks

**Original Sequential Code:**
```python
static_planets = []
orbiting_planets = []
for planet in initial_planets:
    pr = distance((planet[2], planet[3]), (CENTER, CENTER))
    if pr + planet[4] < ROTATION_RADIUS_LIMIT:
        orbiting_planets.append(planet)
    else:
        static_planets.append(planet)
```
*Process:* Iterates through planets on the fly and does the math to sort them into newly allocated sub-lists for movement or collision checks.

**JAX Vectorized Code:**
```python
# Calculated exactly once during setup()
pr = distance(planets[:, 2:4], jnp.array([CENTER, CENTER]))
is_orbiting_base = (pr + planets[:, 4]) < ROTATION_RADIUS_LIMIT

is_orbiting = is_orbiting_base & ~is_comet & (planets[:, 0] != -1.0)
is_static = ~is_orbiting_base & ~is_comet & (planets[:, 0] != -1.0)
```
*Process Differences:*
1. **Native Branching:** JAX forbids data-dependent `if` statements. Having parallel arrays of `True`/`False` values natively plugs into functions like `jnp.where(is_orbiting, new_pos, old_pos)`.
2. **Zero Re-computation:** Calculating if a planet is orbiting requires square roots and distance checks. By evaluating this exactly once during `setup()` and saving the resulting boolean mask into the `EnvState`, we save the GPU from doing redundant math every single game tick!
3. **Type Uniformity:** Using parallel boolean masks keeps conceptual metadata out of the pure floating-point `planets` matrices.


## Mental Framework for the `step()` Function

**The Golden Rule of JAX:** *Calculate everything for everyone simultaneously, then use boolean masks to pick the right answers.*

1. **Advance Time:** Increment `step`. For array indices like `comet_step`, use `jnp.clip` to prevent out-of-bounds errors.
2. **Compute All Realities:** Instead of branching, calculate 3 full `(44, 2)` arrays as if *all* slots are static, *all* are orbiting, and *all* are comets simultaneously.
3. **Blend with Masks:** Use `jnp.where` like a vectorized `if/elif/else` to stitch the correct coordinates together using your boolean masks.
4. **Despawn via Masks:** Don't delete items from lists. Create a boolean mask for "dead" items and use `jnp.where` to overwrite their row with `-1.0`.
5. **Immutable Returns:** `EnvState` cannot be mutated. Pack updated arrays into a new state using `state._replace(...)`.

> 🚨 **CRITICAL REMINDER: THE OOB COMET STATE LEAK** 🚨
> **DO NOT** just return your updated coordinates as-is at the end of the `step` function! 
> Because our path arrays are padded to 150 steps, comets that finish their visible arc will step into actual out-of-bounds (OOB) deep-space coordinates. If you return this raw state, **the agents will receive observations of planets existing outside the map.**
> 
> **The Fix:** Right before returning your final `EnvState`, you **MUST** apply your `oob_bodies` mask to wipe expired comets:
> * Force their `planet_coords` to `-99.0`
> * Force their `planet_owners` to `-1`
> 
> This turns them back into deactivated "ghosts" that the Kaggle adapter will naturally ignore, perfectly mimicking the original Python list deletion!

### 6. The Pre-computation Principle (Comet Spawning)
**The Problem:** In an RL setting, multiple batched environments will be on different `step` counts. If we put the heavy 300-iteration `generate_comet_paths` function inside `step()` using `lax.cond(step == 50)`, JAX's `vmap` will convert it into a `select` operation. This forces the GPU to execute the massive comet generation loop on *every single game tick*, absolutely tanking performance.

**The Solution:** Comets only collision-check against the Sun, Static planets, and Orbiting planets. Because planetary orbits are deterministic, comets do not depend on the dynamic game state (fleets, ships, ownership). 
Therefore, **all 5 waves of comets can be pre-computed during `setup()`!**

* **In `setup()`:** 
  Use `jax.vmap` over the `COMET_SPAWN_STEPS` array to generate an `all_comet_paths` tensor of shape `[5, 4, 150, 2]`. Store this in `EnvState`.
* **In `step()`:** 
  When `step` hits a spawn milestone, simply copy the pre-computed coordinates into the `planets` buffer, set the comet ships/IDs, and reset the `comet_step_index` to 0. No heavy generation logic runs during the game loop!

### 7. Unified Stateless Tracking (The `spawn_steps` Array)
Instead of treating base planets and comets as two completely different entities with different time-tracking logic, we can unify them using a single `spawn_steps` array of shape `[60]`.

1. **Initialization (`setup`):**
   * Base planets (indices 0-39) are assigned a spawn step of `0`.
   * Comets (indices 40-59) are assigned their respective spawn steps (`50, 150, ...`).
   * If a comet wave fails to generate, its slots are assigned a sentinel value of `-1`, **and `is_comet` is explicitly set to `False` for those slots.**
2. **The Unified Mask (`step`):**
   * Because `is_comet` only tracks successful comets, you can determine if a comet should be on the board with a single broadcasted check: `is_active_comet = state.is_comet & (step >= state.spawn_steps)`.
   * To check expiration for comets: `is_expired = state.is_comet & ((step - state.spawn_steps) >= 150)`.
3. **Why this is brilliant:** It completely eliminates the need to cross-reference the global `step` against a separate 5-element array. Every single slot in the 60-element tensor knows exactly when it should exist!


## 8. Inference Adapter Strategy (Deploying to Kaggle)

When training is complete, the JAX RL agent must be deployed back into the standard Python Kaggle environment. To minimize translation overhead and avoid hitting the 1-second `actTimeout`, the adapter should use **pre-allocated NumPy buffers** and **direct index mapping**.

### The Translation Pipeline:
1. **Pre-allocate:** The Kaggle agent class should initialize empty `np.full(...)` buffers for planets and fleets on Turn 0.
2. **Observation to Array (O(N)):** 
   * Wipe the buffers to `-1.0`.
   * Loop through `obs.planets`.
   * Drop each planet directly into its ID slot: `planet_buffer[p.id] = [p.owner, p.y, p.x, ...]`.
3. **Network Inference:** Pass the NumPy buffers to the trained policy network to get an `EnvAction` PyTree (containing parallel `ships` and `angle` arrays of shape `[60]`). *(Note: Using an exported ONNX or TFLite model, or standard NumPy/Flax inference is recommended. Avoid JIT-compiling JAX functions on step 0 in Kaggle, as the cold-start compilation time might exceed the `agentTimeout`!)*
4. **Array to Action (O(N)):**
   * Loop through the network's output arrays.
   * If the output dictates a launch at index `i` (e.g., `ships[i] > 0`):
   * Append `[i, angle[i], ships[i]]` directly to the Kaggle moves list.

### Why this is fast:
Because the JAX environment was designed so that **Array Index == Planet ID**, the adapter never has to dynamically resolve mapping dictionaries or search lists. Translation is purely sequential memory assignment, taking only microseconds.



---

### Reference Code for Phase 1: Unified Stateless Movement & Despawning

*(Use this as a guide for how to structure JAX array manipulation when you build your `step` function)*

```python
# 1. Advance Time
next_step = state.step + 1

# Calculate the age/index of every slot dynamically using the unified array
ages = next_step - state.spawn_steps
safe_path_indices = jnp.clip(ages, 0, 149)

# 2. Calculate Potential Positions
# Static coords
static_coords = state.initial_planets[:, 2:4]

# Orbit coords (recalculated from precomputed initial parameters to avoid float drift and expensive math)
current_angles = params.planet_initial_angles + (params.angular_velocity * next_step)
orbit_x = CENTER + params.planet_orbital_radii * jnp.cos(current_angles)
orbit_y = CENTER + params.planet_orbital_radii * jnp.sin(current_angles)
orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1)

# Comet coords (Extract directly from all_comet_paths using the safe indices)
padded_comet_coords = jnp.zeros((60, 2))
active_coords = state.all_comet_paths[jnp.arange(20), safe_path_indices[-20:], :]
padded_comet_coords = padded_comet_coords.at[-20:].set(active_coords)

# 3. Blend Positions
new_coords = jnp.where(
    state.is_orbiting[:, None], orbit_coords,
    jnp.where(
        state.is_comet[:, None], padded_comet_coords,
        static_coords
    )
)
updated_planets = state.planets.at[:, 2:4].set(new_coords)

# 4. Despawn Comets
comets_oob = (new_coords[:, 0] < 0.0) | (new_coords[:, 0] > BOARD_SIZE) | \
             (new_coords[:, 1] < 0.0) | (new_coords[:, 1] > BOARD_SIZE)
             
despawn_mask = state.is_comet & (comets_oob | (ages >= 150))

updated_planets = jnp.where(
    despawn_mask[:, None], 
    -1.0, 
    updated_planets
)
```

---

## Implementation Checklist: Completing `step()` for Logical Parity

**What's already done in `step()`:**
- `step` counter increment
- `ages` / `safe_comet_ages` computation from `params.comet_spawn_steps`
- `active_comets` and `active_bodies` masks
- Comet coordinate update (path indexing + lifespan guard)
- Orbiting planet coordinate update → `body_coord_update`

**Execution order for the remaining phases:**

```
[A] Add ship_speed to EnvParams      (prerequisite)
[B] Fleet Launch                     (EnvAction → fleet buffer)
[C] Production                       (owned planets gain ships)
[D] Fleet Movement                   (advance all active fleets)
[E] Collision Detection              (planet hits, sun, OOB)
[F] Combat Resolution                (landed fleets resolve)
[G] OOB Comet Despawn Fix            (apply active_bodies mask — CRITICAL)
[H] Termination & Scoring            (game-over check)
[I] Assemble new EnvState            (return all updated arrays)
```

---

### [A] Add `ship_speed` to `EnvParams`

1. Add field `ship_speed: Float[Array, ""]` to the `EnvParams` NamedTuple.
2. In `setup()`, add `ship_speed=jnp.array(6.0)` to the `EnvParams(...)` constructor (6.0 is the original default `configuration.shipSpeed`).

---

### [B] Fleet Launch

**Goal:** Translate `EnvAction[MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP]` into new fleet entries in the fleet buffer.

`EnvAction` is indexed by planet slot (not player). The implicit launching player for planet slot `i` is `state.planet_owners[i]`. All validation flows from ownership.

#### B.1 — Flatten the action arrays

```python
# Shape: [MAX_MOVES_PER_STEP] = [MAX_BODIES * MAX_FLEETS_PER_PLANET_PER_STEP]
flat_ships  = actions.ships.reshape(-1)
flat_angles = actions.angle.reshape(-1)
# Which planet does each flat slot belong to?
planet_idx  = jnp.arange(MAX_MOVES_PER_STEP) // MAX_FLEETS_PER_PLANET_PER_STEP  # [MAX_MOVES_PER_STEP]
```

#### B.2 — Compute per-planet total ships requested

Only count slots where `ships > 0`:
```python
ships_requested = jnp.sum(jnp.where(actions.ships > 0, actions.ships, 0), axis=1)  # [MAX_BODIES]
```

#### B.3 — Determine which planets can afford their launches

```python
planet_can_launch = (
    active_bodies &                                 # planet exists in the game
    (state.planet_owners != -1) &                   # planet is owned
    (state.planet_ships >= ships_requested)         # garrison covers total requested
)
```

#### B.4 — Final per-slot validity mask

```python
valid = (flat_ships > 0) & planet_can_launch[planet_idx]   # [MAX_MOVES_PER_STEP]
```

#### B.5 — Deduct ships from planet garrisons

```python
actual_deducted = jnp.sum(
    jnp.where(actions.ships > 0, actions.ships, 0) * planet_can_launch[:, None].astype(jnp.int32),
    axis=1
)  # [MAX_BODIES]
new_planet_ships = state.planet_ships - actual_deducted
```

#### B.6 — Compute fleet start positions

Fleets spawn just outside the planet's radius so they don't immediately collide with their origin. Use `state.planet_coords` (start-of-tick positions, not `body_coord_update`):

```python
start_x = state.planet_coords[planet_idx, 0] + jnp.cos(flat_angles) * (params.planet_radii[planet_idx] + 0.1)
start_y = state.planet_coords[planet_idx, 1] + jnp.sin(flat_angles) * (params.planet_radii[planet_idx] + 0.1)
flat_start_coords = jnp.stack([start_x, start_y], axis=-1)  # [MAX_MOVES_PER_STEP, 2]
flat_owners = state.planet_owners[planet_idx]                # [MAX_MOVES_PER_STEP]
```

#### B.7 — Insert into the fleet buffer (rank-and-select pattern, no `.at[].set()`)

Rank valid launches:
```python
launch_rank = jnp.cumsum(valid) - 1   # [MAX_MOVES_PER_STEP]; -1 for invalid slots, 0..N-1 for valid
n_valid     = jnp.sum(valid)
```

Build rank-indexed arrays of size `[MAX_MOVES_PER_STEP]` where position `r` holds the data for the `r`-th valid launch. Use a `[MAX_MOVES_PER_STEP, MAX_MOVES_PER_STEP]` selector matrix (4MB at worst, compile-time static):

```python
rank_targets = jnp.arange(MAX_MOVES_PER_STEP)
selector     = (launch_rank[:, None] == rank_targets[None, :]) & valid[:, None]  # [MAX_MOVES_PER_STEP, MAX_MOVES_PER_STEP]

rank_ships  = jnp.sum(flat_ships[:, None]                 * selector, axis=0)
rank_angles = jnp.sum(flat_angles[:, None]                * selector, axis=0)
rank_owners = jnp.sum(flat_owners.astype(float)[:, None]  * selector, axis=0).astype(jnp.int32)
rank_coords = jnp.sum(flat_start_coords[:, None, :] * selector[:, :, None], axis=0)  # [MAX_MOVES_PER_STEP, 2]
```

Assign each empty fleet slot its launch by matching slot rank to launch rank:
```python
empty_mask      = (state.fleet_owners == -1)                        # [MAX_FLEETS]
slot_rank       = jnp.cumsum(empty_mask) - 1                        # [MAX_FLEETS]
should_fill     = empty_mask & (slot_rank < n_valid)                # [MAX_FLEETS]
safe_slot_rank  = jnp.clip(slot_rank, 0, MAX_MOVES_PER_STEP - 1)   # prevent OOB on gather

new_fleet_owners      = jnp.where(should_fill, rank_owners[safe_slot_rank],         state.fleet_owners)
new_fleet_ship_count  = jnp.where(should_fill, rank_ships[safe_slot_rank],          state.fleet_ship_count)
new_fleet_angles      = jnp.where(should_fill, rank_angles[safe_slot_rank],         state.fleet_angles)
new_fleet_coords      = jnp.where(should_fill[:, None], rank_coords[safe_slot_rank], state.fleet_coords)
```

---

### [C] Production

Only active, owned planets produce ships. Apply after fleet launch so deducted ships are not re-added.

```python
prod_delta       = jnp.where((state.planet_owners != -1) & active_bodies, params.planet_prod, 0)
new_planet_ships = new_planet_ships + prod_delta   # use the post-launch planet_ships from B.5
```

---

### [D] Fleet Movement

An active fleet has `fleet_owners != -1`.

#### D.1 — Speed formula (matches original exactly)

```python
active_fleets = (new_fleet_owners != -1)   # [MAX_FLEETS]
safe_ships    = jnp.maximum(new_fleet_ship_count, 1)   # prevent log(0)

raw_speed = 1.0 + (params.ship_speed - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
speed     = jnp.minimum(raw_speed, params.ship_speed)  # [MAX_FLEETS]
```

#### D.2 — Advance positions

```python
dx = jnp.cos(new_fleet_angles) * speed
dy = jnp.sin(new_fleet_angles) * speed
moved_fleet_coords = new_fleet_coords + jnp.stack([dx, dy], axis=-1)  # [MAX_FLEETS, 2]
```

---

### [E] Collision Detection

Compute three removal masks simultaneously, then combine.

#### E.1 — Fleet-planet collision

Check the fleet's **new** position against each planet's **new** position (i.e., use `body_coord_update`, not `state.planet_coords`). This is a per-step distance check (not swept), which is the intended JAX approach.

```python
# dists: [MAX_FLEETS, MAX_BODIES]
dists = distance(
    moved_fleet_coords[:, None, :],   # [MAX_FLEETS, 1, 2]
    body_coord_update[None, :, :]     # [1, MAX_BODIES, 2]
)
within_radius = dists < params.planet_radii[None, :]   # [MAX_FLEETS, MAX_BODIES]
# Mask to only valid fleet-body pairs
hit_matrix = within_radius & active_fleets[:, None] & active_bodies[None, :]  # [MAX_FLEETS, MAX_BODIES]

# A fleet "lands" on the first planet it enters; for removal we only need to know it hit *something*
fleet_hit_planet = jnp.any(hit_matrix, axis=1)   # [MAX_FLEETS]
```

`hit_matrix` is also needed for combat resolution in [F].

#### E.2 — Fleet-sun collision

```python
sun_dist         = distance(moved_fleet_coords, jnp.array([CENTER, CENTER]))  # [MAX_FLEETS]
fleet_hit_sun    = active_fleets & (sun_dist < SUN_RADIUS)
```

#### E.3 — Fleet out-of-bounds

```python
x, y = moved_fleet_coords[:, 0], moved_fleet_coords[:, 1]
fleet_oob = active_fleets & ~((x >= 0.0) & (x <= BOARD_SIZE) & (y >= 0.0) & (y <= BOARD_SIZE))
```

#### E.4 — Combined removal mask

Planets take priority over sun/OOB so that fast fleets still register a hit before leaving the board (same priority as original):

```python
fleet_removed  = fleet_hit_planet | fleet_hit_sun | fleet_oob    # [MAX_FLEETS]
surviving_fleets = active_fleets & ~fleet_removed
```

Final fleet state after movement (dead fleets get owner=-1):
```python
final_fleet_owners     = jnp.where(surviving_fleets, new_fleet_owners,     -1)
final_fleet_ship_count = jnp.where(surviving_fleets, new_fleet_ship_count, 0)
final_fleet_angles     = jnp.where(surviving_fleets, new_fleet_angles,     0.0)
final_fleet_coords     = jnp.where(surviving_fleets[:, None], moved_fleet_coords, 0.0)
```

---

### [F] Combat Resolution

Only fleets that hit a planet (`fleet_hit_planet`) participate. Use `hit_matrix` from E.1 (restricted to planet-hitting fleets only).

```python
landing_hit_matrix = hit_matrix & fleet_hit_planet[:, None]   # [MAX_FLEETS, MAX_BODIES]
```

#### F.1 — Aggregate arriving ships per planet per player

Build `ships_by_player[MAX_BODIES, AGENT_COUNT]` using `jnp.where`-based masking, looping over the 4 compile-time-constant player IDs:

```python
def ships_for_player(p):
    # [MAX_FLEETS, MAX_BODIES] mask: fleet f hits planet b and is owned by player p
    hit_by_p = landing_hit_matrix & (final_fleet_owners[:, None] == p)
    return jnp.sum(
        jnp.where(hit_by_p, final_fleet_ship_count[:, None], 0),
        axis=0
    )  # [MAX_BODIES]

ships_by_player = jax.vmap(ships_for_player)(jnp.arange(AGENT_COUNT)).T   # [MAX_BODIES, AGENT_COUNT]
```

#### F.2 — Determine combat outcome per planet

```python
sorted_ships  = jnp.sort(ships_by_player, axis=-1)[:, ::-1]  # descending per planet
top_ships     = sorted_ships[:, 0]                            # [MAX_BODIES]
second_ships  = sorted_ships[:, 1]                            # [MAX_BODIES]

# If tied, mutual destruction — no survivors
survivor_ships  = jnp.where(top_ships == second_ships, 0, top_ships - second_ships)
top_player      = jnp.argmax(ships_by_player, axis=-1)        # [MAX_BODIES]
survivor_owner  = jnp.where(survivor_ships > 0, top_player, jnp.full(MAX_BODIES, -1))
```

#### F.3 — Apply to planetary garrisons

```python
has_combat      = jnp.any(landing_hit_matrix, axis=0) & active_bodies   # [MAX_BODIES]
is_friendly     = has_combat & (survivor_owner == state.planet_owners) & (survivor_ships > 0)
is_hostile      = has_combat & (survivor_owner != state.planet_owners) & (survivor_ships > 0)

# Reinforcement: add surviving ships to existing garrison
post_reinforce  = new_planet_ships + jnp.where(is_friendly, survivor_ships, 0)

# Assault: subtract ships; if result goes negative, planet is captured and ships flip sign
post_assault    = new_planet_ships - jnp.where(is_hostile, survivor_ships, 0)
captured        = is_hostile & (post_assault < 0)

new_planet_ships_combat = jnp.where(
    is_friendly, post_reinforce,
    jnp.where(is_hostile, jnp.abs(post_assault), new_planet_ships)
)
new_planet_owners_combat = jnp.where(captured, survivor_owner, state.planet_owners)
```

---

### [G] OOB Comet Despawn Fix ⚠️ CRITICAL

Because `params.comet_paths` is padded to `MAX_COMET_PATH_LEN`, comets that have expired still have valid-looking path coordinates from their last alive position. If `active_bodies` is not applied before returning, agents receive observations of planets at stale on-board positions.

Apply `active_bodies` to force inactive bodies off-map and to neutral:
```python
final_planet_coords = jnp.where(active_bodies[:, None], body_coord_update, jnp.array([-99.0, -99.0]))
final_planet_owners = jnp.where(active_bodies, new_planet_owners_combat, -1)
final_planet_ships  = jnp.where(active_bodies, new_planet_ships_combat, 0)
```

---

### [H] Termination & Scoring

These are returned alongside the state; the exact integration with a training loop is TBD, but the logic should live inside `step()` as computed scalars/arrays.

#### H.1 — Check alive players

A player is "alive" if they own at least one active planet **or** have at least one active fleet:

```python
player_ids        = jnp.arange(AGENT_COUNT)                                         # [AGENT_COUNT]
owns_planet       = jnp.any((final_planet_owners[:, None] == player_ids) & active_bodies[:, None], axis=0)
owns_fleet        = jnp.any(final_fleet_owners[:, None] == player_ids, axis=0)
alive             = owns_planet | owns_fleet                                          # [AGENT_COUNT]
n_alive           = jnp.sum(alive)
```

#### H.2 — Termination condition

```python
MAX_EPISODE_STEPS = 500   # define as a top-level constant (matches original's episodeSteps)
terminated = (n_alive <= 1) | (step >= MAX_EPISODE_STEPS - 2)
```

#### H.3 — Final scores (only meaningful when `terminated`)

Score = garrison ships on owned active planets + ships in active fleets:

```python
planet_score = jnp.array([
    jnp.sum(jnp.where((final_planet_owners == p) & active_bodies, final_planet_ships, 0))
    for p in range(AGENT_COUNT)
])  # [AGENT_COUNT]
fleet_score  = jnp.array([
    jnp.sum(jnp.where(final_fleet_owners == p, final_fleet_ship_count, 0))
    for p in range(AGENT_COUNT)
])  # [AGENT_COUNT]
scores = planet_score + fleet_score  # [AGENT_COUNT]

# Reward: 1 for max score, -1 otherwise (only valid when terminated)
max_score = jnp.max(scores)
rewards   = jnp.where((scores == max_score) & (max_score > 0), 1, -1)   # [AGENT_COUNT]
```

---

### [I] Assemble and Return New `EnvState`

Replace the current stub with the full constructor using all updated arrays from phases above:

```python
new_state = EnvState(
    planet_owners   = final_planet_owners,
    planet_coords   = final_planet_coords,
    planet_ships    = final_planet_ships,
    fleet_ids       = state.fleet_ids,         # unchanged; index IS the ID
    fleet_owners    = final_fleet_owners,
    fleet_coords    = final_fleet_coords,
    fleet_angles    = final_fleet_angles,
    fleet_ship_count= final_fleet_ship_count,
    step            = step,
)
return new_state
```

`terminated`, `rewards`, and `scores` should be returned alongside (or wrapped in a separate `StepOutput` tuple) once you decide on the training-loop interface.