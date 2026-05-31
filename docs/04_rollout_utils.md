# Rollout Utilities: `core/rollout_utils.py`

This module handles the two most subtle parts of turning network outputs into game actions: building the observation arrays each player sees, and computing intercept angles to actually hit moving targets.

---

## `build_obs_arrays()`: Egocentric Observation

```python
def build_obs_arrays(state, params, pid, num_players):
    # Returns: planets [60,7], fleets [MAX_FLEETS,6], planet_mask [60], fleet_mask [MAX_FLEETS]
```

**Coordinate rotation**: Every player's observation is rotated so that they always appear in the same relative position. Player 0 sees the world as-is; player 1's view is rotated by `-2π/4 = -90°`; player 2 by `-180°`; player 3 by `-270°`.

```
Original world:           Player 2's view (rotated -180°):
P2 ────── P3              P0 ────── P1
 │                         │
 │  (50,50)                │  (50,50)
 │                         │
P0 ────── P1              P2 ────── P3
```

The rotation is a standard 2D rotation around the board center (50,50):
```python
theta = -pid × (2π / num_players)
dx = x - 50;  dy = y - 50
rot_x = dx × cos(theta) - dy × sin(theta) + 50
rot_y = dx × sin(theta) + dy × cos(theta) + 50
```

Fleet angles are shifted by `+theta` (adding the rotation angle to the heading so it remains correct in the rotated frame).

**Relative ownership encoding**:
```
+1.0 = this is MY planet/fleet
 0.0 = neutral (no owner)  [planets] / inactive slot [fleets]
-1.0 = enemy
```

This means the network never sees raw player IDs. It doesn't matter whether the enemy is player 1 or player 3 — they're all `-1.0`. This keeps the input consistent regardless of player assignment.

---

## `active_body_mask()`: Which Slots Are Live?

```python
def active_body_mask(state, params):
    ages = state.step - params.comet_spawn_steps   # [60]
    comet_active = is_comet & (ages >= 0) & (ages < body_lifespans)
    return is_static_planet | is_orbiting_planet | comet_active
```

**Why not use coordinates?** Failed-to-generate planet slots sit at their initial position (wherever the failed attempt left them), not at the off-board sentinel `(-99,-99)`. Using `coord > -50` as an activity check would accidentally include these phantom slots. The flags in `EnvParams` are authoritative.

---

## `calculate_intercept_angle()`: Hitting a Moving Target

This is the most algorithmically complex function in the codebase.

**The problem**: The actor outputs "send ships from planet A to planet B". But planet B may be orbiting, or it may be a comet on a curved path. If we aim directly at where B is now, the fleet will arrive too late and miss. We need to aim at where B *will be* when the fleet arrives.

**The solution**: Simulate the next 150 timesteps of every target's position, compute for each timestep whether the fleet could have traveled far enough to reach the target by then, and pick the first timestep where it can.

```
For each source planet i, target planet j, timestep t:
    future_pos_j(t)   = position of planet j at step (current + t)
    travel_distance_i = R_i + 0.1 + speed_i × t      (fleet spawn offset + travel)
    required_distance = dist(i, future_pos_j(t)) - R_j   (target's edge)

    can_reach(i, j, t) = travel_distance >= required_distance

intercept_t(i, j) = first t where can_reach(i, j, t) is True
angle(i, j)       = arctan2(future_pos_j(intercept_t) - pos_i)
```

### Shape Dimensions

The function operates on batched inputs:
```
[B, 60, 60, 150]  — batch × sources × targets × timesteps
```

This is a lot of computation. The bfloat16 cast for distance computation (`src` and `tfc_bf16`) halves the memory bandwidth for that tensor without meaningfully affecting angle precision (we only need rough angular accuracy).

### Speed Must Match the Engine

```python
safe_ships = jnp.maximum(ships.astype(jnp.int32), 1)   # ← must be int32!
raw_speed = 1.0 + (ship_speed - 1.0) × (log(safe_ships) / log(1000))^1.5
```

The `astype(jnp.int32)` is critical. In `step()`, ship counts are stored as `int32`. The speed formula uses `log(ships)`. At small ship counts (e.g., 7 ships), `log(7.0) ≠ log(7)` in practice because float ships might be `6.99...` due to softmax × current_ships arithmetic, producing a slightly lower log and thus a slightly slower predicted speed.

If the intercept calculator uses float ships, it predicts higher speeds and aims too far ahead, causing systematic misses on orbiting targets.

### Vectorized Future Coordinates

All three body types (orbiting, comet, static) are computed in one expression:

```python
# Orbiting: angle advances each step
current_angles = initial_angle + angular_velocity × future_steps    # [B, 60, 150]
orbit_coords = stack([50 + r × cos(angle), 50 + r × sin(angle)])    # [B, 60, 150, 2]

# Comet: index into precomputed path
safe_comet_ages = clip(comet_ages, 0, 149).astype(int32)
comet_coords = comet_paths[b_idx, c_idx, safe_comet_ages]           # [B, 20, 150, 2]

# Static: just broadcast current position
static_coords = broadcast(planet_coords[:, :, None, :])             # [B, 60, 150, 2]

future_coords = where(is_orbiting, orbit_coords,
                      where(is_comet, padded_comet_coords, static_coords))
```

### Finding the First Reachable Timestep

```python
can_reach = travel_dist >= req_dist                  # [B, 60, 60, 150] bool
# If any timestep is reachable, take the first one; else default to 149
intercept_t_idx = where(any(can_reach), argmax(can_reach), 149)   # [B, 60, 60]
```

`jnp.argmax` on a boolean array returns the index of the first `True` value — a vectorized "find first" operation. No Python loop required.

---

## Usage in the Scoring Function

The scoring function (in `train.py`) calls `calculate_intercept_angle()` for all 4 players every game step, inside a `lax.scan`. This is one of the most expensive operations per step — it's a `[B, 60, 60, 150]` computation.

For the 60 deep-space targets, the actor directly outputs sin/cos angles, which are decoded with `arctan2`. No intercept calculation needed for free-angle launches.

Final action assembly:
```python
intercept_angles = calculate_intercept_angle(state, params, ships[:, :, :60])
ds_angles = arctan2(ds_sin, ds_cos)
angles = concatenate([intercept_angles, ds_angles], axis=-1)   # [B, 60, 64]

# Rotate back to world frame (undo the egocentric rotation)
angles = angles + (pid × 2π / num_players)
```

The world-frame rotation is added back after computing the actor's egocentric angles. The actor always outputs angles in its own rotated frame; the engine expects world-frame angles.
