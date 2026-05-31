# Game Engine: `core/orbit_wars_jax.py`

The game engine is a fully JAX-compatible, JIT-compiled simulation. Every array has a fixed shape known at compile time — no Python loops, no dynamic allocation, no data-dependent control flow (outside of `lax.while_loop` during map setup, which only runs at environment init time).

---

## The Board

```
(0,0)────────────────────────(100,0)
  │                              │
  │    ┌─────────────┐           │
  │    │ orbiting    │           │
  │    │ zone r<50   │  ●  ←── static planet
  │    │    ●        │           │
  │    │  (orbits)   │ ●         │
  │    │      ╔════╗ │           │
  │    │      ║SUN ║ │           │
  │    │      ╚════╝ │           │
  │    │      r<10   │           │
  │    └─────────────┘           │
  │                              │
(0,100)─────────────────────(100,100)
```

- Board: continuous 100×100 2D space
- Sun: center (50,50), radius 10 — destroys any fleet that enters
- Orbiting zone: radius < 50 — planets inside orbit the sun with a shared angular velocity
- Static zone: radius ≥ 50 — planets are fixed in place

Coordinate system: arrays store `[x, y]` but internally the engine sometimes uses `[y, x]` notation in intermediate planet arrays (fields 2 and 3). The final `EnvState.planet_coords` is always `[x, y]`.

---

## Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_BODIES` | 60 | Total planet+comet slots |
| `MAX_PLANETS_BASE` | 40 | Slots for generated planets |
| `TOTAL_COMETS` | 20 | 4 comets × 5 spawn waves |
| `MAX_FLEETS` | 1024 | Concurrent fleet slots (power of 2) |
| `MAX_EPISODE_STEPS` | 500 | Game length |
| `BOARD_SIZE` | 100.0 | Width and height |
| `SUN_RADIUS` | 10.0 | Fleet death zone |
| `ROTATION_RADIUS_LIMIT` | 50.0 | Boundary between orbiting/static |

---

## State and Params

The engine splits the game state into two NamedTuples:

**`EnvState`** — changes every step:
```
planet_owners   [60]        integer owner id, -1 = neutral
planet_coords   [60, 2]     current (x,y) position
planet_ships    [60]        ship count currently garrisoned
fleet_owners    [1024]      owner id, -1 = inactive slot
fleet_coords    [1024, 2]   current (x,y)
fleet_angles    [1024]      heading angle in radians
fleet_ship_count [1024]     ships in this fleet
step            scalar      current timestep
```

**`EnvParams`** — fixed for the whole episode:
```
planet_radii          [60]         collision radius
planet_prod           [60]         ships produced per turn
initial_planet_coords [60, 2]      spawn position for orbital math
planet_initial_angles [60]         angle at t=0 (orbiting planets)
planet_orbital_radii  [60]         distance from sun (orbiting planets)
comet_paths           [20, 150, 2] precomputed trajectory per comet
comet_lifespans       [20]         how many steps each comet is on board
is_static_planet      [60]         mask
is_orbiting_planet    [60]         mask
is_comet              [60]         mask
angular_velocity      scalar       shared rotation speed for orbiting planets
ship_speed            scalar       max fleet speed (= 6.0)
comet_spawn_steps     [60]         which step each slot spawns
body_lifespans        [60]         999999 for planets, short for comets
```

The separation keeps JIT-compiled code clean: `EnvParams` lives outside the function's output, so XLA doesn't need to track it as a mutable output.

---

## Planet Generation

Planet generation is the most complex part of setup because it must satisfy:
1. 4-fold rotational symmetry (fairness between players)
2. No planet overlaps (clearance radius enforcement)
3. At least 3 "static" groups (outside the orbiting zone)
4. At least 1 "orbiting" group (inside the orbiting zone)
5. Exactly N groups total (N drawn randomly between 5 and 10)

**Why symmetry?** Any group of 4 planets is generated as a "quad" — one planet in quadrant 1 (x>50, y>50), then three 90° rotational copies:
```
Q2 (100-x, y) ─────── Q1 (x, y)
       │                    │
       │     (50,50)        │
       │                    │
Q3 (100-y, 100-x) ─── Q4 (x, 100-y)
```

When 4 players start, each gets one planet from a single quad — they're guaranteed symmetric starting conditions.

**Why JAX while_loop?** Python `while` loops can't be JIT-compiled because the iteration count is data-dependent. `lax.while_loop` compiles the loop body once and runs it on-device, with no Python overhead per iteration.

**Phase 1** (static planets): Tries random positions in the static outer ring, keeping a count. Stops when ≥ 3 static groups are placed, or 5000 attempts are exhausted.

**Phase 2** (full target): Tries to reach the target total group count AND place at least one orbiting group.

---

## Comet System

Comets are temporary planets that pass through the board on precomputed elliptical trajectories. They:
- Spawn at predetermined steps (50, 150, 250, 350, 450)
- Have a lifespan of 5–40 turns on-board
- Produce 1 ship per turn while active
- Are occupied at spawn and claimable by any player

**Precomputed paths**: All 5 × 4 = 20 comet paths are computed at `setup()` time and stored in `EnvParams.comet_paths [20, 150, 2]`. During `step()`, looking up a comet's position is just an array index — no per-step trajectory computation.

**Trajectory generation algorithm**:
1. Sample a highly-eccentric ellipse (e=0.75–0.93) oriented at a random angle through the center
2. Discretize the ellipse into 5000 dense points
3. **Vectorized resampling**: use `jnp.cumsum` to compute arc length, then `jnp.searchsorted` to pick equally-spaced samples at intervals of `comet_speed`. This replaces a sequential `for` loop that accumulated distance step by step.
4. **Vectorized clipping**: use argmax on a boolean mask to find where the path enters/exits the board
5. **Collision check**: verify the path doesn't hit the sun or any planet (for orbiting planets, simulate their future positions during the comet's entire visit)
6. **4 symmetric copies**: one for each quadrant, stored together

If the comet path is invalid (too short, too long, collides), retry up to 300 times. Failed waves get `is_comet = False` and stay off-board.

**Off-board sentinel**: Inactive comets are placed at `(-99, -99)`. This ensures the visualizer (which filters `coord > -50`) doesn't render phantom neutral planets.

---

## Fleet Launch (`EnvAction`)

```
EnvAction.ships  [60, 64]   float ship allocations: 60 planet targets + 4 deep-space bins
EnvAction.angle  [60, 64]   angle in world frame for each launch
```

Planet-to-planet launches use a pre-computed intercept angle (see `rollout_utils.py`). The 4 "deep-space" columns are free-angle launches — the actor outputs sin/cos for these and the angle is decoded with arctan2.

**Pack-to-front**: The engine supports up to 64 launches per planet per step, totaling `60 × 64 = 3840` potential launches. Most are zero. The engine uses a stable sort to pack valid launches to the front of a fixed-size array, then fills vacant `MAX_FLEETS` slots from that sorted array.

```
valid_launches → [valid, valid, 0, valid, 0, 0, valid, ...]
                                        ↓  argsort(~valid)
sorted_launches → [valid, valid, valid, valid, 0, 0, 0, ...]
                                        ↓  fill vacant fleet slots
fleet array    [slot0=valid, slot1=valid, slot2=valid, slot3=valid, slot4=-1, ...]
```

---

## The step() Function: 9 Phases

Each call to `step()` advances the game by one timestep. The 9 phases run sequentially within a single JIT-compiled function.

```
step(state, params, actions) → (new_state, scores, terminated)

Phase 1: Planet positions       ← orbiting bodies rotate, comets advance on precomputed path
Phase 2: Active masking         ← mark expired comets as inactive
Phase 3: Fleet launch           ← deduct ships, pack launches to vacant fleet slots
Phase 4: Production             ← owned active planets generate ships
Phase 5: Fleet movement         ← advance all fleets along their heading
Phase 6: Collision detection    ← swept linear test against all bodies
Phase 7: Combat resolution      ← landed fleets battle garrison
Phase 8: State assembly         ← move expired comets off-board (-99,-99)
Phase 9: Scoring & termination  ← count ships, check win condition
```

### Phase 1: Planet Positions

Orbiting planets update each step:
```
angle(t) = initial_angle + angular_velocity × t
x(t) = 50 + orbital_radius × cos(angle(t))
y(t) = 50 + orbital_radius × sin(angle(t))
```

All planets share the same `angular_velocity`, so their relative spacing is preserved. This makes the rotational symmetry hold throughout the episode.

Comets just index into their precomputed path array:
```python
comet_age = step - comet_spawn_step
position = comet_paths[comet_idx, comet_age]
```

### Phase 5: Fleet Speed

Fleet speed follows a logarithmic curve — large fleets are slower than small ones:
```
speed = 1 + (max_speed - 1) × (log(ships) / log(1000))^1.5
speed = min(speed, max_speed)   # cap at max_speed=6
```

Analogy: it's like convoy logistics — a huge convoy moves slower because it needs to stay together. A single fast scout moves at full speed. A 1000-ship armada crawls slightly faster than a 100-ship fleet.

**Critical implementation detail**: `rollout_utils.calculate_intercept_angle()` must use `ships.astype(int32)` to compute speed, matching the `int32` truncation in `step()`. Using float ships in the intercept calculator makes predicted speeds 20–35% higher for small fleets, causing systematic misses on moving targets.

### Phase 6: Swept Collision Detection

Instead of checking point-in-circle at end of step (which misses fast fleets crossing through small planets), the engine uses swept collision: it checks whether the fleet's *movement segment* intersects the planet's swept *movement segment*.

**The quadratic**: Given fleet moving from A to B and planet moving from P to Q, define:
- `d0 = A - P` (initial separation vector)
- `dv = (B-A) - (Q-P)` (relative velocity)
- Find t ∈ [0,1] where |d0 + t·dv| = radius

This is `a·t² + b·t + c = 0` where:
```
a = |dv|²
b = 2·(d0·dv)
c = |d0|² - radius²
discriminant = b² - 4ac
```

Hit if `disc ≥ 0` and the roots overlap `[0,1]`. This is computed for all `[MAX_FLEETS, MAX_BODIES]` pairs in a vectorized broadcast — a single matrix operation.

### Phase 7: Combat Resolution

All fleets that hit a planet in the same step fight simultaneously:
1. Sum ships by player: `ships_by_player [60, num_players]`
2. Sort each planet's array descending: top-1 vs top-2
3. Winner = player with most ships; ships won = top - second
4. If `top == second`, all ships cancel (nobody wins)
5. If winner == garrison owner: friendly reinforcement (ships add)
6. If winner != garrison owner: hostile assault (ships subtract; if remainder < 0, capture)

---

## Scoring

```
player_score = ships on owned planets + ships in owned fleets
terminated   = one or fewer players remain, OR step >= 498
```

The game score is a raw ship count, not normalized. Terminal reward in the RL loop is scaled 100× to give a strong signal.
