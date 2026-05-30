import jax
from jax import lax
from jax import numpy as jnp
import jax.random as jrandom
from typing import NamedTuple
from jaxtyping import Float, Int, Bool, Array

# Named tuples for agent convenience.
# Planets and fleets share a common [id, owner, x, y, ...] prefix.
class Planet(NamedTuple):
    id: float
    owner: float
    x: float
    y: float
    radius: float
    ships: float
    production: float

class Fleet(NamedTuple):
    id: float
    owner: float
    x: float
    y: float
    angle: float
    from_planet_id: float
    ships: float

# ---------------------------------------------------------
# CONSTANTS & BOARD SETUP
# ---------------------------------------------------------
BOARD_SIZE = 100.0            # Board is a continuous 100x100 2D space.
CENTER = BOARD_SIZE / 2.0     # The center of the board is (50, 50).
SUN_RADIUS = 10.0             # Fleets that pass within this distance to the center are destroyed.
ROTATION_RADIUS_LIMIT = 50.0  # Planets inside this limit revolve around the sun. Planets outside are static.
COMET_RADIUS = 1.0            # Radius of extra-solar temporary planets (comets).
COMET_PRODUCTION = 1          # Comets produce 1 ship per turn.
PLANET_CLEARANCE = 7          # Planets must be at least this far apart from each other.
MIN_PLANET_GROUPS = 5         # Minimum number of 4-planet groups (20 planets).
MAX_PLANET_GROUPS = 10        # Maximum number of 4-planet groups (40 planets).
MIN_STATIC_GROUPS = 3         # Minimum number of static (non-revolving) 4-planet groups (12 planets).
MAX_PLANETS_BASE = MAX_PLANET_GROUPS * 4
MAX_COMETS = 4                # Max active comets at any given time.
COMET_SPAWN_STEPS = jnp.array([50, 150, 250, 350, 450]) # Specific turns where comets spawn.
TOTAL_COMETS = MAX_COMETS * COMET_SPAWN_STEPS.shape[0]
MAX_PATH_LEN = MAX_COMET_PATH_LEN = 150

MAX_BODIES = MAX_PLANETS_BASE + TOTAL_COMETS

# ---------------------------------------------------------
# VECTOR CAPS
# ---------------------------------------------------------
MAX_FLEETS_PER_PLANET_PER_STEP = 50
MAX_MOVES_PER_STEP = MAX_FLEETS_PER_PLANET_PER_STEP * MAX_PLANETS_BASE # 50 fleets launched per planet per step should be plenty of overhead
MAX_FLEET_LIFESPAN = 125
AGENT_COUNT = 4
MAX_FLEETS = 50_000 # A practical, large upper bound on concurrent fleets.

def distance(p1, p2):
    """Euclidean distance between two points. Can be batched."""
    diff = p1 - p2
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))

def point_to_segment_distance(p, v, w):
    """
    Not implemented as the separate sun intersection check isn't worth another kernel launch
    """
    raise NotImplementedError

def swept_pair_hit(A, B, P0, P1, r):
    """
    Continuous swept-pair collision detection.
    This checks if a moving fleet (from A to B) collides with a moving planet (from P0 to P1)
    within collision radius `r` at ANY continuous time `t` between [0, 1] during the tick.

    It solves the quadratic inequality for distance <= radius:
    ||(A + (B-A)t) - (P0 + (P1-P0)t)||^2 <= r^2

    Not implemented as I plan on using pure math to model intersections for garrisoning.
    """
    raise NotImplementedError

# ---------------------------------------------------------
# GENERATION FUNCTIONS
# ---------------------------------------------------------
def generate_planets(rng_key):
    """
    Generates planets with 4-fold rotational symmetry in a JAX-friendly way.
    Uses while loops and fixed-size arrays to ensure XLA compilation.
    Returns:
        planets: [MAX_PLANET_GROUPS * 4, 7] array of planets.
                 Fields: [id, owner, y, x, r, ships, production]
                 Empty slots have id = -1.
    """
    rng_key, subkey = jrandom.split(rng_key)
    
    # Decide number of Q1 planets
    num_q1 = jrandom.randint(subkey, (), MIN_PLANET_GROUPS, MAX_PLANET_GROUPS + 1)
    max_total_planets = MAX_PLANET_GROUPS * 4

    # Initialize state for Phase 1 (Static planets)
    # State holds: (planets_array, num_planets, static_groups, rng_key, attempts)
    init_state_1 = (
        jnp.full((max_total_planets, 7), -1.0),
        0,
        0,
        rng_key,
        0
    )

    def phase_1_cond(state):
        _, _, static_groups, _, attempts = state
        return (static_groups < MIN_STATIC_GROUPS) & (attempts < 5000)

    def phase_1_body(state):
        planets, num_planets, static_groups, key, attempts = state
        key, k1, k2, k3, k4a, k4b = jrandom.split(key, 6)

        prod = jrandom.randint(k1, (), 1, 6)
        r = 1.0 + jnp.log(prod)
        angle = jrandom.uniform(k2, (), minval=0.0, maxval=jnp.pi / 2.0)
        
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / jnp.maximum(jnp.cos(angle), jnp.sin(angle))
        
        # Check valid bounds for generation
        valid_bounds = min_orbital <= max_orbital
        
        orbital_r = jrandom.uniform(k3, (), minval=min_orbital, maxval=jnp.maximum(min_orbital, max_orbital))
        x = CENTER + orbital_r * jnp.cos(angle)
        y = CENTER + orbital_r * jnp.sin(angle)

        # Check conditions
        cond1 = (x + r <= BOARD_SIZE) & (x - r >= 0) & (y + r <= BOARD_SIZE) & (y - r >= 0)
        cond2 = ((BOARD_SIZE - x) - r >= 0) & ((BOARD_SIZE - y) - r >= 0)
        cond3 = ((x - CENTER) >= r + 5.0) & ((y - CENTER) >= r + 5.0)
        
        valid_generation = valid_bounds & cond1 & cond2 & cond3

        ships = jnp.minimum(jrandom.randint(k4a, (), 5, 100), jrandom.randint(k4b, (), 5, 100))

        id_base = num_planets

        tp = jnp.array([
            [id_base,     -1.0, y, x, r, ships, prod],
            [id_base + 1, -1.0, BOARD_SIZE - x, y, r, ships, prod],
            [id_base + 2, -1.0, x, BOARD_SIZE - y, r, ships, prod],
            [id_base + 3, -1.0, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod]
        ])

        # Distance checks
        # Create pairwise distance mask. We only care about populated planets (i < num_planets)
        def check_overlap(new_planet, planets, num_planets):
            # Check all existing planets
            dists = distance(jnp.array([new_planet[2], new_planet[3]]), planets[:, 2:4])
            min_dist_req = new_planet[4] + planets[:, 4] + PLANET_CLEARANCE
            # Only active planets
            active_mask = jnp.arange(max_total_planets) < num_planets
            overlap = active_mask & (dists < min_dist_req)
            return jnp.any(overlap)

        overlap1 = check_overlap(tp[0], planets, num_planets)
        overlap2 = check_overlap(tp[1], planets, num_planets)
        overlap3 = check_overlap(tp[2], planets, num_planets)
        overlap4 = check_overlap(tp[3], planets, num_planets)
        
        has_overlap = overlap1 | overlap2 | overlap3 | overlap4

        success = valid_generation & (~has_overlap)
        
        new_planets = lax.cond(
            success,
            lambda p: lax.dynamic_update_slice(p, tp, (num_planets, 0)),
            lambda p: p,
            planets
        )
        
        new_num = num_planets + jnp.where(success, 4, 0)
        new_static = static_groups + jnp.where(success, 1, 0)
        
        return (new_planets, new_num, new_static, key, attempts + 1)

    state_after_1 = lax.while_loop(phase_1_cond, phase_1_body, init_state_1)

    # Phase 2
    planets, num_planets, _, rng_key, _ = state_after_1
    
    init_state_2 = (
        planets,
        num_planets,
        False, # has_orbiting
        rng_key,
        0      # attempts
    )

    def phase_2_cond(state):
        _, num_planets, has_orbiting, _, attempts = state
        target_planets = num_q1 * 4
        # Need to reach target planets, OR we need at least one orbiting planet
        needs_more = num_planets < target_planets
        needs_orbiting = (~has_orbiting)
        return (needs_more | needs_orbiting) & (attempts < 5000) & (num_planets < max_total_planets)

    def phase_2_body(state):
        planets, num_planets, has_orbiting, key, attempts = state
        key, k1, k2, k3, k4 = jrandom.split(key, 5)

        prod = jrandom.randint(k1, (), 1, 6)
        r = 1.0 + jnp.log(prod)
        x = jrandom.uniform(k2, (), minval=CENTER + 15.0, maxval=BOARD_SIZE - r - 5.0)
        y = jrandom.uniform(k3, (), minval=CENTER + 15.0, maxval=BOARD_SIZE - r - 5.0)

        orbital_radius = distance(jnp.array([x, y]), jnp.array([CENTER, CENTER]))

        cond_sun = orbital_radius >= (SUN_RADIUS + r + 10.0)
        
        is_static = (orbital_radius + r) >= ROTATION_RADIUS_LIMIT
        cond_static_bounds = jnp.where(
            is_static,
            (x + r <= BOARD_SIZE) & (x - r >= 0) & (y + r <= BOARD_SIZE) & (y - r >= 0),
            True
        )

        valid_generation = cond_sun & cond_static_bounds

        ships = jrandom.randint(k4, (), 5, 31)
        
        id_base = num_planets
        tp = jnp.array([
            [id_base,     -1.0, y, x, r, ships, prod],
            [id_base + 1, -1.0, BOARD_SIZE - x, y, r, ships, prod],
            [id_base + 2, -1.0, x, BOARD_SIZE - y, r, ships, prod],
            [id_base + 3, -1.0, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod]
        ])

        def check_phase2_overlap(new_planet, planets, num_planets):
            tp_orbital = distance(jnp.array([new_planet[2], new_planet[3]]), jnp.array([CENTER, CENTER]))
            tp_is_rotating = (tp_orbital + new_planet[4]) < ROTATION_RADIUS_LIMIT
            
            p_orbitals = distance(planets[:, 2:4], jnp.array([CENTER, CENTER]))
            p_is_rotating = (p_orbitals + planets[:, 4]) < ROTATION_RADIUS_LIMIT
            
            dists = distance(jnp.array([new_planet[2], new_planet[3]]), planets[:, 2:4])
            min_dist_req = new_planet[4] + planets[:, 4] + PLANET_CLEARANCE
            
            overlap_standard = dists < min_dist_req
            
            # Cross-check for static vs. rotating
            cross_diff = jnp.abs(tp_orbital - p_orbitals)
            overlap_cross = (tp_is_rotating != p_is_rotating) & (cross_diff < min_dist_req)
            
            overlap_total = overlap_standard | overlap_cross
            active_mask = jnp.arange(max_total_planets) < num_planets
            return jnp.any(active_mask & overlap_total)

        overlap1 = check_phase2_overlap(tp[0], planets, num_planets)
        overlap2 = check_phase2_overlap(tp[1], planets, num_planets)
        overlap3 = check_phase2_overlap(tp[2], planets, num_planets)
        overlap4 = check_phase2_overlap(tp[3], planets, num_planets)
        
        has_overlap = overlap1 | overlap2 | overlap3 | overlap4

        success = valid_generation & (~has_overlap)
        
        new_planets = lax.cond(
            success,
            lambda p: lax.dynamic_update_slice(p, tp, (num_planets, 0)),
            lambda p: p,
            planets
        )
        
        new_num = num_planets + jnp.where(success, 4, 0)
        
        # Check if the generated group has orbiting planets
        tp_orbital = distance(jnp.array([x, y]), jnp.array([CENTER, CENTER]))
        just_generated_orbiting = (tp_orbital + r) < ROTATION_RADIUS_LIMIT
        new_has_orbiting = has_orbiting | (success & just_generated_orbiting)

        return (new_planets, new_num, new_has_orbiting, key, attempts + 1)

    state_after_2 = lax.while_loop(phase_2_cond, phase_2_body, init_state_2)
    final_planets = state_after_2[0]
    
    return final_planets

def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    assert rng is not None, "rng must be provided"
    
    # state format: (rng_key, attempts, success_flag, best_paths, valid_mask)
    init_state = (
        rng,
        0,
        jnp.array(False),
        jnp.zeros((4, MAX_PATH_LEN, 2)),
        jnp.array(0, dtype=jnp.int32) # jnp.zeros((MAX_PATH_LEN,), dtype=bool)
    )
    def comet_gen_cond(state):
        _, attempts, success, _, _ = state
        return (attempts < 300) & (~success)
    def comet_gen_body(state):
        key, attempts, _, _, _ = state
        key, k1, k2, k3 = jrandom.split(key, num=4)

        # Generate highly eccentric ellipse
        e = jrandom.uniform(k1, (), minval=0.75, maxval=0.93)
        a = jrandom.uniform(k2, (), minval=60.0, maxval=150.0)
        perihelion = a * (1 - e)
        valid_perihelion = perihelion >= SUN_RADIUS + COMET_RADIUS

        b = a * jnp.sqrt(1 - e**2)
        c_val = a * e
        phi = jrandom.uniform(k3, (), minval=jnp.pi/6, maxval=jnp.pi/3)

        # Discretize the continuous ellipse into dense array of points
        num = 5000
        t = 0.3 * jnp.pi + 1.4 * jnp.pi * jnp.arange(num) / (num - 1)
        ex = c_val + a * jnp.cos(t)
        ey = b * jnp.sin(t)
        x = CENTER + ex * jnp.cos(phi) - ey * jnp.sin(phi)
        y = CENTER + ex * jnp.sin(phi) + ey * jnp.cos(phi)
        dense = jnp.stack((x,y), axis=1)

        # ---------------------------------------------------------
        # Re-sample points (Vectorized Cumulative Sum)
        # Original sequential logic used a `for` loop to accumulate distance step-by-step
        # and conditionally `.append()` to a dynamic list.
        # ---------------------------------------------------------
        
        # Compute distances between all sequential points in parallel
        dists = distance(dense[1:], dense[:-1])
        
        # jnp.cumsum acts as a parallel cumulative sum to replace `cum += distance`
        cumulative_dists = jnp.cumsum(dists)
        cumulative_dists = jnp.pad(cumulative_dists, (1,0), constant_values=0.0)
        
        # XLA compilation requires static array sizes, so we pre-allocate an array based
        # on the mathematical absolute maximum length an arc could be (~139 steps).
        targets = jnp.arange(MAX_PATH_LEN) * comet_speed
        
        # jnp.searchsorted acts as a vectorized version of `if cum >= target`. It uses binary 
        # search to map every multiple of comet_speed directly to an index in the dense array.
        indices = jnp.searchsorted(cumulative_dists, targets) 
        
        # Prevent out-of-bounds errors for target distances that exceed the total generated arc length
        safe_idxs = jnp.clip(indices, 0, dense.shape[0] - 1)
        path = dense[safe_idxs]

        # ---------------------------------------------------------
        # Clip path to board (Vectorized Masking)
        # Original code used a `for` loop to track start/end indices of visible segments.
        # ---------------------------------------------------------
        
        #   1. Mask out padded targets that overshot the actual arc length
        valid_dist_mask = targets <= cumulative_dists[-1]
        #   2. Identify which points land inside the board boundaries
        on_board_mask = (path[:, 0] >= 0.0) & (path[:, 0] <= BOARD_SIZE) & \
                        (path[:, 1] >= 0.0) & (path[:, 1] <= BOARD_SIZE)
        valid_mask = valid_dist_mask & on_board_mask
        
        #   3. Find the continuous visible segment lengths using argmax on the boolean mask
        #      (argmax on a boolean array returns the index of the first True value).
        board_start = jnp.argmax(valid_mask)
        board_end = MAX_PATH_LEN - 1 - jnp.argmax(valid_mask[::-1])
        visible_len = jnp.where(jnp.any(valid_mask), board_end - board_start + 1, 0).astype(jnp.int32)
        
        #   4. Enforce the 5 to 40 turn lifespan constraint
        is_valid_comet = (visible_len >= 5) & (visible_len <= 40)
        
        # ---------------------------------------------------------
        # Generate 4 Symmetric Paths
        # ---------------------------------------------------------
        all_paths = jnp.stack([
            jnp.stack((path[:, 1], path[:, 0]), axis=-1),
            jnp.stack((BOARD_SIZE - path[:, 0], path[:, 1]), axis=-1),
            jnp.stack((path[:, 0], BOARD_SIZE - path[:, 1]), axis=-1),
            jnp.stack((BOARD_SIZE - path[:, 1], BOARD_SIZE - path[:, 0]), axis=-1),
        ], axis=0) # Shape: [4, MAX_PATH_LEN, 2]

        # ---------------------------------------------------------
        # Planet Masking & Separation
        # ---------------------------------------------------------
        is_active = initial_planets[:, 0] != -1.0
        if comet_planet_ids is not None:
            is_comet = jnp.isin(initial_planets[:, 0], comet_planet_ids)
        else:
            is_comet = jnp.zeros_like(is_active)
        valid_target = is_active & ~is_comet
        
        pr = distance(initial_planets[:, 2:4], jnp.array([CENTER, CENTER]))
        is_orbiting = pr + initial_planets[:, 4] < ROTATION_RADIUS_LIMIT
        is_static = ~is_orbiting

        # ---------------------------------------------------------
        # Vectorized Collision Checks (Broadcasting over [N, 4, 150])
        # ---------------------------------------------------------
        
        # 1. Sun Collision
        sun_dist = distance(path, jnp.array([CENTER, CENTER]))
        sun_collision = jnp.any(valid_mask & (sun_dist < SUN_RADIUS + COMET_RADIUS))

        # 2. Static Planets Collision
        # initial_planets: [N, 1, 1, 2], all_paths: [1, 4, MAX_PATH_LEN, 2]
        dists_to_static = distance(
            initial_planets[:, None, None, 2:4],
            all_paths[None, :, :, :]
        )
        req_dist_static = initial_planets[:, None, None, 4] + COMET_RADIUS + 0.5
        static_collision = jnp.any(
            valid_target[:, None, None] & is_static[:, None, None] & valid_mask[None, None, :] & (dists_to_static < req_dist_static)
        )

        # 3. Orbiting Planets Collision (Compute their positions at every step of the path)
        # Adjust time offset so that step 0 aligns with the first point that enters the board (board_start)
        steps = spawn_step - 1 + jnp.arange(MAX_PATH_LEN) - board_start
        dx = initial_planets[:, 2] - CENTER
        dy = initial_planets[:, 3] - CENTER
        init_angle = jnp.arctan2(dy, dx)
        cur_angle = init_angle[:, None] + angular_velocity * steps[None, :]
        
        orb_x = CENTER + pr[:, None] * jnp.cos(cur_angle)
        orb_y = CENTER + pr[:, None] * jnp.sin(cur_angle)
        orbiting_pos = jnp.stack((orb_x, orb_y), axis=-1) # Shape: [N, MAX_PATH_LEN, 2]
        
        # orbiting_pos: [N, 1, MAX_PATH_LEN, 2], all_paths: [1, 4, MAX_PATH_LEN, 2]
        dists_to_orbiting = distance(
            orbiting_pos[:, None, :, :],
            all_paths[None, :, :, :]
        )
        req_dist_orb = initial_planets[:, None, None, 4] + COMET_RADIUS
        orb_collision = jnp.any(
            valid_target[:, None, None] & is_orbiting[:, None, None] & valid_mask[None, None, :] & (dists_to_orbiting < req_dist_orb)
        )

        # Check if all conditions are met
        success = is_valid_comet & valid_perihelion & ~sun_collision & ~static_collision & ~orb_collision

        # Shift the output paths so that index 0 is the exact moment the comet hits the board bounds
        shifted_idxs = jnp.clip(jnp.arange(MAX_PATH_LEN) + board_start, 0, MAX_PATH_LEN - 1)
        shifted_all_paths = all_paths[:, shifted_idxs, :]

        return (key, attempts + 1, success, shifted_all_paths, visible_len)

    # Execute the tracing loop
    final_state = lax.while_loop(comet_gen_cond, comet_gen_body, init_state)
    _, _, success, final_paths, final_lifespan = final_state
    
    return final_paths, final_lifespan, success


class EnvAction(NamedTuple):
    # Arrays where index `[i, j]` dictates the j-th launch from Planet ID `i`
    ships: Int[Array, "MAX_BODIES MAX_FLEETS_PER_PLANET_PER_STEP"]
    angle: Float[Array, "MAX_BODIES MAX_FLEETS_PER_PLANET_PER_STEP"]

class AgentActions:
    pass

class EnvAction(NamedTuple):
    def __init__(self):
        pass

class EnvState(NamedTuple):
    # PLANET ARRS ARE MAX_BODIES long
    planet_owners: Int[Array, "MAX_BODIES"] 
    planet_coords: Float[Array, "MAX_BODIES 2"]
    planet_ships: Int[Array, "MAX_BODIES"]
    
    fleet_ids: Int[Array, "MAX_FLEETS"]
    fleet_owners: Int[Array, "MAX_FLEETS"]
    fleet_coords: Float[Array, "MAX_FLEETS 2"]
    fleet_angles: Float[Array, "MAX_FLEETS"]
    fleet_ship_count: Int[Array, "MAX_FLEETS"]
    step: Int[Array, ""] # scalar

class EnvParams(NamedTuple):
    # PLANET ARRS ARE MAX_BODIES long
    planet_radii: Float[Array, "MAX_BODIES"]
    planet_prod: Int[Array, "MAX_BODIES"]
    initial_planet_coords: Float[Array, "MAX_BODIES 2"]
    planet_initial_angles: Float[Array, "MAX_BODIES"]
    planet_orbital_radii: Float[Array, "MAX_BODIES"]
    
    comet_paths: Float[Array, "TOTAL_COMETS MAX_COMET_PATH_LEN 2"]
    is_static_planet: Bool[Array, "MAX_BODIES"]
    is_orbiting_planet: Bool[Array, "MAX_BODIES"]
    is_comet: Bool[Array, "MAX_BODIES"]
    
    angular_velocity: Float[Array, ""] # scalar
    comet_spawn_steps: Int[Array, "MAX_BODIES"]
    body_lifespans: Int[Array, "MAX_BODIES"]

    ship_speed: Float[Array, ""]

@jax.jit
def setup(rng_key: jrandom.PRNGKey) -> tuple[EnvState, EnvParams]:
    # Fix unpack bug and grab enough keys for our pre-generation
    key, k1, k2, k3, k4, k5 = jrandom.split(rng_key, 6)
    angular_velocity = jrandom.uniform(k1, (), minval=0.025, maxval=0.05)
    
    # Generate base planets (Shape: [40, 7])
    base_planets = generate_planets(k2)

    # --- Pre-generate all Comet Waves ---
    wave_keys = jrandom.split(k3, COMET_SPAWN_STEPS.shape[0])
    all_paths, all_lifespans, all_success = jax.vmap(
        lambda step, wave_rng: generate_comet_paths(
            base_planets,
            angular_velocity,
            step,
            rng=wave_rng,
        )
    )(COMET_SPAWN_STEPS, wave_keys)
    
    # Pre-roll ship counts for the 5 waves (min of 4 random ints 1-99)
    def roll_ships(rng):
        subkeys = jrandom.split(rng, 4)
        rolls = jax.vmap(lambda k: jrandom.randint(k, (), 1, 100))(subkeys)
        return jnp.min(rolls)
        
    ship_keys = jrandom.split(k4, COMET_SPAWN_STEPS.shape[0])
    all_comet_ships = jax.vmap(roll_ships)(ship_keys)
    
    valid_comet_mask = jnp.repeat(all_success, MAX_COMETS)
    
    # --- Build Pre-populated Comet Tail ---
    comet_ids = jnp.arange(MAX_PLANETS_BASE, MAX_BODIES)
    expanded_ships = jnp.repeat(all_comet_ships, MAX_COMETS)
    
    comet_tail = jnp.stack([
        comet_ids,                               # id
        jnp.full(TOTAL_COMETS, -1.0),            # owner
        jnp.full(TOTAL_COMETS, -99.0),           # y (off-board)
        jnp.full(TOTAL_COMETS, -99.0),           # x (off-board)
        jnp.full(TOTAL_COMETS, COMET_RADIUS),    # radius
        expanded_ships,                          # ships
        jnp.full(TOTAL_COMETS, COMET_PRODUCTION) # production
    ], axis=1)
    
    # Wipe dud waves back to -1.0
    comet_tail = jnp.where(valid_comet_mask[:, None], comet_tail, -1.0)
    
    # --- Joint Planet Array ---
    planets = jnp.concatenate([base_planets, comet_tail], axis=0)
    fleets = jnp.full((MAX_FLEETS, 5), -1.0)
    
    # --- Compute Planet Type Masks ---
    is_comet_slot = jnp.arange(MAX_BODIES) >= MAX_PLANETS_BASE
    is_comet = is_comet_slot & jnp.concatenate([jnp.zeros(MAX_PLANETS_BASE, dtype=bool), valid_comet_mask])

    pr = distance(planets[:, 2:4], jnp.array([CENTER, CENTER]))
    is_orbiting_base = (pr + planets[:, 4]) < ROTATION_RADIUS_LIMIT
    
    is_orbiting = is_orbiting_base & ~is_comet_slot & (planets[:, 0] != -1.0)
    is_static = ~is_orbiting_base & ~is_comet_slot & (planets[:, 0] != -1.0)

    # --- Unified Time Tracking ---
    base_spawn_steps = jnp.zeros(MAX_PLANETS_BASE, dtype=jnp.int32)
    valid_comet_spawns = jnp.where(all_success, COMET_SPAWN_STEPS, -1)
    comet_spawn_steps = jnp.repeat(valid_comet_spawns, MAX_COMETS)
    spawn_steps = jnp.concatenate([base_spawn_steps, comet_spawn_steps])

    # --- Unified Lifespan Tracking ---
    base_lifespans = jnp.full(MAX_PLANETS_BASE, 999999, dtype=jnp.int32)
    comet_lifespans = jnp.repeat(all_lifespans, MAX_COMETS).astype(jnp.int32)
    body_lifespans = jnp.concatenate([base_lifespans, comet_lifespans])

    # --- Precompute Orbital Math ---
    initial_coords = planets[:, 2:4][:, ::-1] # Swap y, x to x, y
    dx = initial_coords[:, 0] - CENTER
    dy = initial_coords[:, 1] - CENTER
    orbital_radii = jnp.sqrt(dx**2 + dy**2)
    initial_angles = jnp.arctan2(dy, dx)


    # Home planet assignment: pick a random group of 4 and assign one planet per player.
    # Symmetry is guaranteed by generate_planets, which always writes groups of 4 as
    # 90° rotational copies — any consecutive group [base, base+3] is a fair starting set.
    num_valid_planets = jnp.sum(base_planets[:, 0] != -1.0).astype(jnp.int32)
    num_groups = num_valid_planets // 4
    home_group = jrandom.randint(k5, (), 0, MAX_PLANET_GROUPS) % num_groups
    base = home_group * 4

    body_indices = jnp.arange(MAX_BODIES)
    is_home = (body_indices >= base) & (body_indices < base + AGENT_COUNT)

    planet_owners_col = jnp.where(is_home, body_indices - base, planets[:, 1].astype(jnp.int32))
    planet_ships_col  = jnp.where(is_home, 10.0, planets[:, 5])

    # planets is currently in [id, owner, y, x, radius, ships, prod] format
    # fleets is currently in [id, angle, x, y, ships]
    state = EnvState(
        planet_owners=planet_owners_col,
        planet_coords=planets[:, 2:4][:, ::-1],  # swap y, x to x, y
        planet_ships=planet_ships_col.astype(jnp.int32),
        fleet_ids=fleets[:, 0].astype(jnp.int32),
        fleet_owners=jnp.full(MAX_FLEETS, -1, dtype=jnp.int32),
        fleet_coords=fleets[:, 2:4],
        fleet_angles=fleets[:, 1],
        fleet_ship_count=fleets[:, 4].astype(jnp.int32),
        step = jnp.array(0),
    )
    
    params = EnvParams(
        planet_radii=planets[:, 4],
        planet_prod=planets[:, 6].astype(jnp.int32),
        initial_planet_coords=planets[:, 2:4][:, ::-1],  # swap y, x to x, y
        planet_initial_angles=initial_angles,
        planet_orbital_radii=orbital_radii,
        comet_paths=all_paths.reshape((TOTAL_COMETS, MAX_COMET_PATH_LEN, 2)),
        is_static_planet=is_static,
        is_orbiting_planet=is_orbiting,
        is_comet=is_comet,
        angular_velocity=jnp.array(angular_velocity),
        comet_spawn_steps=spawn_steps,
        body_lifespans=body_lifespans,
        ship_speed = jnp.array(6.0)
    )

    return state, params

def step(state: EnvState, params: EnvParams, actions: EnvAction) -> EnvState:
    # TODO: Implement JAX environment step logic

    step = state.step + 1
    
    # Calculate ages and safely index into the [20, 150, 2] path array

    # how old each body is, in steps, accounting for the step they were spawned
    ages = step - params.comet_spawn_steps
    # just the ages of the comets, they're negative when they haven't been spawned
    comet_ages = ages[-TOTAL_COMETS:]
    # we restrict the ages array to be in [0, MAX_COMET_PATH_LEN-1] as we use it to index our path array
    safe_comet_ages = jnp.clip(comet_ages, 0, MAX_COMET_PATH_LEN - 1)
    # this updates both inactive and active comets' paths
    # a side effect of the clip op is that unspawned and despawned comets are given a position
    raw_comet_locations = params.comet_paths[jnp.arange(TOTAL_COMETS), safe_comet_ages, :]
    # this is the full array where we have the planet positions and the updated (raw) comet positions
    comet_location_update = jnp.concatenate((state.planet_coords[:-TOTAL_COMETS], raw_comet_locations), axis=0)
    
    # these two lines basically check if comets are actually alive, so we know which ones to actually update
    # Instead of MAX_COMET_PATH_LEN, we perfectly replicate the original code by using the comet's specific generated lifespan!
    spawned_comets = (0 <= ages) & (ages < params.body_lifespans)
    active_comets = params.is_comet & spawned_comets
    
    # once we figured out which comets are active, we update the locations of only active comets
    comet_updated_coords = jnp.where(active_comets[:, None], comet_location_update, state.planet_coords)

    # here we mask all bodies that aren't in play, ensuring we only have spawned bodies in the env
    # We no longer need spatial OOB checks for bodies! The lifespan timeline natively filters out dead comets.
    active_bodies = active_comets | params.is_static_planet | params.is_orbiting_planet

    # now for planet position update
    new_orbital_angles = params.planet_initial_angles + params.angular_velocity * step
    new_orbital_ratios = jnp.stack((jnp.cos(new_orbital_angles),jnp.sin(new_orbital_angles)), axis=1)
    new_orbital_coords = new_orbital_ratios * params.planet_orbital_radii[:, None] + CENTER
    body_coord_update = jnp.where(params.is_orbiting_planet[:,None], new_orbital_coords, comet_updated_coords)



    new_state = EnvState(
        planet_coords=body_coord_update,
    )

    return new_state