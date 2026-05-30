"""
JAX Orbit Wars Simulator
========================
A fully vectorized, JIT/vmap-compatible implementation of the Orbit Wars game engine.
See orbit_wars_sim_docs.md for a human-readable explanation of the rules and design.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax
from typing import NamedTuple
from jaxtyping import Float, Int, Bool, Array

# Generation functions are defined inline below (orbit_wars_jax.py has syntax errors
# in its class stubs that prevent importing from it).

# ---------------------------------------------------------------------------
# GAME CONSTANTS  (must match original_orbit_wars.py)
# ---------------------------------------------------------------------------
BOARD_SIZE             = 100.0
CENTER                 = 50.0
SUN_RADIUS             = 10.0
ROTATION_RADIUS_LIMIT  = 50.0
COMET_RADIUS           = 1.0
COMET_PRODUCTION       = 1
PLANET_CLEARANCE       = 7
MIN_PLANET_GROUPS      = 5
MAX_PLANET_GROUPS      = 10
MIN_STATIC_GROUPS      = 3
COMET_SPAWN_STEPS_LIST = [50, 150, 250, 350, 450]
COMET_SPAWN_STEPS      = jnp.array(COMET_SPAWN_STEPS_LIST, dtype=jnp.int32)

# ---------------------------------------------------------------------------
# CAPACITY CONSTANTS  (tunable — increase if you hit overflow)
# ---------------------------------------------------------------------------
MAX_AGENTS         = 4     # static; 2-player games just leave agents 2/3 inactive
MAX_PLANETS_BASE   = MAX_PLANET_GROUPS * 4   # 40
MAX_COMETS_PER_WAVE = 4                       # one per quadrant per spawn wave
NUM_COMET_WAVES    = len(COMET_SPAWN_STEPS_LIST)
TOTAL_COMETS       = MAX_COMETS_PER_WAVE * NUM_COMET_WAVES  # 20
MAX_BODIES         = MAX_PLANETS_BASE + TOTAL_COMETS        # 60
MAX_PATH_LEN       = 150
MAX_FLEETS         = 8_000
MAX_MOVES_PER_STEP = 10    # max launches any single player can request per step

# Sentinel values for inactive/empty slots
OWNER_NEUTRAL  = -1   # planet exists but belongs to nobody
OWNER_INACTIVE = -2   # planet slot is not in play at all
FLEET_EMPTY    = -1   # fleet slot is unused

DEFAULT_SHIP_SPEED  = 6.0
DEFAULT_EPISODE_STEPS = 500
DEFAULT_COMET_SPEED = 4.0


# ---------------------------------------------------------------------------
# PLANET & COMET GENERATION  (JAX-native, JIT/vmap-safe)
# ---------------------------------------------------------------------------

def _gen_distance(p1, p2):
    diff = p1 - p2
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))


def generate_planets(rng_key):
    """
    Generates planets with 4-fold rotational symmetry.
    Returns [MAX_PLANETS_BASE, 7] — columns: [id, owner, y, x, r, ships, production].
    Empty slots have id == -1.
    """
    rng_key, subkey = jrandom.split(rng_key)
    num_q1          = jrandom.randint(subkey, (), MIN_PLANET_GROUPS, MAX_PLANET_GROUPS + 1)
    max_total       = MAX_PLANETS_BASE

    # ---- Phase 1: guarantee MIN_STATIC_GROUPS static planet groups ----
    init1 = (jnp.full((max_total, 7), -1.0), 0, 0, rng_key, 0)

    def p1_cond(s):
        _, _, static_groups, _, attempts = s
        return (static_groups < MIN_STATIC_GROUPS) & (attempts < 5000)

    def p1_body(s):
        planets, num_planets, static_groups, key, attempts = s
        key, k1, k2, k3, k4a, k4b = jrandom.split(key, 6)

        prod = jrandom.randint(k1, (), 1, 6)
        r    = 1.0 + jnp.log(prod)
        ang  = jrandom.uniform(k2, (), minval=0.0, maxval=jnp.pi / 2.0)

        min_orb = ROTATION_RADIUS_LIMIT - r
        max_orb = (BOARD_SIZE - CENTER - r) / jnp.maximum(jnp.cos(ang), jnp.sin(ang))
        valid_bounds = min_orb <= max_orb
        orb_r = jrandom.uniform(k3, (), minval=min_orb, maxval=jnp.maximum(min_orb, max_orb))
        x = CENTER + orb_r * jnp.cos(ang)
        y = CENTER + orb_r * jnp.sin(ang)

        c1 = (x + r <= BOARD_SIZE) & (x - r >= 0) & (y + r <= BOARD_SIZE) & (y - r >= 0)
        c2 = ((BOARD_SIZE - x) - r >= 0) & ((BOARD_SIZE - y) - r >= 0)
        c3 = ((x - CENTER) >= r + 5.0) & ((y - CENTER) >= r + 5.0)
        valid_gen = valid_bounds & c1 & c2 & c3

        ships = jnp.minimum(jrandom.randint(k4a, (), 5, 100), jrandom.randint(k4b, (), 5, 100))
        ib = num_planets
        tp = jnp.array([
            [ib,   -1.0, y,              x,              r, ships, prod],
            [ib+1, -1.0, BOARD_SIZE - x, y,              r, ships, prod],
            [ib+2, -1.0, x,              BOARD_SIZE - y, r, ships, prod],
            [ib+3, -1.0, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ])

        def check_overlap(new_p, existing, n):
            dists = _gen_distance(jnp.array([new_p[2], new_p[3]]), existing[:, 2:4])
            req   = new_p[4] + existing[:, 4] + PLANET_CLEARANCE
            mask  = jnp.arange(max_total) < n
            return jnp.any(mask & (dists < req))

        overlap = (check_overlap(tp[0], planets, num_planets) |
                   check_overlap(tp[1], planets, num_planets) |
                   check_overlap(tp[2], planets, num_planets) |
                   check_overlap(tp[3], planets, num_planets))
        success = valid_gen & ~overlap

        new_planets = lax.cond(
            success,
            lambda p: lax.dynamic_update_slice(p, tp, (num_planets, 0)),
            lambda p: p,
            planets,
        )
        return (new_planets,
                num_planets + jnp.where(success, 4, 0),
                static_groups + jnp.where(success, 1, 0),
                key, attempts + 1)

    planets, num_planets, _, rng_key, _ = lax.while_loop(p1_cond, p1_body, init1)

    # ---- Phase 2: fill remaining groups, ensure at least one orbiting ----
    init2 = (planets, num_planets, False, rng_key, 0)

    def p2_cond(s):
        _, num_p, has_orb, _, attempts = s
        target = num_q1 * 4
        return (((num_p < target) | ~has_orb) & (attempts < 5000) & (num_p < max_total))

    def p2_body(s):
        planets, num_planets, has_orbiting, key, attempts = s
        key, k1, k2, k3, k4 = jrandom.split(key, 5)

        prod = jrandom.randint(k1, (), 1, 6)
        r    = 1.0 + jnp.log(prod)
        x    = jrandom.uniform(k2, (), minval=CENTER + 15.0, maxval=BOARD_SIZE - r - 5.0)
        y    = jrandom.uniform(k3, (), minval=CENTER + 15.0, maxval=BOARD_SIZE - r - 5.0)

        orb_r    = _gen_distance(jnp.array([x, y]), jnp.array([CENTER, CENTER]))
        cond_sun = orb_r >= (SUN_RADIUS + r + 10.0)
        is_stat  = (orb_r + r) >= ROTATION_RADIUS_LIMIT
        cond_bnd = jnp.where(
            is_stat,
            (x + r <= BOARD_SIZE) & (x - r >= 0) & (y + r <= BOARD_SIZE) & (y - r >= 0),
            True,
        )
        valid_gen = cond_sun & cond_bnd
        ships = jrandom.randint(k4, (), 5, 31)

        ib = num_planets
        tp = jnp.array([
            [ib,   -1.0, y,              x,              r, ships, prod],
            [ib+1, -1.0, BOARD_SIZE - x, y,              r, ships, prod],
            [ib+2, -1.0, x,              BOARD_SIZE - y, r, ships, prod],
            [ib+3, -1.0, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ])

        def check_p2_overlap(new_p, existing, n):
            tp_orb  = _gen_distance(jnp.array([new_p[2], new_p[3]]), jnp.array([CENTER, CENTER]))
            tp_rot  = (tp_orb + new_p[4]) < ROTATION_RADIUS_LIMIT
            p_orbs  = _gen_distance(existing[:, 2:4], jnp.array([CENTER, CENTER]))
            p_rot   = (p_orbs + existing[:, 4]) < ROTATION_RADIUS_LIMIT
            dists   = _gen_distance(jnp.array([new_p[2], new_p[3]]), existing[:, 2:4])
            req     = new_p[4] + existing[:, 4] + PLANET_CLEARANCE
            std_ov  = dists < req
            cross_ov= (tp_rot != p_rot) & (jnp.abs(tp_orb - p_orbs) < req)
            mask    = jnp.arange(max_total) < n
            return jnp.any(mask & (std_ov | cross_ov))

        overlap = (check_p2_overlap(tp[0], planets, num_planets) |
                   check_p2_overlap(tp[1], planets, num_planets) |
                   check_p2_overlap(tp[2], planets, num_planets) |
                   check_p2_overlap(tp[3], planets, num_planets))
        success = valid_gen & ~overlap

        new_planets = lax.cond(
            success,
            lambda p: lax.dynamic_update_slice(p, tp, (num_planets, 0)),
            lambda p: p,
            planets,
        )
        just_orb    = (orb_r + r) < ROTATION_RADIUS_LIMIT
        new_has_orb = has_orbiting | (success & just_orb)
        return (new_planets,
                num_planets + jnp.where(success, 4, 0),
                new_has_orb, key, attempts + 1)

    final, _, _, _, _ = lax.while_loop(p2_cond, p2_body, init2)
    return final


def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    """
    Tries up to 300 times to find a valid 4-fold symmetric comet trajectory.
    Returns (paths [4, MAX_PATH_LEN, 2], lifespan int32, success bool).
    """
    assert rng is not None, "rng key required"

    init = (rng, 0, jnp.array(False),
            jnp.zeros((4, MAX_PATH_LEN, 2)),
            jnp.array(0, dtype=jnp.int32))

    def cond(s):
        _, attempts, success, _, _ = s
        return (attempts < 300) & ~success

    def body(s):
        key, attempts, _, _, _ = s
        key, k1, k2, k3 = jrandom.split(key, 4)

        e   = jrandom.uniform(k1, (), minval=0.75, maxval=0.93)
        a   = jrandom.uniform(k2, (), minval=60.0, maxval=150.0)
        peri = a * (1 - e)
        valid_peri = peri >= SUN_RADIUS + COMET_RADIUS

        b     = a * jnp.sqrt(1 - e ** 2)
        c_val = a * e
        phi   = jrandom.uniform(k3, (), minval=jnp.pi / 6, maxval=jnp.pi / 3)

        num = 5000
        t   = 0.3 * jnp.pi + 1.4 * jnp.pi * jnp.arange(num) / (num - 1)
        ex  = c_val + a * jnp.cos(t)
        ey  = b * jnp.sin(t)
        x   = CENTER + ex * jnp.cos(phi) - ey * jnp.sin(phi)
        y   = CENTER + ex * jnp.sin(phi) + ey * jnp.cos(phi)
        dense = jnp.stack((x, y), axis=1)

        dists   = _gen_distance(dense[1:], dense[:-1])
        cum     = jnp.cumsum(dists)
        cum     = jnp.pad(cum, (1, 0), constant_values=0.0)
        targets = jnp.arange(MAX_PATH_LEN) * comet_speed
        idxs    = jnp.searchsorted(cum, targets)
        safe    = jnp.clip(idxs, 0, num - 1)
        path    = dense[safe]

        valid_dist = targets <= cum[-1]
        on_board   = ((path[:, 0] >= 0.0) & (path[:, 0] <= BOARD_SIZE) &
                      (path[:, 1] >= 0.0) & (path[:, 1] <= BOARD_SIZE))
        valid_mask = valid_dist & on_board

        board_start = jnp.argmax(valid_mask)
        board_end   = MAX_PATH_LEN - 1 - jnp.argmax(valid_mask[::-1])
        vis_len     = jnp.where(jnp.any(valid_mask), board_end - board_start + 1, 0).astype(jnp.int32)
        valid_comet = (vis_len >= 5) & (vis_len <= 40)

        all_paths = jnp.stack([
            jnp.stack((path[:, 1], path[:, 0]), axis=-1),
            jnp.stack((BOARD_SIZE - path[:, 0], path[:, 1]), axis=-1),
            jnp.stack((path[:, 0], BOARD_SIZE - path[:, 1]), axis=-1),
            jnp.stack((BOARD_SIZE - path[:, 1], BOARD_SIZE - path[:, 0]), axis=-1),
        ], axis=0)

        is_active = initial_planets[:, 0] != -1.0
        is_comet_p = jnp.zeros_like(is_active) if comet_planet_ids is None else \
                     jnp.isin(initial_planets[:, 0], comet_planet_ids)
        valid_tgt = is_active & ~is_comet_p

        pr       = _gen_distance(initial_planets[:, 2:4], jnp.array([CENTER, CENTER]))
        is_orb   = pr + initial_planets[:, 4] < ROTATION_RADIUS_LIMIT
        is_stat  = ~is_orb

        # Sun check
        sun_dist  = _gen_distance(path, jnp.array([CENTER, CENTER]))
        sun_coll  = jnp.any(valid_mask & (sun_dist < SUN_RADIUS + COMET_RADIUS))

        # Static planet check
        d_static  = _gen_distance(initial_planets[:, None, None, 2:4], all_paths[None])
        req_stat  = initial_planets[:, None, None, 4] + COMET_RADIUS + 0.5
        stat_coll = jnp.any(
            valid_tgt[:, None, None] & is_stat[:, None, None] &
            valid_mask[None, None, :] & (d_static < req_stat)
        )

        # Orbiting planet check
        steps_arr = spawn_step - 1 + jnp.arange(MAX_PATH_LEN) - board_start
        dx_p = initial_planets[:, 2] - CENTER
        dy_p = initial_planets[:, 3] - CENTER
        ia   = jnp.arctan2(dy_p, dx_p)
        ca   = ia[:, None] + angular_velocity * steps_arr[None, :]
        orb_pos = jnp.stack([CENTER + pr[:, None] * jnp.cos(ca),
                              CENTER + pr[:, None] * jnp.sin(ca)], axis=-1)
        d_orb   = _gen_distance(orb_pos[:, None, :, :], all_paths[None])
        req_orb = initial_planets[:, None, None, 4] + COMET_RADIUS
        orb_coll= jnp.any(
            valid_tgt[:, None, None] & is_orb[:, None, None] &
            valid_mask[None, None, :] & (d_orb < req_orb)
        )

        success = valid_comet & valid_peri & ~sun_coll & ~stat_coll & ~orb_coll

        shifted_idxs = jnp.clip(jnp.arange(MAX_PATH_LEN) + board_start, 0, MAX_PATH_LEN - 1)
        shifted = all_paths[:, shifted_idxs, :]

        return (key, attempts + 1, success, shifted, vis_len)

    _, _, success, final_paths, final_lifespan = lax.while_loop(cond, body, init)
    return final_paths, final_lifespan, success


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

class EnvParams(NamedTuple):
    """Episode-level constants.  Frozen at setup(); passed to every step()."""
    # Per-body geometry (MAX_BODIES entries; inactive slots are zeroed/flagged)
    planet_radii:          Float[Array, "MAX_BODIES"]
    planet_prod:           Float[Array, "MAX_BODIES"]
    initial_planet_coords: Float[Array, "MAX_BODIES 2"]   # (x, y) at t=0
    planet_orbital_radii:  Float[Array, "MAX_BODIES"]
    planet_initial_angles: Float[Array, "MAX_BODIES"]

    # Body-type masks — mutually exclusive except for invalid slots
    is_static:      Bool[Array, "MAX_BODIES"]  # orbital_r + r >= ROTATION_RADIUS_LIMIT
    is_orbiting:    Bool[Array, "MAX_BODIES"]  # orbital_r + r <  ROTATION_RADIUS_LIMIT
    is_comet_slot:  Bool[Array, "MAX_BODIES"]  # slot index >= MAX_PLANETS_BASE

    # Comet scheduling — indexed by body slot (not wave)
    comet_paths:         Float[Array, "TOTAL_COMETS MAX_PATH_LEN 2"]  # pre-generated paths
    comet_spawn_steps:   Int[Array, "MAX_BODIES"]   # when each slot becomes active (0 for base planets)
    body_lifespans:      Int[Array, "MAX_BODIES"]   # how many steps a body stays active
    planet_ships_at_spawn: Float[Array, "MAX_BODIES"]  # ships when a comet first appears

    # Scalars
    angular_velocity: Float[Array, ""]
    ship_speed:       Float[Array, ""]

    # Python ints (static for JIT — changing these requires recompile)
    episode_steps:     int
    num_active_agents: int  # 2 or 4; controls home planet assignment


class EnvState(NamedTuple):
    """Per-step mutable game state."""
    planet_owners: Int[Array, "MAX_BODIES"]     # OWNER_NEUTRAL=-1, OWNER_INACTIVE=-2, else player id
    planet_coords: Float[Array, "MAX_BODIES 2"] # current (x, y) of each body
    planet_ships:  Float[Array, "MAX_BODIES"]

    fleet_owners:  Int[Array, "MAX_FLEETS"]     # FLEET_EMPTY=-1, else player id
    fleet_coords:  Float[Array, "MAX_FLEETS 2"]
    fleet_angles:  Float[Array, "MAX_FLEETS"]
    fleet_ships:   Float[Array, "MAX_FLEETS"]

    next_fleet_slot: Int[Array, ""]  # circular insertion pointer
    step:            Int[Array, ""]


class EnvAction(NamedTuple):
    """Actions for one step.  Always [MAX_AGENTS, MAX_MOVES_PER_STEP] shaped."""
    planet_ids: Int[Array, "MAX_AGENTS MAX_MOVES_PER_STEP"]   # body-array index; -1 = no-op
    angles:     Float[Array, "MAX_AGENTS MAX_MOVES_PER_STEP"]
    ships:      Int[Array, "MAX_AGENTS MAX_MOVES_PER_STEP"]   # 0 = no-op


# ---------------------------------------------------------------------------
# MATH HELPERS
# ---------------------------------------------------------------------------

def _dist(a: Array, b: Array) -> Array:
    """Euclidean distance. Works on any trailing 2-D coordinate axis."""
    diff = a - b
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1))


def _swept_pair_hit(
    ax: Array, ay: Array, bx: Array, by: Array,
    p0x: Array, p0y: Array, p1x: Array, p1y: Array,
    r: Array,
) -> Array:
    """
    Continuous collision: does a fleet moving A→B come within radius r of a
    body moving P0→P1 at any moment during the tick?

    Solves ||(A + (B-A)t) - (P0 + (P1-P0)t)||^2 <= r^2  for t ∈ [0,1].
    Returns a bool (or bool array when inputs are batched).
    """
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)

    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    disc    = b * b - 4.0 * a * c
    safe_a  = jnp.where(a > 1e-12, a, 1.0)
    sq      = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1      = (-b - sq) / (2.0 * safe_a)
    t2      = (-b + sq) / (2.0 * safe_a)

    linear_hit = c <= 0.0
    quad_hit   = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(a < 1e-12, linear_hit, quad_hit)


def _point_to_segment_dist(px: float, py: float,
                            v: Array, w: Array) -> Array:
    """
    Minimum distance from fixed point (px, py) to each line segment v[i]→w[i].
    v, w: [..., 2]   returns [...]
    """
    vx, vy = v[..., 0], v[..., 1]
    wx, wy = w[..., 0], w[..., 1]
    dx, dy = wx - vx, wy - vy
    l2 = dx * dx + dy * dy
    t = jnp.where(
        l2 > 0,
        jnp.clip(((px - vx) * dx + (py - vy) * dy) / jnp.where(l2 > 0, l2, 1.0), 0.0, 1.0),
        0.0,
    )
    proj_x = vx + t * dx
    proj_y = vy + t * dy
    return jnp.sqrt((proj_x - px) ** 2 + (proj_y - py) ** 2)


# ---------------------------------------------------------------------------
# ACTIVE-BODY MASK
# ---------------------------------------------------------------------------

def _active_bodies(state: EnvState, params: EnvParams) -> Array:
    """Bool [MAX_BODIES]: which body slots are currently in play."""
    comet_age = state.step - params.comet_spawn_steps
    spawned   = (comet_age >= 0) & (comet_age < params.body_lifespans)
    return params.is_static | params.is_orbiting | (params.is_comet_slot & spawned)


# ---------------------------------------------------------------------------
# PLANET POSITION UPDATE
# ---------------------------------------------------------------------------

def _compute_new_coords(state: EnvState, params: EnvParams):
    """
    Compute where every body will be at the END of this tick.
    Returns (new_coords [MAX_BODIES, 2], check_collision_mask [MAX_BODIES]).

    check_collision_mask is False for comets on their very first step (old pos
    is off-board, so swept collision would give a bogus result).
    """
    step = state.step.astype(jnp.float32)

    # --- orbiting base planets ---
    new_angle  = params.planet_initial_angles + params.angular_velocity * step
    orb_x = CENTER + params.planet_orbital_radii * jnp.cos(new_angle)
    orb_y = CENTER + params.planet_orbital_radii * jnp.sin(new_angle)
    orbital_coords = jnp.stack([orb_x, orb_y], axis=-1)

    # orbiting planets replace their stored coord; static keep current coord
    new_coords = jnp.where(
        params.is_orbiting[:, None],
        orbital_coords,
        state.planet_coords,
    )

    # --- comets: index into pre-generated path ---
    comet_age = state.step - params.comet_spawn_steps   # [MAX_BODIES], int
    # comet slots are [MAX_PLANETS_BASE .. MAX_BODIES)
    # comet_paths is indexed by comet slot index (0 .. TOTAL_COMETS-1)
    comet_slot_idx = jnp.arange(MAX_BODIES) - MAX_PLANETS_BASE   # negative for base slots
    safe_slot      = jnp.clip(comet_slot_idx, 0, TOTAL_COMETS - 1)
    safe_age       = comet_age.astype(jnp.int32)
    safe_age       = jnp.clip(safe_age, 0, MAX_PATH_LEN - 1)

    comet_pos = params.comet_paths[safe_slot, safe_age, :]  # [MAX_BODIES, 2]

    new_coords = jnp.where(
        params.is_comet_slot[:, None],
        comet_pos,
        new_coords,
    )

    # check_collision: False on a comet's very first step (age==0, old pos is -99)
    first_step_comet = params.is_comet_slot & (comet_age == 0)
    check_collision  = ~first_step_comet

    return new_coords, check_collision


# ---------------------------------------------------------------------------
# FLEET LAUNCH
# ---------------------------------------------------------------------------


def _do_launches(state: EnvState, params: EnvParams, actions: EnvAction) -> EnvState:
    """
    Process all launch requests from all players.
    Uses lax.scan to iterate over (MAX_AGENTS * MAX_MOVES_PER_STEP) action slots.
    Each slot either does nothing (no-op) or launches one fleet.
    """
    # Flatten actions to [MAX_AGENTS * MAX_MOVES_PER_STEP] 1-D slices
    total = MAX_AGENTS * MAX_MOVES_PER_STEP

    # player_ids[i] tells us which player owns action slot i
    player_ids = jnp.repeat(jnp.arange(MAX_AGENTS, dtype=jnp.int32), MAX_MOVES_PER_STEP)
    planet_idxs = actions.planet_ids.reshape(total)   # [total]
    angles      = actions.angles.reshape(total)        # [total]
    ships_reqs  = actions.ships.reshape(total).astype(jnp.float32)

    action_flat = (player_ids, planet_idxs, angles, ships_reqs)

    init_carry = (
        state.planet_owners,
        state.planet_ships,
        state.planet_coords,   # needed to compute launch start position
        state.fleet_owners,
        state.fleet_coords,
        state.fleet_angles,
        state.fleet_ships,
        state.next_fleet_slot,
    )

    def scan_body(carry, xs):
        (p_owners, p_ships, p_coords,
         f_owners, f_coords, f_angles, f_ships, next_slot) = carry
        player_id, planet_idx, angle, ships_req = xs

        planet_idx_i = planet_idx.astype(jnp.int32)
        ships_f      = ships_req.astype(jnp.float32)

        # Validate
        in_range        = (planet_idx_i >= 0) & (planet_idx_i < MAX_BODIES)
        owned           = p_owners[planet_idx_i] == player_id
        enough_ships    = p_ships[planet_idx_i] >= ships_f
        positive        = ships_f > 0.0
        valid           = in_range & owned & enough_ships & positive

        # Deduct ships from source planet
        body_range = jnp.arange(MAX_BODIES, dtype=jnp.int32)
        deduct_mask = (body_range == planet_idx_i) & valid
        p_ships_new = p_ships - jnp.where(deduct_mask, ships_f, 0.0)

        # Compute start position (just outside planet radius)
        r_src    = params.planet_radii[planet_idx_i]
        src_xy   = p_coords[planet_idx_i]
        start_xy = src_xy + jnp.array([jnp.cos(angle), jnp.sin(angle)]) * (r_src + 0.1)

        # Insert into fleet pool at next_slot (circular, wraps at MAX_FLEETS)
        slot = next_slot % MAX_FLEETS
        fleet_range = jnp.arange(MAX_FLEETS, dtype=jnp.int32)
        insert_mask = (fleet_range == slot) & valid

        f_owners_new = jnp.where(insert_mask, player_id,    f_owners)
        f_coords_new = jnp.where(insert_mask[:, None], start_xy[None, :], f_coords)
        f_angles_new = jnp.where(insert_mask, angle,        f_angles)
        f_ships_new  = jnp.where(insert_mask, ships_f,      f_ships)

        next_slot_new = jnp.where(valid, next_slot + 1, next_slot)

        new_carry = (
            p_owners, p_ships_new, p_coords,
            f_owners_new, f_coords_new, f_angles_new, f_ships_new,
            next_slot_new,
        )
        return new_carry, None

    xs = (player_ids, planet_idxs, angles, ships_reqs)
    final_carry, _ = lax.scan(scan_body, init_carry, xs)

    (p_owners, p_ships, p_coords,
     f_owners, f_coords, f_angles, f_ships, next_slot) = final_carry

    return EnvState(
        planet_owners    = p_owners,
        planet_coords    = p_coords,
        planet_ships     = p_ships,
        fleet_owners     = f_owners,
        fleet_coords     = f_coords,
        fleet_angles     = f_angles,
        fleet_ships      = f_ships,
        next_fleet_slot  = next_slot,
        step             = state.step,
    )


# ---------------------------------------------------------------------------
# STEP FUNCTION
# ---------------------------------------------------------------------------

@jax.jit
def step(state: EnvState, params: EnvParams, actions: EnvAction) -> EnvState:
    """One game tick.  Pure function — safe to jit and vmap."""

    # ------------------------------------------------------------------ #
    # 1. Fleet launch                                                      #
    # ------------------------------------------------------------------ #
    state = _do_launches(state, params, actions)

    # ------------------------------------------------------------------ #
    # 2. Production — owned active bodies gain ships                      #
    # ------------------------------------------------------------------ #
    active = _active_bodies(state, params)
    owned  = (state.planet_owners >= 0) & active
    state  = EnvState(
        planet_owners    = state.planet_owners,
        planet_coords    = state.planet_coords,
        planet_ships     = state.planet_ships + jnp.where(owned, params.planet_prod, 0.0),
        fleet_owners     = state.fleet_owners,
        fleet_coords     = state.fleet_coords,
        fleet_angles     = state.fleet_angles,
        fleet_ships      = state.fleet_ships,
        next_fleet_slot  = state.next_fleet_slot,
        step             = state.step,
    )

    # ------------------------------------------------------------------ #
    # 3. Compute where every body will be at end of tick                  #
    # ------------------------------------------------------------------ #
    old_coords              = state.planet_coords          # [B, 2]
    new_coords, check_mask  = _compute_new_coords(state, params)   # [B, 2], [B]

    # ------------------------------------------------------------------ #
    # 4. Fleet movement and collision detection                            #
    # ------------------------------------------------------------------ #
    live      = state.fleet_owners >= 0                    # [F]
    ships_f   = jnp.maximum(state.fleet_ships, 1.0)
    raw_speed = (
        1.0 + (params.ship_speed - 1.0)
            * (jnp.log(ships_f) / jnp.log(1000.0)) ** 1.5
    )
    speed     = jnp.minimum(raw_speed, params.ship_speed)  # [F]

    old_f = state.fleet_coords                              # [F, 2]
    dxy   = jnp.stack([jnp.cos(state.fleet_angles),
                        jnp.sin(state.fleet_angles)], axis=-1)
    new_f = old_f + dxy * speed[:, None]                   # [F, 2]

    # Swept collision: [F, B]
    # Broadcast: fleet axes → (F, 1), body axes → (1, B)
    hit = _swept_pair_hit(
        old_f[:, 0:1], old_f[:, 1:2],      # A  [F, 1]
        new_f[:, 0:1], new_f[:, 1:2],      # B  [F, 1]
        old_coords[None, :, 0], old_coords[None, :, 1],   # P0 [1, B]
        new_coords[None, :, 0], new_coords[None, :, 1],   # P1 [1, B]
        params.planet_radii[None, :],                      # r  [1, B]
    )                                                       # [F, B]
    hit = hit & check_mask[None, :]      # ignore first-placement comets
    hit = hit & active[None, :]          # ignore inactive slots
    hit = hit & live[:, None]            # ignore dead fleets

    # First hit per fleet (lowest body index = matches original iteration order)
    any_hit = jnp.any(hit, axis=1)      # [F]
    hit_pid = jnp.argmax(hit, axis=1)   # [F]  (argmax returns 0 when all False — masked by any_hit)

    # Sun collision — minimum distance from fleet path to sun center
    sun_dist = _point_to_segment_dist(CENTER, CENTER, old_f, new_f)   # [F]
    sun_hit  = sun_dist < SUN_RADIUS

    # Out-of-bounds
    oob = ~(
        (new_f[:, 0] >= 0.0) & (new_f[:, 0] <= BOARD_SIZE) &
        (new_f[:, 1] >= 0.0) & (new_f[:, 1] <= BOARD_SIZE)
    )

    # Fleets that survive this tick
    # Priority: planet collision first (credit hit before dying to OOB/sun)
    fleet_survives = live & ~any_hit & ~oob & ~sun_hit

    fleet_coords_new = jnp.where(fleet_survives[:, None], new_f, state.fleet_coords)
    fleet_owners_new = jnp.where(fleet_survives, state.fleet_owners, FLEET_EMPTY)
    fleet_ships_new  = jnp.where(fleet_survives, state.fleet_ships, 0.0)

    # ------------------------------------------------------------------ #
    # 5. Apply planet movement                                             #
    # ------------------------------------------------------------------ #
    # Comet first-appearance: when a comet just spawned, also set its ships
    comet_age = state.step - params.comet_spawn_steps   # [B]
    just_spawned = params.is_comet_slot & (comet_age == 0)

    planet_coords_new = jnp.where(active[:, None], new_coords, state.planet_coords)

    # Set ship count on comet slots that just spawned
    planet_ships_spawned = jnp.where(just_spawned, params.planet_ships_at_spawn, state.planet_ships)

    # ------------------------------------------------------------------ #
    # 6. Combat resolution                                                 #
    # ------------------------------------------------------------------ #
    planet_owners_c = state.planet_owners
    planet_ships_c  = planet_ships_spawned

    # exact_hit[f, b]: fleet f hit planet b as its FIRST collision
    body_idx   = jnp.arange(MAX_BODIES, dtype=jnp.int32)
    exact_hit  = any_hit[:, None] & (body_idx[None, :] == hit_pid[:, None])  # [F, B]

    # Ships arriving per player per planet: [B, MAX_AGENTS]
    def ships_for_player(pl):
        player_mask = (state.fleet_owners == pl)[:, None]  # [F, 1]
        return jnp.sum(
            jnp.where(player_mask & exact_hit, state.fleet_ships[:, None], 0.0),
            axis=0,
        )  # [B]

    ships_by_player = jnp.stack(
        [ships_for_player(pl) for pl in range(MAX_AGENTS)], axis=-1
    )  # [B, MAX_AGENTS]

    # Top-2 rule
    top_val   = jnp.max(ships_by_player, axis=-1)           # [B]
    top_owner = jnp.argmax(ships_by_player, axis=-1)         # [B]
    one_hot   = jax.nn.one_hot(top_owner, MAX_AGENTS, dtype=jnp.bool_)
    second    = jnp.max(jnp.where(one_hot, -1.0, ships_by_player), axis=-1)  # [B]
    survivor  = jnp.maximum(0.0, top_val - second)           # [B] — tie → 0
    surv_own  = jnp.where(survivor > 0.0, top_owner,
                          jnp.full_like(top_owner, OWNER_NEUTRAL))  # [B]

    any_combat  = jnp.any(exact_hit, axis=0)                 # [B]
    same_owner  = (surv_own == planet_owners_c) & any_combat
    reinforce   = same_owner & (survivor > 0.0)
    damage      = ~same_owner & (survivor > 0.0) & any_combat

    planet_ships_c  = jnp.where(reinforce, planet_ships_c + survivor, planet_ships_c)
    planet_ships_c  = jnp.where(damage,    planet_ships_c - survivor, planet_ships_c)
    captured        = damage & (planet_ships_c < 0.0)
    planet_owners_c = jnp.where(captured, surv_own,         planet_owners_c)
    planet_ships_c  = jnp.where(captured, -planet_ships_c,  planet_ships_c)

    # Deactivate comet slots whose lifespan just ran out this tick
    comet_expired = params.is_comet_slot & (comet_age >= params.body_lifespans)
    planet_owners_c = jnp.where(comet_expired, OWNER_INACTIVE, planet_owners_c)

    # ------------------------------------------------------------------ #
    # 7. Return new state                                                  #
    # ------------------------------------------------------------------ #
    return EnvState(
        planet_owners    = planet_owners_c,
        planet_coords    = planet_coords_new,
        planet_ships     = planet_ships_c,
        fleet_owners     = fleet_owners_new,
        fleet_coords     = fleet_coords_new,
        fleet_angles     = state.fleet_angles,
        fleet_ships      = fleet_ships_new,
        next_fleet_slot  = state.next_fleet_slot,
        step             = state.step + 1,
    )


# ---------------------------------------------------------------------------
# TERMINATION AND REWARDS
# ---------------------------------------------------------------------------

def is_done(state: EnvState, params: EnvParams) -> Array:
    """True if the episode has ended."""
    timeout = state.step >= params.episode_steps - 2

    active = _active_bodies(state, params)
    live_fleet = state.fleet_owners >= 0

    def has_presence(pl):
        owns_planet = jnp.any((state.planet_owners == pl) & active)
        owns_fleet  = jnp.any(state.fleet_owners == pl)
        return owns_planet | owns_fleet

    num_alive = jnp.sum(
        jnp.stack([has_presence(pl) for pl in range(MAX_AGENTS)]).astype(jnp.int32)
    )
    elimination = num_alive <= 1
    return timeout | elimination


def compute_rewards(state: EnvState, params: EnvParams) -> Array:
    """Terminal rewards: [MAX_AGENTS] float32.  +1 for winner(s), -1 otherwise."""
    active     = _active_bodies(state, params)
    live_fleet = state.fleet_owners >= 0

    def score(pl):
        p = jnp.sum(jnp.where((state.planet_owners == pl) & active, state.planet_ships, 0.0))
        f = jnp.sum(jnp.where(state.fleet_owners == pl, state.fleet_ships, 0.0))
        return p + f

    scores    = jnp.stack([score(pl) for pl in range(MAX_AGENTS)])
    max_score = jnp.max(scores)
    winner    = (scores == max_score) & (max_score > 0.0)
    return jnp.where(winner, 1.0, -1.0)


# ---------------------------------------------------------------------------
# SETUP  (episode initialisation)
# ---------------------------------------------------------------------------

def setup(
    rng_key: Array,
    num_active_agents: int = 2,
    ship_speed: float = DEFAULT_SHIP_SPEED,
    episode_steps: int = DEFAULT_EPISODE_STEPS,
    comet_speed: float = DEFAULT_COMET_SPEED,
) -> tuple[EnvState, EnvParams]:
    """
    Initialise a new episode.  Fully JAX-native: safe to jit/vmap.

    Args:
        rng_key:           JAX PRNG key.
        num_active_agents: 2 or 4.  In a 2-player game, agents 2/3 never get
                           home planets and are effectively spectators.
        ship_speed:        Maximum fleet speed (default 6.0).
        episode_steps:     Episode length (default 500).
        comet_speed:       Comet path arc-length per step (default 4.0).

    Returns:
        (EnvState, EnvParams) — pass both to step() every tick.
    """
    k0, k1, k2, k3, k4 = jrandom.split(rng_key, 5)

    # ------------------------------------------------------------------
    # Base planets  [MAX_PLANETS_BASE, 7]
    # Columns: [id, owner, y, x, r, ships, production]  (note: y before x)
    # ------------------------------------------------------------------
    base_planets = generate_planets(k0)   # from orbit_wars_jax.py

    # ------------------------------------------------------------------
    # Angular velocity
    # ------------------------------------------------------------------
    angular_velocity = jrandom.uniform(k1, (), minval=0.025, maxval=0.05)

    # ------------------------------------------------------------------
    # Pre-generate all comet waves
    # vmap over the 5 spawn steps, one rng key per wave
    # ------------------------------------------------------------------
    wave_keys = jrandom.split(k2, NUM_COMET_WAVES)

    all_paths, all_lifespans, all_success = jax.vmap(
        lambda spawn_step, wave_key: generate_comet_paths(
            base_planets,
            angular_velocity,
            spawn_step,
            rng=wave_key,
            comet_speed=comet_speed,
        )
    )(COMET_SPAWN_STEPS, wave_keys)
    # all_paths:    [NUM_COMET_WAVES, 4, MAX_PATH_LEN, 2]
    # all_lifespans:[NUM_COMET_WAVES]
    # all_success:  [NUM_COMET_WAVES]  bool

    # Pre-roll comet ship counts (min of 4 draws, one per wave)
    def roll_comet_ships(rng):
        keys  = jrandom.split(rng, 4)
        rolls = jax.vmap(lambda k: jrandom.randint(k, (), 1, 100))(keys)
        return jnp.min(rolls).astype(jnp.float32)

    ship_keys       = jrandom.split(k3, NUM_COMET_WAVES)
    comet_ships_per_wave = jax.vmap(roll_comet_ships)(ship_keys)   # [NUM_COMET_WAVES]

    # ------------------------------------------------------------------
    # Build the flat comet_paths array [TOTAL_COMETS, MAX_PATH_LEN, 2]
    # and per-slot metadata arrays [MAX_BODIES]
    # ------------------------------------------------------------------
    # Layout: comet slot i = wave (i // MAX_COMETS_PER_WAVE), quadrant (i % MAX_COMETS_PER_WAVE)
    spawn_steps_raw = jnp.repeat(COMET_SPAWN_STEPS, MAX_COMETS_PER_WAVE)   # [TOTAL_COMETS]
    lifespan_raw    = jnp.repeat(all_lifespans, MAX_COMETS_PER_WAVE)        # [TOTAL_COMETS]
    success_raw     = jnp.repeat(all_success, MAX_COMETS_PER_WAVE)          # [TOTAL_COMETS]
    ships_raw       = jnp.repeat(comet_ships_per_wave, MAX_COMETS_PER_WAVE) # [TOTAL_COMETS]

    comet_paths_flat = all_paths.reshape(TOTAL_COMETS, MAX_PATH_LEN, 2)    # [TC, L, 2]

    # Zero out paths for failed waves
    comet_paths_flat = jnp.where(success_raw[:, None, None], comet_paths_flat, 0.0)
    lifespan_raw     = jnp.where(success_raw, lifespan_raw, jnp.zeros_like(lifespan_raw))
    ships_raw        = jnp.where(success_raw, ships_raw, jnp.zeros_like(ships_raw))

    # ------------------------------------------------------------------
    # Assemble full [MAX_BODIES] arrays
    # base slots [0 .. MAX_PLANETS_BASE), comet slots [MAX_PLANETS_BASE .. MAX_BODIES)
    # ------------------------------------------------------------------

    # base_planets columns: [id, owner, y, x, r, ships, prod]
    # We convert to (x, y) coords for consistency everywhere else
    base_y    = base_planets[:, 2]
    base_x    = base_planets[:, 3]
    base_r    = base_planets[:, 4]
    base_ships= base_planets[:, 5]
    base_prod = base_planets[:, 6]
    base_valid= base_planets[:, 0] != -1.0   # id == -1 means empty slot

    # Initial coords (x, y) for base planets
    base_xy   = jnp.stack([base_x, base_y], axis=-1)   # [MAX_PLANETS_BASE, 2]

    # Orbital geometry
    dx  = base_x - CENTER
    dy  = base_y - CENTER
    orb_r     = jnp.sqrt(dx ** 2 + dy ** 2)
    init_angle= jnp.arctan2(dy, dx)

    is_orbit_base  = (orb_r + base_r) < ROTATION_RADIUS_LIMIT
    is_static_base = base_valid & ~is_orbit_base
    is_orbit_base  = base_valid & is_orbit_base

    # Comet slots — static geometry defaults (coords set dynamically)
    comet_r    = jnp.full(TOTAL_COMETS, COMET_RADIUS)
    comet_prod = jnp.full(TOTAL_COMETS, float(COMET_PRODUCTION))
    comet_xy   = jnp.full((TOTAL_COMETS, 2), -99.0)   # off-board until spawned

    # Full arrays
    all_r        = jnp.concatenate([base_r,    comet_r],    axis=0)
    all_prod     = jnp.concatenate([base_prod, comet_prod], axis=0)
    all_init_xy  = jnp.concatenate([base_xy,   comet_xy],   axis=0)
    all_orb_r    = jnp.concatenate([orb_r,     jnp.zeros(TOTAL_COMETS)], axis=0)
    all_init_ang = jnp.concatenate([init_angle,jnp.zeros(TOTAL_COMETS)], axis=0)

    is_static_all  = jnp.concatenate([is_static_base, jnp.zeros(TOTAL_COMETS, dtype=jnp.bool_)], axis=0)
    is_orbit_all   = jnp.concatenate([is_orbit_base,  jnp.zeros(TOTAL_COMETS, dtype=jnp.bool_)], axis=0)
    is_comet_all   = jnp.concatenate([jnp.zeros(MAX_PLANETS_BASE, dtype=jnp.bool_),
                                       jnp.ones(TOTAL_COMETS,     dtype=jnp.bool_)],  axis=0)

    base_spawn_steps = jnp.zeros(MAX_PLANETS_BASE, dtype=jnp.int32)
    all_spawn_steps  = jnp.concatenate([base_spawn_steps, spawn_steps_raw], axis=0)

    base_lifespans   = jnp.full(MAX_PLANETS_BASE, 999999, dtype=jnp.int32)
    all_lifespans_f  = jnp.concatenate([base_lifespans, lifespan_raw.astype(jnp.int32)], axis=0)

    base_ships_spawn = base_ships   # not used for base planets but must have same shape
    all_ships_spawn  = jnp.concatenate([base_ships_spawn, ships_raw], axis=0)

    # ------------------------------------------------------------------
    # Home planet assignment
    # ------------------------------------------------------------------
    num_valid_groups = jnp.sum(base_valid).astype(jnp.int32) // 4
    home_group = jrandom.randint(k4, (), 0, MAX_PLANET_GROUPS) % jnp.maximum(num_valid_groups, 1)
    base_idx   = home_group * 4   # first slot of the chosen group

    body_range = jnp.arange(MAX_BODIES, dtype=jnp.int32)

    # Initial owners: neutral for valid base planets, inactive for invalid/comet
    init_owners = jnp.where(
        jnp.concatenate([base_valid, jnp.zeros(TOTAL_COMETS, dtype=jnp.bool_)], axis=0),
        OWNER_NEUTRAL,
        OWNER_INACTIVE,
    )

    # Assign home planets based on num_active_agents
    # 2-player: base_idx → player 0, base_idx+3 → player 1
    # 4-player: base_idx+0..3 → players 0..3
    is_home_0 = (body_range == base_idx)
    is_home_1 = (body_range == base_idx + 1)
    is_home_2 = (body_range == base_idx + 2)
    is_home_3 = (body_range == base_idx + 3)

    home_owners = init_owners
    home_owners = jnp.where(is_home_0, 0, home_owners)
    if num_active_agents == 2:
        home_owners = jnp.where(is_home_3, 1, home_owners)
    else:  # 4-player
        home_owners = jnp.where(is_home_1, 1, home_owners)
        home_owners = jnp.where(is_home_2, 2, home_owners)
        home_owners = jnp.where(is_home_3, 3, home_owners)

    # Home planets start with 10 ships
    is_any_home = is_home_0 | is_home_3 | (is_home_1 | is_home_2 if num_active_agents == 4 else jnp.zeros(MAX_BODIES, dtype=jnp.bool_))
    init_ships  = jnp.concatenate([base_ships, jnp.full(TOTAL_COMETS, 0.0)], axis=0)
    init_ships  = jnp.where(is_any_home, 10.0, init_ships)

    # ------------------------------------------------------------------
    # Build EnvParams and EnvState
    # ------------------------------------------------------------------
    params = EnvParams(
        planet_radii          = all_r,
        planet_prod           = all_prod,
        initial_planet_coords = all_init_xy,
        planet_orbital_radii  = all_orb_r,
        planet_initial_angles = all_init_ang,
        is_static             = is_static_all,
        is_orbiting           = is_orbit_all,
        is_comet_slot         = is_comet_all,
        comet_paths           = comet_paths_flat,
        comet_spawn_steps     = all_spawn_steps,
        body_lifespans        = all_lifespans_f,
        planet_ships_at_spawn = all_ships_spawn,
        angular_velocity      = angular_velocity,
        ship_speed            = jnp.array(ship_speed),
        episode_steps         = episode_steps,
        num_active_agents     = num_active_agents,
    )

    state = EnvState(
        planet_owners    = home_owners,
        planet_coords    = all_init_xy,
        planet_ships     = init_ships,
        fleet_owners     = jnp.full(MAX_FLEETS, FLEET_EMPTY, dtype=jnp.int32),
        fleet_coords     = jnp.zeros((MAX_FLEETS, 2)),
        fleet_angles     = jnp.zeros(MAX_FLEETS),
        fleet_ships      = jnp.zeros(MAX_FLEETS),
        next_fleet_slot  = jnp.array(0, dtype=jnp.int32),
        step             = jnp.array(0, dtype=jnp.int32),
    )

    return state, params


# ---------------------------------------------------------------------------
# GUI / RENDERER ADAPTER
# ---------------------------------------------------------------------------

def to_observation(state: EnvState, params: EnvParams) -> dict:
    """
    Convert JAX state to the dict format the kaggle_environments JS renderer
    expects.  Call this each step to build a replay for visualisation.
    """
    active = _active_bodies(state, params)

    # Materialise to numpy for the renderer
    import numpy as np
    owners  = np.asarray(state.planet_owners)
    coords  = np.asarray(state.planet_coords)
    ships   = np.asarray(state.planet_ships)
    radii   = np.asarray(params.planet_radii)
    prods   = np.asarray(params.planet_prod)
    active_np = np.asarray(active)

    planets_out = []
    for i in range(MAX_BODIES):
        if not active_np[i] or owners[i] == OWNER_INACTIVE:
            continue
        planets_out.append([
            int(i),           # id
            int(owners[i]),   # owner
            float(coords[i, 0]),  # x
            float(coords[i, 1]),  # y
            float(radii[i]),
            float(ships[i]),
            float(prods[i]),
        ])

    f_owners_np = np.asarray(state.fleet_owners)
    f_coords_np = np.asarray(state.fleet_coords)
    f_angles_np = np.asarray(state.fleet_angles)
    f_ships_np  = np.asarray(state.fleet_ships)

    fleets_out = []
    for i in range(MAX_FLEETS):
        if f_owners_np[i] < 0:
            continue
        fleets_out.append([
            int(i),
            int(f_owners_np[i]),
            float(f_coords_np[i, 0]),
            float(f_coords_np[i, 1]),
            float(f_angles_np[i]),
            -1,                    # from_planet_id (not tracked in JAX sim)
            float(f_ships_np[i]),
        ])

    # Comet groups for tail rendering
    comet_age_np = np.asarray(state.step - params.comet_spawn_steps)
    spawn_steps_np = np.asarray(params.comet_spawn_steps)
    lifespans_np   = np.asarray(params.body_lifespans)
    paths_np       = np.asarray(params.comet_paths)
    is_comet_np    = np.asarray(params.is_comet_slot)

    comets_out = []
    comet_pids = []

    for wave_i in range(NUM_COMET_WAVES):
        wave_slot_start = MAX_PLANETS_BASE + wave_i * MAX_COMETS_PER_WAVE
        # Check if any slot of this wave is active
        age_0 = int(comet_age_np[wave_slot_start])
        lifespan_0 = int(lifespans_np[wave_slot_start])
        if lifespan_0 == 0 or age_0 < 0 or age_0 >= lifespan_0:
            continue  # wave never generated or fully expired

        group_pids = []
        group_paths = []
        for q in range(MAX_COMETS_PER_WAVE):
            body_slot = wave_slot_start + q
            comet_idx = wave_i * MAX_COMETS_PER_WAVE + q
            pid = body_slot
            group_pids.append(pid)
            comet_pids.append(pid)
            group_paths.append(paths_np[comet_idx].tolist())

        comets_out.append({
            "planet_ids": group_pids,
            "paths": group_paths,
            "path_index": age_0,
        })

    return {
        "step": int(state.step),
        "angular_velocity": float(params.angular_velocity),
        "planets": planets_out,
        "fleets": fleets_out,
        "comets": comets_out,
        "comet_planet_ids": comet_pids,
    }


# ---------------------------------------------------------------------------
# WRAPPER CLASS  (FastOrbitWarsSimulator-compatible API)
# ---------------------------------------------------------------------------

class JAXSimWrapper:
    """
    Thin stateful wrapper around the pure JAX functions.
    API mirrors FastOrbitWarsSimulator so it can drop into existing tooling
    and feed the kaggle_environments JS renderer.

    For RL training, prefer using setup()/step() directly with jax.vmap.
    """

    def __init__(
        self,
        num_agents: int = 2,
        seed: int | None = None,
        ship_speed: float = DEFAULT_SHIP_SPEED,
        episode_steps: int = DEFAULT_EPISODE_STEPS,
        comet_speed: float = DEFAULT_COMET_SPEED,
    ):
        import random as pyrand
        self.num_agents   = num_agents
        self.ship_speed   = ship_speed
        self.episode_steps= episode_steps
        self.comet_speed  = comet_speed
        self._seed        = seed if seed is not None else pyrand.randrange(2 ** 31)
        self.state:  EnvState  | None = None
        self.params: EnvParams | None = None
        self._done        = False
        self._reward      = [0.0] * num_agents
        self.reset(self._seed)

    @property
    def done(self) -> bool:
        return self._done

    def reset(self, seed: int | None = None) -> list[dict]:
        import random as pyrand
        self._seed  = seed if seed is not None else pyrand.randrange(2 ** 31)
        key         = jrandom.PRNGKey(self._seed)
        self.state, self.params = setup(
            key,
            num_active_agents = self.num_agents,
            ship_speed        = self.ship_speed,
            episode_steps     = self.episode_steps,
            comet_speed       = self.comet_speed,
        )
        self._done   = False
        self._reward = [0.0] * self.num_agents
        return self.observations()

    def observations(self) -> list[dict]:
        obs = to_observation(self.state, self.params)
        return [dict(obs, player=i) for i in range(self.num_agents)]

    def step(self, actions: list) -> tuple[list[dict], bool, list[float]]:
        """
        actions: list of per-player move lists.
        Each player's list is [[planet_id, angle, ships], ...].
        """
        import numpy as np

        env_action = _encode_actions(actions, self.num_agents)
        self.state  = step(self.state, self.params, env_action)
        self._done  = bool(is_done(self.state, self.params))
        if self._done:
            r = compute_rewards(self.state, self.params)
            self._reward = [float(r[i]) for i in range(self.num_agents)]
        return self.observations(), self._done, list(self._reward)

    def snapshot(self) -> dict:
        obs = to_observation(self.state, self.params)
        obs["done"]   = self._done
        obs["reward"] = self._reward
        return obs


def _encode_actions(
    player_actions: list,
    num_agents: int,
) -> EnvAction:
    """Convert list-of-list agent actions to EnvAction arrays."""
    import numpy as np

    planet_ids = np.full((MAX_AGENTS, MAX_MOVES_PER_STEP), -1, dtype=np.int32)
    angles     = np.zeros((MAX_AGENTS, MAX_MOVES_PER_STEP), dtype=np.float32)
    ships      = np.zeros((MAX_AGENTS, MAX_MOVES_PER_STEP), dtype=np.int32)

    for player_i, moves in enumerate(player_actions):
        if player_i >= num_agents:
            break
        for move_j, move in enumerate(moves):
            if move_j >= MAX_MOVES_PER_STEP:
                break
            if len(move) != 3:
                continue
            pid, angle, n_ships = move
            planet_ids[player_i, move_j] = int(pid)
            angles[player_i, move_j]     = float(angle)
            ships[player_i, move_j]      = int(n_ships)

    return EnvAction(
        planet_ids = jnp.asarray(planet_ids),
        angles     = jnp.asarray(angles),
        ships      = jnp.asarray(ships),
    )
