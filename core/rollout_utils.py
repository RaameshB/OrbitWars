import jax.numpy as jnp
from core.orbit_wars_jax import MAX_FLEETS, MAX_COMET_PATH_LEN, TOTAL_COMETS


def active_body_mask(state, params):
    """True for each body slot that is currently active (valid planet or live comet).

    Uses params flags rather than coordinate sentinels. Failed-to-generate planet slots
    sit at (-1,-1), not (-99,-99), so a coordinate threshold would include them.
    """
    ages = state.step - params.comet_spawn_steps  # [..., 60]
    comet_active = params.is_comet & (ages >= 0) & (ages < params.body_lifespans)
    return params.is_static_planet | params.is_orbiting_planet | comet_active


def calculate_intercept_angle(state, params, ships):
    """Compute world-frame launch angle from each planet to intercept each target body.

    ships: [B, 60, 60] float ship counts — cast to int32 internally so predicted speed
    matches the int-truncated speed used in step(). Float speeds are 20-35% higher for
    small ship counts, causing late arrivals and systematic misses on orbiting targets.

    Returns: [B, 60, 60] angles in radians.
    """
    B = ships.shape[0]
    safe_ships = jnp.maximum(ships.astype(jnp.int32), 1)
    raw_speed = 1.0 + (params.ship_speed[..., None, None] - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
    v_fleet = jnp.minimum(raw_speed, params.ship_speed[..., None, None])

    ts = jnp.arange(1, 151, dtype=jnp.float32)
    future_steps = state.step[..., None] + ts  # [B, 150]

    pr = params.planet_orbital_radii
    initial_angles = params.planet_initial_angles
    current_angles = initial_angles[..., None] + (params.angular_velocity[..., None, None] * future_steps[:, None, :])
    orbit_x = 50.0 + pr[..., None] * jnp.cos(current_angles)
    orbit_y = 50.0 + pr[..., None] * jnp.sin(current_angles)
    orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1)  # [B, 60, 150, 2]

    ages = future_steps[:, None, :] - params.comet_spawn_steps[..., None]
    comet_ages = ages[:, -TOTAL_COMETS:, :]
    safe_comet_ages = jnp.clip(comet_ages, 0, MAX_COMET_PATH_LEN - 1).astype(jnp.int32)

    b_idx = jnp.arange(B)[:, None, None]
    c_idx = jnp.arange(TOTAL_COMETS)[None, :, None]
    idxed_comet_locations = params.comet_paths[b_idx, c_idx, safe_comet_ages, :]

    n_planet_slots = 60 - TOTAL_COMETS
    padded_comet_coords = jnp.concatenate([
        jnp.zeros((B, n_planet_slots, 150, 2)),
        idxed_comet_locations,
    ], axis=1)

    static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B, 60, 150, 2))

    future_coords = jnp.where(
        params.is_orbiting_planet[..., None, None], orbit_coords,
        jnp.where(params.is_comet[..., None, None], padded_comet_coords, static_coords)
    )

    tfc = future_coords[:, None, :, :, :]  # [B, 1, 60, 150, 2] broadcasts to [B, 60, 60, 150, 2]

    src = state.planet_coords[:, :, None, None, :].astype(jnp.bfloat16)
    tfc_bf16 = tfc.astype(jnp.bfloat16)
    dists = jnp.sqrt(((tfc_bf16[..., 0] - src[..., 0])**2 + (tfc_bf16[..., 1] - src[..., 1])**2).astype(jnp.float32) + 1e-8)

    R_src = params.planet_radii[:, :, None, None]
    R_tgt = params.planet_radii[:, None, :, None]

    travel_dist = R_src + 0.1 + v_fleet[..., None] * ts[None, None, None, :]
    req_dist = dists - R_tgt

    can_reach = travel_dist >= req_dist
    intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 149)

    idx = intercept_t_idx[..., None]
    ic_x = jnp.take_along_axis(tfc[..., 0], idx, axis=-1)[..., 0]
    ic_y = jnp.take_along_axis(tfc[..., 1], idx, axis=-1)[..., 0]

    src_x = state.planet_coords[..., 0][:, :, None]
    src_y = state.planet_coords[..., 1][:, :, None]

    return jnp.arctan2(ic_y - src_y, ic_x - src_x)


def build_obs_arrays(state, params, pid, num_players):
    """Build observation arrays for a single (unbatched) game state.

    Returns (planets [60,7], fleets [MAX_FLEETS,6], planet_mask [60], fleet_mask [MAX_FLEETS]).
    """
    theta = -pid * (2 * jnp.pi / num_players)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    rel_p_owner = jnp.where(state.planet_owners == pid, 1.0,
                            jnp.where(state.planet_owners == -1, 0.0, -1.0))
    dx = state.planet_coords[:, 0] - 50.0
    dy = state.planet_coords[:, 1] - 50.0
    rot_x = dx * cos_t - dy * sin_t + 50.0
    rot_y = dx * sin_t + dy * cos_t + 50.0
    planets = jnp.stack([
        jnp.arange(60).astype(jnp.float32),
        rel_p_owner,
        rot_x,
        rot_y,
        params.planet_radii,
        state.planet_ships.astype(jnp.float32),
        params.planet_prod.astype(jnp.float32),
    ], axis=-1)  # [60, 7]

    rel_f_owner = jnp.where(state.fleet_owners == pid, 1.0,
                            jnp.where(state.fleet_owners == -1, 0.0, -1.0))
    fdx = state.fleet_coords[:, 0] - 50.0
    fdy = state.fleet_coords[:, 1] - 50.0
    frot_x = fdx * cos_t - fdy * sin_t + 50.0
    frot_y = fdx * sin_t + fdy * cos_t + 50.0
    fleets = jnp.stack([
        jnp.arange(MAX_FLEETS).astype(jnp.float32),
        rel_f_owner,
        state.fleet_angles + theta,
        frot_x,
        frot_y,
        state.fleet_ship_count.astype(jnp.float32),
    ], axis=-1)  # [MAX_FLEETS, 6]

    p_mask = state.planet_owners != -1
    f_mask = state.fleet_owners != -1
    return planets, fleets, p_mask, f_mask