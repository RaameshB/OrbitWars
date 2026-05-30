import json

with open("eval_local.ipynb", "r") as f:
    nb = json.load(f)

# Find the cell that contains 'def rollout'
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "@jax.jit\ndef rollout(" in source:
            inject_code = """
def calculate_intercept_angle(state, params, ships):
    B = ships.shape[0]
    target_ids = jnp.broadcast_to(jnp.arange(60)[None, None, :], (B, 60, 60))
    # 1. Fleet speeds [B, 60, 60]
    safe_ships = jnp.maximum(ships, 1)
    raw_speed = 1.0 + (params.ship_speed[..., None, None] - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
    v_fleet = jnp.minimum(raw_speed, params.ship_speed[..., None, None])
    
    # 2. Simulate future coordinates for t=1..150
    ts = jnp.arange(1, 151, dtype=jnp.float32)
    future_steps = state.step[..., None] + ts # [B, 150]
    
    pr = params.planet_orbital_radii
    initial_angles = params.planet_initial_angles
    current_angles = initial_angles[..., None] + (params.angular_velocity[..., None, None] * future_steps[:, None, :])
    orbit_x = 500.0 + pr[..., None] * jnp.cos(current_angles)
    orbit_y = 500.0 + pr[..., None] * jnp.sin(current_angles)
    orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1) # [B, 60, 150, 2]
    
    ages = future_steps[:, None, :] - params.comet_spawn_steps[..., None]
    comet_ages = ages[:, -20:, :]
    safe_comet_ages = jnp.clip(comet_ages, 0, 150 - 1).astype(jnp.int32)
    
    B_dim = safe_comet_ages.shape[0]
    b_idx = jnp.arange(B_dim)[:, None, None]
    c_idx = jnp.arange(20)[None, :, None]
    idxed_comet_locations = params.comet_paths[b_idx, c_idx, safe_comet_ages, :]
    
    padded_comet_coords = jnp.zeros((B_dim, 60, 150, 2))
    padded_comet_coords = padded_comet_coords.at[:, -20:, :, :].set(idxed_comet_locations)
    
    static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B_dim, 60, 150, 2))
    
    future_coords = jnp.where(
        params.is_orbiting_planet[..., None, None], orbit_coords,
        jnp.where(
            params.is_comet[..., None, None], padded_comet_coords,
            static_coords
        )
    )
    
    # 3. Extract future coords for chosen targets [B, 60, 4, 150, 2]
    tfc = future_coords[b_idx, target_ids]
    
    # 4. Calculate Distance and Intercept Time
    src = state.planet_coords[:, :, None, None, :]
    dists = jnp.sqrt((tfc[..., 0] - src[..., 0])**2 + (tfc[..., 1] - src[..., 1])**2)
    
    # Account for fleet spawn offset (R_src + 0.1) and target collision radius (R_tgt)
    R_src = params.planet_radii[:, :, None, None]
    R_tgt = params.planet_radii[:, None, :, None]
    
    travel_dist = R_src + 0.1 + v_fleet[..., None] * ts[None, None, None, :]
    req_dist = dists - R_tgt
    
    can_reach = travel_dist >= req_dist
    intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 149)
    
    # 5. Extract exact physical coordinates at intercept time
    idx = intercept_t_idx[..., None]
    ic_x = jnp.take_along_axis(tfc[..., 0], idx, axis=-1)[..., 0]
    ic_y = jnp.take_along_axis(tfc[..., 1], idx, axis=-1)[..., 0]
    
    src_x = state.planet_coords[..., 0][:, :, None]
    src_y = state.planet_coords[..., 1][:, :, None]
    
    return jnp.arctan2(ic_y - src_y, ic_x - src_x)

"""
            # Insert calculate_intercept_angle right before rollout
            new_source = source.replace("@jax.jit\ndef rollout(", inject_code + "\n@jax.jit\ndef rollout(")
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_source.split("\n")][:-1]

with open("eval_local.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

