import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 1. Dummy actions
    content = content.replace("dummy_actions = jnp.zeros((60, 61), dtype=jnp.float32)", "dummy_actions = jnp.zeros((60, 62), dtype=jnp.float32)")
    
    # 2. is_self mask (We keep it as 61, because it masks out the 61 target bins. Wait, the target bins are 61 now! So is_self = jnp.arange(60) == jnp.arange(61) is still correct!)
    
    # 3. Revert calculate_intercept_angle
    old_calc = """    def calculate_intercept_angle(state, params, ships):
        ships = ships[..., :60]
        B = ships.shape[0]"""
    new_calc = """    def calculate_intercept_angle(state, params, ships):
        B = ships.shape[0]"""
    content = content.replace(old_calc, new_calc)
    
    # Remove raycast math from calculate_intercept_angle
    import re
    # We find the return statement of the old standard angles
    # and rip out everything between it and the end of the function.
    
    # We'll just replace the whole calculate_intercept_angle body since we know what it should look like.
    old_func = re.search(r"    def calculate_intercept_angle\(state, params, ships\):.*?return jnp\.concatenate\[standard_angles, deep_space_angles\], axis=-1\)", content, flags=re.DOTALL)
    
    if old_func:
        new_func = """    def calculate_intercept_angle(state, params, ships):
        B = ships.shape[0]
        target_ids = jnp.broadcast_to(jnp.arange(60)[None, None, :], (B, 60, 60))
        # 1. Fleet speeds [B, 60, 60]
        safe_ships = jnp.maximum(ships, 1)
        raw_speed = 1.0 + (params.ship_speed[..., None, None] - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
        v_fleet = jnp.minimum(raw_speed, params.ship_speed[..., None, None])
        
        # 2. Simulate future coordinates for t=1..50
        ts = jnp.arange(1, 51, dtype=jnp.float32)
        future_steps = state.step[..., None] + ts # [B, 50]
        
        pr = params.planet_orbital_radii
        initial_angles = params.planet_initial_angles
        current_angles = initial_angles[..., None] + (params.angular_velocity[..., None, None] * future_steps[:, None, :])
        orbit_x = 50.0 + pr[..., None] * jnp.cos(current_angles)
        orbit_y = 50.0 + pr[..., None] * jnp.sin(current_angles)
        orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1) # [B, 60, 50, 2]
        
        ages = future_steps[:, None, :] - params.comet_spawn_steps[..., None]
        comet_ages = ages[:, -20:, :]
        safe_comet_ages = jnp.clip(comet_ages, 0, 150 - 1).astype(jnp.int32)
        
        B = safe_comet_ages.shape[0]
        b_idx = jnp.arange(B)[:, None, None]
        c_idx = jnp.arange(20)[None, :, None]
        idxed_comet_locations = params.comet_paths[b_idx, c_idx, safe_comet_ages, :]
        
        padded_comet_coords = jnp.zeros((B, 60, 50, 2))
        padded_comet_coords = padded_comet_coords.at[:, -20:, :, :].set(idxed_comet_locations)
        
        static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B, 60, 50, 2))
        
        future_coords = jnp.where(
            params.is_orbiting_planet[..., None, None], orbit_coords,
            jnp.where(
                params.is_comet[..., None, None], padded_comet_coords,
                static_coords
            )
        )
        
        # 3. Extract future coords for chosen targets [B, 60, 4, 50, 2]
        tfc = future_coords[b_idx, target_ids]
        
        # 4. Calculate Distance and Intercept Time
        src = state.planet_coords[:, :, None, None, :]
        dists = jnp.sqrt((tfc[..., 0] - src[..., 0])**2 + (tfc[..., 1] - src[..., 1])**2)
        travel_dist = v_fleet[..., None] * ts[None, None, None, :]
        
        can_reach = travel_dist >= dists
        intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 49)
        
        # 5. Extract exact physical coordinates at intercept time
        idx = intercept_t_idx[..., None]
        ic_x = jnp.take_along_axis(tfc[..., 0], idx, axis=-1)[..., 0]
        ic_y = jnp.take_along_axis(tfc[..., 1], idx, axis=-1)[..., 0]
        
        src_x = state.planet_coords[..., 0][:, :, None]
        src_y = state.planet_coords[..., 1][:, :, None]
        
        return jnp.arctan2(ic_y - src_y, ic_x - src_x)"""
        
        content = content.replace(old_func.group(0), new_func)

    # 4. run_1v1 replacements
    content = content.replace("ships_p0 = logits_to_action(logits_p0, state.planet_ships)", "ships_p0, ds_angle_p0 = logits_to_action(logits_p0, state.planet_ships)")
    content = content.replace("ships_p1 = logits_to_action(logits_p1, state.planet_ships)", "ships_p1, ds_angle_p1 = logits_to_action(logits_p1, state.planet_ships)")
    
    content = content.replace("angles_p0 = calculate_intercept_angle(state, params, ships_p0)", "intercept_p0 = calculate_intercept_angle(state, params, ships_p0[..., :60])\n            angles_p0 = jnp.concatenate([intercept_p0, ds_angle_p0], axis=-1)")
    content = content.replace("angles_p1 = calculate_intercept_angle(state, params, ships_p1)", "intercept_p1 = calculate_intercept_angle(state, params, ships_p1[..., :60])\n            angles_p1 = jnp.concatenate([intercept_p1, ds_angle_p1], axis=-1)")
    
    # 5. run_ffa replacements
    content = content.replace("ships_p2 = logits_to_action(logits_p2, state.planet_ships)", "ships_p2, ds_angle_p2 = logits_to_action(logits_p2, state.planet_ships)")
    content = content.replace("ships_p3 = logits_to_action(logits_p3, state.planet_ships)", "ships_p3, ds_angle_p3 = logits_to_action(logits_p3, state.planet_ships)")
    
    content = content.replace("angles_p2 = calculate_intercept_angle(state, params, ships_p2)", "intercept_p2 = calculate_intercept_angle(state, params, ships_p2[..., :60])\n            angles_p2 = jnp.concatenate([intercept_p2, ds_angle_p2], axis=-1)")
    content = content.replace("angles_p3 = calculate_intercept_angle(state, params, ships_p3)", "intercept_p3 = calculate_intercept_angle(state, params, ships_p3[..., :60])\n            angles_p3 = jnp.concatenate([intercept_p3, ds_angle_p3], axis=-1)")
    
    with open(filepath, 'w') as f:
        f.write(content)

patch_file('train.py')
patch_file('train_tpu.py')
patch_file('train_blackwell.py')
