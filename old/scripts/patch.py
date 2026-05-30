import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # 1. Dummy actions
    content = content.replace("dummy_actions = jnp.zeros((60, 60), dtype=jnp.float32)", "dummy_actions = jnp.zeros((60, 61), dtype=jnp.float32)")
    
    # 2. is_self mask
    content = content.replace("is_self = jnp.arange(60)[None, :, None] == jnp.arange(60)[None, None, :]", "is_self = jnp.arange(60)[None, :, None] == jnp.arange(61)[None, None, :]")
    
    # 3. calculate_intercept_angle
    old_calc = """        src_x = state.planet_coords[..., 0][:, :, None]
        src_y = state.planet_coords[..., 1][:, :, None]
        
        return jnp.arctan2(ic_y - src_y, ic_x - src_x)"""
    
    new_calc = """        src_x = state.planet_coords[..., 0][:, :, None]
        src_y = state.planet_coords[..., 1][:, :, None]
        
        standard_angles = jnp.arctan2(ic_y - src_y, ic_x - src_x)
        
        # 6. Deep Space 32-angle sweep (Raycast)
        N = 32
        test_angles = jnp.linspace(0, 2*jnp.pi, N, endpoint=False) # [32]
        ray_dx = jnp.cos(test_angles) # [32]
        ray_dy = jnp.sin(test_angles) # [32]
        
        P = state.planet_coords
        W = P[:, None, :, :] - P[:, :, None, :] # [B, src, tgt, 2]
        
        W_exp = W[..., None, :]
        V_exp = jnp.stack([ray_dx, ray_dy], axis=-1)[None, None, None, :, :]
        
        t = jnp.sum(W_exp * V_exp, axis=-1) # [B, 60, 60, 32]
        
        W_sq = jnp.sum(W**2, axis=-1)[..., None]
        d_sq = W_sq - t**2 # [B, 60, 60, 32]
        
        R = params.planet_radii[None, None, :, None]
        hit = (t > 0) & (d_sq < R**2)
        
        x = P[..., 0, None] # [B, 60, 1]
        y = P[..., 1, None] # [B, 60, 1]
        tx = jnp.where(ray_dx > 0, (100.0 - x) / (ray_dx + 1e-8), -x / (ray_dx - 1e-8))
        ty = jnp.where(ray_dy > 0, (100.0 - y) / (ray_dy + 1e-8), -y / (ray_dy - 1e-8))
        t_bounds = jnp.minimum(tx, ty) # [B, 60, 32]
        
        hit_dist = jnp.where(hit, t - jnp.sqrt(jnp.maximum(0.0, R**2 - d_sq)), t_bounds[:, :, None, :])
        
        is_self_ray = jnp.arange(60)[None, :, None, None] == jnp.arange(60)[None, None, :, None]
        hit_dist = jnp.where(is_self_ray, t_bounds[:, :, None, :], hit_dist)
        
        min_hit_dist = jnp.min(hit_dist, axis=2) # [B, 60, 32]
        
        best_angle_idx = jnp.argmax(min_hit_dist, axis=-1) # [B, 60]
        deep_space_angles = test_angles[best_angle_idx][..., None] # [B, 60, 1]
        
        return jnp.concatenate([standard_angles, deep_space_angles], axis=-1)"""
    
    content = content.replace(old_calc, new_calc)
    
    with open(filepath, 'w') as f:
        f.write(content)
        
patch_file('train_tpu.py')
patch_file('train_blackwell.py')
