import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the ships bug
    old_def = "def calculate_intercept_angle(state, params, ships):"
    new_def = "def calculate_intercept_angle(state, params, ships):\n        ships = ships[..., :60]"
    content = content.replace(old_def, new_def)
    
    # Implement blurred annulus
    old_raycast = """        W_sq = jnp.sum(W**2, axis=-1)[..., None]
        d_sq = W_sq - t**2 # [B, 60, 60, 32]
        
        R = params.planet_radii[None, None, :, None]
        hit = (t > 0) & (d_sq < R**2)"""
        
    new_raycast = """        W_sq = jnp.sum(W**2, axis=-1)[..., None]
        d_sq = W_sq - t**2 # [B, 60, 60, 32]
        
        R = params.planet_radii[None, None, :, None]
        
        # Static Planet hit
        static_hit = (t > 0) & (d_sq < R**2)
        static_hit_dist = t - jnp.sqrt(jnp.maximum(0.0, R**2 - d_sq))
        
        # Blurred Annulus (Orbiting Planet) hit
        # Annulus is centered at (50, 50)
        W_center = jnp.array([50.0, 50.0])[None, None, None, :] - P[:, :, None, :] # [B, 60, 1, 2]
        t_center = jnp.sum(W_center * V_exp, axis=-1) # [B, 60, 1, 32] -> broadcast to [B, 60, 60, 32]
        W_center_sq = jnp.sum(W_center**2, axis=-1)[..., None]
        d_center_sq = W_center_sq - t_center**2
        
        pr = params.planet_orbital_radii[None, None, :, None]
        outer_R = pr + R
        
        # Does the ray intersect the outer radius of the orbit?
        orbit_hit = (t_center > 0) & (d_center_sq < outer_R**2)
        orbit_hit_dist = t_center - jnp.sqrt(jnp.maximum(0.0, outer_R**2 - d_center_sq))
        
        is_orbiting = params.is_orbiting_planet[None, None, :, None]
        
        hit = jnp.where(is_orbiting, orbit_hit, static_hit)
        # Note: We replace the single hit distance line below
"""
    
    # We also need to replace the hit_dist calculation
    old_hit_dist = "hit_dist = jnp.where(hit, t - jnp.sqrt(jnp.maximum(0.0, R**2 - d_sq)), t_bounds[:, :, None, :])"
    new_hit_dist = "hit_dist = jnp.where(hit, jnp.where(is_orbiting, orbit_hit_dist, static_hit_dist), t_bounds[:, :, None, :])"
    
    content = content.replace(old_raycast, new_raycast)
    content = content.replace(old_hit_dist, new_hit_dist)
    
    with open(filepath, 'w') as f:
        f.write(content)
        
patch_file('train.py')
patch_file('train_tpu.py')
patch_file('train_blackwell.py')
