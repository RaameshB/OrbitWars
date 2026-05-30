import sys
import glob
import re

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    old_math = """        # 4. Calculate Distance and Intercept Time
        src = state.planet_coords[:, :, None, None, :]
        dists = jnp.sqrt((tfc[..., 0] - src[..., 0])**2 + (tfc[..., 1] - src[..., 1])**2)
        travel_dist = v_fleet[..., None] * ts[None, None, None, :]
        
        can_reach = travel_dist >= dists"""

    new_math = """        # 4. Calculate Distance and Intercept Time
        src = state.planet_coords[:, :, None, None, :]
        dists = jnp.sqrt((tfc[..., 0] - src[..., 0])**2 + (tfc[..., 1] - src[..., 1])**2)
        
        # Account for fleet spawn offset (R_src + 0.1) and target collision radius (R_tgt)
        R_src = params.planet_radii[:, :, None, None]
        R_tgt = params.planet_radii[:, None, :, None]
        
        travel_dist = R_src + 0.1 + v_fleet[..., None] * ts[None, None, None, :]
        req_dist = dists - R_tgt
        
        can_reach = travel_dist >= req_dist"""

    if old_math in content:
        content = content.replace(old_math, new_math)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Patched {filepath}")
    else:
        print(f"Failed to find target in {filepath}")

for f in glob.glob("train*.py"):
    patch_file(f)
