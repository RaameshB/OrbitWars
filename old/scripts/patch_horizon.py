import sys
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace arange(1, 51)
    content = content.replace("jnp.arange(1, 51, dtype=jnp.float32)", "jnp.arange(1, 151, dtype=jnp.float32)")
    
    # Replace 49 with 149
    content = content.replace("intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 49)", "intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 149)")
    
    # Replace padded_comet_coords
    content = content.replace("padded_comet_coords = jnp.zeros((B, 60, 50, 2))", "padded_comet_coords = jnp.zeros((B, 60, 150, 2))")
    
    # Replace static_coords
    content = content.replace("static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B, 60, 50, 2))", "static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B, 60, 150, 2))")

    with open(filepath, 'w') as f:
        f.write(content)

for f in glob.glob("train*.py"):
    patch_file(f)
    print(f"Patched {f}")
