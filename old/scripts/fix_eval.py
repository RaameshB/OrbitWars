import json

with open("eval_local.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # 1. Bump v3 to v4
        source = source.replace("checkpoints_v3", "checkpoints_v4")
        
        # 2. Fix angles unpacking and inject calculate_intercept_angle
        if "ships, angles = logits_to_action(logits, state.planet_ships)" in source:
            source = source.replace("from networks import Actor, logits_to_action", "from networks import Actor, logits_to_action\nfrom train import calculate_intercept_angle")
            
            old_code = """            ships, angles = logits_to_action(logits, state.planet_ships)
            
            # Restore angle
            angles = angles + (pid * 2 * jnp.pi / 4)"""
            
            new_code = """            ships, ds_angles = logits_to_action(logits, state.planet_ships)
            intercept_angles = calculate_intercept_angle(state, params_inner, ships[..., :60])
            angles = jnp.concatenate([intercept_angles, ds_angles], axis=-1)
            
            # Restore angle
            angles = angles + (pid * 2 * jnp.pi / 4)"""
            
            source = source.replace(old_code, new_code)
            
        cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

with open("eval_local.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

