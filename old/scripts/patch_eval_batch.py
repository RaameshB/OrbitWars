import json

with open("eval_local.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "intercept_angles = calculate_intercept_angle(" in source:
            old_line = "            intercept_angles = calculate_intercept_angle(state, params_inner, ships[..., :60])"
            new_lines = """            # The calculate_intercept_angle function expects batched tensors [B, ...]
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            params_b = jax.tree_util.tree_map(lambda x: x[None, ...], params_inner)
            ships_b = ships[None, ..., :60]
            intercept_angles = calculate_intercept_angle(state_b, params_b, ships_b)[0]"""
            source = source.replace(old_line, new_lines)
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

with open("eval_local.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

