import json

with open("eval_local.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "from orbit_wars_jax import" in source:
            # Add calculate_intercept_angle to the imports
            source = source.replace("from orbit_wars_jax import EnvAction, setup, step", "from orbit_wars_jax import EnvAction, setup, step, calculate_intercept_angle")
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

with open("eval_local.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

