import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Fix the params shape bugs
    content = content.replace("R = params.planet_radii[None, None, :, None]", "R = params.planet_radii[:, None, :, None]")
    content = content.replace("pr = params.planet_orbital_radii[None, None, :, None]", "pr = params.planet_orbital_radii[:, None, :, None]")
    content = content.replace("is_orbiting = params.is_orbiting_planet[None, None, :, None]", "is_orbiting = params.is_orbiting_planet[:, None, :, None]")
    
    with open(filepath, 'w') as f:
        f.write(content)

patch_file('train.py')
patch_file('train_tpu.py')
patch_file('train_blackwell.py')
