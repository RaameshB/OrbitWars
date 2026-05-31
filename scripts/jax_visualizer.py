import numpy as np
from kaggle_environments import make
from kaggle_environments.utils import Struct
from core.orbit_wars_jax import TOTAL_COMETS, MAX_COMET_PATH_LEN

def jax_states_to_kaggle_env(states, params):
    """
    Takes a list of JAX EnvState objects (a trajectory) and a single EnvParams,
    and returns a Kaggle environment populated with those states so you can render it.
    """
    # Create a dummy Kaggle env
    env = make("orbit_wars", debug=True)
    # Reset it to get the base structure (AGENT_COUNT=4)
    env.reset(num_agents=4)

    # We will replace env.steps with our custom steps
    new_steps = []

    # Extract static info from params
    planet_radii = np.array(params.planet_radii)
    planet_prod = np.array(params.planet_prod)
    angular_velocity = float(np.array(params.angular_velocity))
    comet_paths = np.array(params.comet_paths)       # [TOTAL_COMETS, MAX_COMET_PATH_LEN, 2]
    comet_spawn_steps = np.array(params.comet_spawn_steps)[-TOTAL_COMETS:]  # [TOTAL_COMETS]
    body_lifespans = np.array(params.body_lifespans)         # [MAX_BODIES]
    comet_lifespans = body_lifespans[-TOTAL_COMETS:]         # last TOTAL_COMETS entries

    for state in states:
        # Convert JAX arrays to NumPy
        p_owners = np.array(state.planet_owners)
        p_coords = np.array(state.planet_coords)
        p_ships = np.array(state.planet_ships)

        f_owners = np.array(state.fleet_owners)
        f_coords = np.array(state.fleet_coords)
        f_angles = np.array(state.fleet_angles)
        f_ships = np.array(state.fleet_ship_count)

        step_idx = int(np.array(state.step))

        # Compute active comets for this step
        comet_ages = step_idx - comet_spawn_steps  # [TOTAL_COMETS]
        k_comets = []
        for ci in range(TOTAL_COMETS):
            age = comet_ages[ci]
            is_spawned = age >= 0
            is_expired = age >= comet_lifespans[ci]
            if not is_spawned or is_expired:
                continue
            safe_age = int(np.clip(age, 0, MAX_COMET_PATH_LEN - 1))
            cx, cy = comet_paths[ci, safe_age, :]
            # Skip comets that are out of board bounds
            if cx < 0 or cx > 100 or cy < 0 or cy > 100:
                continue
            k_comets.append([ci, float(cx), float(cy)])

        # Build Kaggle planets list: [id, owner, y, x, r, ships, prod]
        k_planets = []
        for i in range(len(p_owners)):
            # Only include planets that are actually active (not at placeholder coords)
            if p_coords[i, 0] > -50 and p_coords[i, 1] > -50:
                k_planets.append([
                    i,                       # id
                    int(p_owners[i]),        # owner
                    float(p_coords[i, 0]),   # x
                    float(p_coords[i, 1]),   # y
                    float(planet_radii[i]),  # radius
                    int(p_ships[i]),         # ships
                    int(planet_prod[i])      # production
                ])

        # Build Kaggle fleets list: [id, owner, x, y, angle, from_planet, ships]
        k_fleets = []
        for i in range(len(f_owners)):
            if f_owners[i] != -1:
                k_fleets.append([
                    i,                       # id
                    int(f_owners[i]),        # owner
                    float(f_coords[i, 0]),   # x
                    float(f_coords[i, 1]),   # y
                    float(f_angles[i]),      # angle
                    -1,                      # from_planet_id (dummy, mostly unused for rendering)
                    int(f_ships[i])          # ships
                ])

        # Extract initial planets from the very first state if not already done
        if "initial_planets" not in locals():
            initial_planets = [p[:] for p in k_planets]  # deep copy

        # Build Kaggle observation struct for Player 0 (the renderer only looks at obs[0])
        obs = Struct(**{
            "step": step_idx,
            "player": 0,
            "planets": k_planets,
            "initial_planets": initial_planets,
            "fleets": k_fleets,
            "angular_velocity": angular_velocity,
            "comets": k_comets,
            "comet_planet_ids": [],
            "next_fleet_id": 0,
            "remainingOverageTime": 60.0
        })

        # Build the step frame (list of agents, we just need the first agent to have the obs)
        # We copy the structure from a default step
        agent_frame = []
        is_last = (state is states[-1])
        for p_idx in range(4):
            # Create a shallow copy of obs and update player ID for each agent
            p_obs = obs.copy() if hasattr(obs, 'copy') else dict(obs)
            p_obs["player"] = p_idx
            p_struct = Struct(**p_obs)
            agent_frame.append(Struct(**{
                "observation": p_struct,
                "status": "DONE" if is_last else "ACTIVE",
                "reward": 0,
                "action": [],
                "info": {}
            }))

        new_steps.append(agent_frame)

    env.steps = new_steps
    if new_steps:
        env.state = new_steps[-1]
    return env

# You can use this in your notebook like so:
#
# from jax_visualizer import jax_states_to_kaggle_env
#
# # Run your JAX loop and collect `states`
# states = []
# state, params = setup(key)
# for i in range(100):
#     states.append(state)
#     state, _, _ = step(state, params, some_actions)
#
# # Convert and render
# env = jax_states_to_kaggle_env(states, params)
# env.render(mode="ipython")
