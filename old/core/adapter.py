import numpy as np
import jax.numpy as jnp
from orbit_wars_jax import EnvState, EnvParams, EnvAction, AGENT_COUNT, MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP
from kaggle_environments import make as k_make
import original_orbit_wars as ref
import jax.random as jrandom
import time

def kaggle_action_to_jax_action(k_actions, p_owners):
    """
    Converts a list of Kaggle moves to a JAX EnvAction PyTree.
    This is a simplified scatter operation for the demo.
    """
    ships = np.zeros((MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP), dtype=np.int32)
    angles = np.zeros((MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP), dtype=np.float32)
    
    # A simple counter to handle multiple launches from the same planet
    launch_slots = {}

    for move in k_actions:
        planet_id, angle, num_ships = move
        
        # Ensure the agent actually owns the planet
        if p_owners[planet_id] != 0: # Assuming player 0 for this demo
            continue

        slot = launch_slots.get(planet_id, 0)
        if slot < MAX_FLEETS_PER_PLANET_PER_STEP:
            ships[planet_id, slot] = int(num_ships)
            angles[planet_id, slot] = angle
            launch_slots[planet_id] = slot + 1
            
    return EnvAction(ships=jnp.array(ships), angle=jnp.array(angles))

def run_side_by_side_demo(jax_setup, jax_step, num_steps=100):
    """
    Runs the JAX and original Python environments side-by-side
    and prints a visual comparison at each step.
    """
    
    # --- Initialization ---
    seed = 42
    key = jrandom.PRNGKey(seed)
    
    # JAX Env
    print("Setting up JAX environment...")
    jax_state, jax_params = jax_setup(key)
    
    # Kaggle Env
    print("Setting up Kaggle environment...")
    kaggle_env = k_make("orbit_wars", debug=True, configuration={"seed": seed})
    kaggle_obs = kaggle_env.reset(num_agents=AGENT_COUNT)[0].observation
    
    print("\n--- INITIAL STATE (Step 0) ---")
    print_comparison(jax_state, kaggle_obs)
    
    # --- Step Loop ---
    for i in range(num_steps):
        print(f"\n--- STEP {i+1} ---")
        
        # Generate actions from the Kaggle observation (as our "source of truth" agent)
        # We use a simple random agent for this demo
        p0_kaggle_moves = ref.random_agent(kaggle_obs)
        
        # Translate Kaggle actions to JAX action format
        jax_actions = kaggle_action_to_jax_action(p0_kaggle_moves, np.array(jax_state.planet_owners))
        
        # Step both environments with the same actions
        jax_state, _, _ = jax_step(jax_state, jax_params, jax_actions)
        
        # For the Kaggle env, we assume a 4-player game and give only player 0 the actions
        full_k_actions = [[]] * AGENT_COUNT
        full_k_actions[0] = p0_kaggle_moves
        kaggle_obs = kaggle_env.step(full_k_actions)[0].observation
        
        print_comparison(jax_state, kaggle_obs)
        
        if kaggle_env.done:
            print("\nKaggle Environment terminated.")
            break
        time.sleep(0.5)

def print_comparison(jax_state, kaggle_obs):
    """Prints a side-by-side view of JAX and Kaggle states."""
    
    jax_planets = np.array(jax_state.planet_coords)
    jax_owners = np.array(jax_state.planet_owners)
    jax_ships = np.array(jax_state.planet_ships)
    # Active = on the board (not at placeholder -99 coords)
    jax_active = (jax_planets[:, 0] > -50) & (jax_planets[:, 1] > -50)
    
    k_planets = {p[0]: p for p in kaggle_obs.planets}
    
    print("--- JAX State ---")
    for i in range(MAX_BODIES):
        if jax_active[i]:
            print(f"  Planet {i}: Owner={jax_owners[i]}, Ships={jax_ships[i]}, Pos=({jax_planets[i, 0]:.1f}, {jax_planets[i, 1]:.1f})")
            
    print("\n--- Kaggle State ---")
    for pid, p in sorted(k_planets.items()):
        print(f"  Planet {pid}: Owner={p[1]}, Ships={p[5]}, Pos=({p[2]:.1f}, {p[3]:.1f})")

if __name__ == "__main__":
    from orbit_wars_jax import setup, step, AGENT_COUNT, EnvAction, MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP
    
    run_side_by_side_demo(setup, step)