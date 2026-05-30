import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
from orbit_wars_jax import setup, step, EnvAction, AGENT_COUNT, MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP

# Simple random agent — pure function, no RNG returned
def random_actions(rng, state, params):
    k1, k2, k3 = jrandom.split(rng, 3)
    
    # 5% chance to launch a fleet per planet per slot
    launch_mask = jrandom.uniform(k1, (MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP)) < 0.05
    
    # Send a random fraction of the ships
    fraction = jrandom.uniform(k2, (MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP))
    ships_to_send = (state.planet_ships[:, None] * fraction * launch_mask).astype(jnp.int32)
    
    # Random angle
    angles = jrandom.uniform(k3, (MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP), minval=0.0, maxval=2*jnp.pi)
    
    return EnvAction(ships=ships_to_send, angle=angles)

# Benchmark configuration
BATCH_SIZE = 100
STEPS = 500

print(f"JAX Backend: {jax.default_backend()}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Steps per Episode: {STEPS}")

# JIT compile setup and step over a batch
print("Compiling setup...")
vmap_setup = jax.jit(jax.vmap(setup))

print("Compiling step...")
vmap_step = jax.jit(jax.vmap(step, in_axes=(0, 0, 0)))

print("Compiling random actions...")
vmap_actions = jax.jit(jax.vmap(random_actions, in_axes=(0, 0, 0)))

# Initialize
key = jrandom.PRNGKey(42)
keys = jrandom.split(key, BATCH_SIZE)

# Warmup setup
start_time = time.time()
state, params = vmap_setup(keys)
jax.block_until_ready(state)
print(f"Setup compiled and warmed up in {time.time() - start_time:.2f}s")

# Warmup step
start_time = time.time()
# Use fold_in for per-step unique keys (avoids vmapping split)
action_keys = jax.vmap(jrandom.fold_in, in_axes=(0, None))(keys, 0)
actions = vmap_actions(action_keys, state, params)
state, scores, done = vmap_step(state, params, actions)
jax.block_until_ready(state)
print(f"Step compiled and warmed up in {time.time() - start_time:.2f}s")

# Run full benchmark
print("\nRunning Benchmark...")
start_time = time.time()

for i in range(STEPS):
    action_keys = jax.vmap(jrandom.fold_in, in_axes=(0, None))(keys, i)
    actions = vmap_actions(action_keys, state, params)
    state, scores, done = vmap_step(state, params, actions)

jax.block_until_ready(state)
end_time = time.time()

total_time = end_time - start_time
total_steps = BATCH_SIZE * STEPS
steps_per_sec = total_steps / total_time

print(f"\n--- Results ---")
print(f"Total Time: {total_time:.2f}s")
print(f"Total Environments Stepped: {total_steps}")
print(f"Steps / Second: {steps_per_sec:,.0f}")