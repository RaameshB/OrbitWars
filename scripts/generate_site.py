import os, glob, sys, argparse, functools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from jax.sharding import SingleDeviceSharding
from jax.tree_util import tree_map

from core.networks import Actor, logits_to_action
from core.orbit_wars_jax import setup, step, EnvAction, MAX_FLEETS, MAX_COMET_PATH_LEN
from core.rollout_utils import calculate_intercept_angle, build_obs_arrays
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from scripts.jax_visualizer import jax_states_to_kaggle_env

parser = argparse.ArgumentParser()
parser.add_argument('--players', type=int, default=4, help='Number of players in the simulation')
args = parser.parse_args()

print(f"Starting {args.players}-Way HTML Site Generation...")

# Find latest checkpoint
checkpoints = glob.glob('/tmp/checkpoints_v6/qdax_rep_*')
if not checkpoints:
    print("Warning: No checkpoints found. Rendering untrained network.")
    latest_ckpt = None
else:
    latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[-1]))
    print(f'Loading Checkpoint: {latest_ckpt}')

# Setup architecture — use a time-based key so each deploy generates a fresh game
import time as _time
dummy_key = jax.random.PRNGKey(int(_time.time()))
actor_nnx = Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(0))
actor_graph, init_params = nnx.split(actor_nnx)
init_params = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_params)

if latest_ckpt:
    centroids = compute_cvt_centroids(num_descriptors=4, num_init_cvt_samples=100000, num_centroids=10000, minval=0.0, maxval=1.0, key=dummy_key)
    dummy_rep = MapElitesRepertoire.init(init_params, jnp.array([-jnp.inf]), jnp.zeros((1, 4)), centroids)

    checkpointer = ocp.PyTreeCheckpointer()
    cpu_device = jax.local_devices(backend='cpu')[0]
    sharding = SingleDeviceSharding(cpu_device)
    sharding_tree = jax.tree_util.tree_map(lambda x: sharding, dummy_rep)
    restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_rep, sharding_tree)

    try:
        repertoire = checkpointer.restore(latest_ckpt, item=dummy_rep, restore_args=restore_args)
        # Select Top 4 Agents
        top_4_indices = jnp.argsort(repertoire.fitnesses.squeeze())[-4:]
        top4_params = jax.tree_util.tree_map(lambda x: x[top_4_indices], repertoire.genotypes)
        print("Successfully loaded Top 4 agents!")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Falling back to untrained network.")
        p_keys = jax.random.split(dummy_key, 4)
        top4_params = jax.vmap(lambda k: nnx.split(Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(k)))[1])(p_keys)
else:
    p_keys = jax.random.split(dummy_key, 4)
    top4_params = jax.vmap(lambda k: nnx.split(Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(k)))[1])(p_keys)

@functools.partial(jax.jit, static_argnames=('num_players',))
def rollout(top4_params, random_key, num_players=4):
    p0_params = jax.tree_util.tree_map(lambda x: x[0], top4_params)
    p1_params = jax.tree_util.tree_map(lambda x: x[1], top4_params)
    p2_params = jax.tree_util.tree_map(lambda x: x[2], top4_params)
    p3_params = jax.tree_util.tree_map(lambda x: x[3], top4_params)

    actor_p0 = nnx.merge(actor_graph, p0_params)
    actor_p1 = nnx.merge(actor_graph, p1_params)
    actor_p2 = nnx.merge(actor_graph, p2_params)
    actor_p3 = nnx.merge(actor_graph, p3_params)

    def scan_step(carry, _):
        state, params_inner, key = carry
        key, subkey = jax.random.split(key)

        # Active-body mask: False for unspawned/expired comets and failed-to-generate planets
        ages_now = state.step - params_inner.comet_spawn_steps      # [60]
        comet_active_now = params_inner.is_comet & (ages_now >= 0) & (ages_now < params_inner.body_lifespans)
        target_active = params_inner.is_static_planet | params_inner.is_orbiting_planet | comet_active_now  # [60]
        target_mask_64 = jnp.concatenate([target_active, jnp.ones(4, dtype=bool)], axis=-1)  # [64]

        def run_player(pid, actor):
            planets, fleets, p_mask, f_mask = build_obs_arrays(state, params_inner, pid, num_players)
            logits = actor(planets, fleets, planet_mask=p_mask, fleet_mask=f_mask)
            ships, ds_angles = logits_to_action(logits, state.planet_ships)
            ships = jnp.where(target_mask_64[None, :], ships, 0.0)
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            params_b = jax.tree_util.tree_map(lambda x: x[None, ...], params_inner)
            ships_b = ships[None, ..., :60]
            intercept_angles = calculate_intercept_angle(state_b, params_b, ships_b)[0]
            # intercept_angles are world-frame; ds_angles are player-rotated and need conversion
            ds_angles_world = ds_angles + (pid * 2 * jnp.pi / num_players)
            angles = jnp.concatenate([intercept_angles, ds_angles_world], axis=-1)
            ships = jnp.where(ships < 1.0, 0.0, ships)
            is_player = (state.planet_owners == pid)[..., None]
            return jnp.where(is_player, ships, 0), jnp.where(is_player, angles, 0.0)

        ships_p0, angles_p0 = run_player(0, actor_p0)
        ships_p1, angles_p1 = run_player(1, actor_p1)

        if num_players == 4:
            ships_p2, angles_p2 = run_player(2, actor_p2)
            ships_p3, angles_p3 = run_player(3, actor_p3)
            final_ships = ships_p0 + ships_p1 + ships_p2 + ships_p3
            final_angles = angles_p0 + angles_p1 + angles_p2 + angles_p3
        else:
            final_ships = ships_p0 + ships_p1
            final_angles = angles_p0 + angles_p1
        env_action = EnvAction(ships=final_ships.astype(jnp.int32), angle=final_angles)

        next_state, _, _ = step(state, params_inner, env_action, num_players)
        return (next_state, params_inner, key), state

    init_state, params_env = setup(random_key, num_players)
    _, history = jax.lax.scan(scan_step, (init_state, params_env, random_key), None, length=500)
    return history, params_env

print("Simulating 500 steps...")
history, params = rollout(top4_params, dummy_key, args.players)

print("Converting to Kaggle Environment...")
states_list = [tree_map(lambda x: x[t], history) for t in range(500)]
env = jax_states_to_kaggle_env(states_list, params)

print("Rendering HTML...")
html_str = env.render(mode="html", width=800, height=800)

os.makedirs("public", exist_ok=True)
with open("public/orbit_wars_replay.html", "w") as f:
    f.write(html_str)

print("HTML artifact successfully written to public/orbit_wars_replay.html")
