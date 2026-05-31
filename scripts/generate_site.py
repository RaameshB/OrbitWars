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

HOF_ARCHIVE_SIZE = 34
HOF_EXPLOIT_SIZE = 16
HOF_SIZE = HOF_ARCHIVE_SIZE + HOF_EXPLOIT_SIZE

# Wong colorblind-safe palette — must match jax_visualizer.py
PLAYER_COLORS = ['#0072B2', '#D55E00', '#009E73', '#F0E442']

print(f"Starting {args.players}-Way HTML Site Generation...")

# Find latest checkpoint (exclude _hof suffix)
checkpoints = [c for c in glob.glob('/tmp/checkpoints_v6/qdax_rep_*') if not c.endswith('_hof')]
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

checkpointer = ocp.PyTreeCheckpointer()
cpu_device = jax.local_devices(backend='cpu')[0]
sharding = SingleDeviceSharding(cpu_device)

repertoire = None
top4_params = None
exploit_params = None  # champion vs exploiters, set if HoF available

if latest_ckpt:
    centroids = compute_cvt_centroids(num_descriptors=4, num_init_cvt_samples=100000, num_centroids=10000, minval=0.0, maxval=1.0, key=dummy_key)
    dummy_rep = MapElitesRepertoire.init(init_params, jnp.array([-jnp.inf]), jnp.zeros((1, 4)), centroids)
    sharding_tree = jax.tree_util.tree_map(lambda x: sharding, dummy_rep)
    restore_args = ocp.checkpoint_utils.construct_restore_args(dummy_rep, sharding_tree)

    try:
        repertoire = checkpointer.restore(latest_ckpt, item=dummy_rep, restore_args=restore_args)
        top_4_indices = jnp.argsort(repertoire.fitnesses.squeeze())[-4:]
        top4_params = jax.tree_util.tree_map(lambda x: x[top_4_indices], repertoire.genotypes)
        print("Successfully loaded Top 4 agents!")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Falling back to untrained network.")

    # Try loading HoF for champion-vs-exploiter replay
    hof_ckpt = latest_ckpt + "_hof"
    if repertoire is not None and os.path.exists(hof_ckpt):
        print(f"Loading HoF from {hof_ckpt}...")
        dummy_hof = jax.tree_util.tree_map(lambda x: jnp.zeros((HOF_SIZE,) + x.shape[1:], dtype=x.dtype), init_params)
        sharding_hof = jax.tree_util.tree_map(lambda x: sharding, dummy_hof)
        restore_args_hof = ocp.checkpoint_utils.construct_restore_args(dummy_hof, sharding_hof)
        try:
            hof_loaded = checkpointer.restore(hof_ckpt, item=dummy_hof, restore_args=restore_args_hof)
            top_idx = jnp.argmax(repertoire.fitnesses.squeeze())
            champion = jax.tree_util.tree_map(lambda x: x[top_idx], repertoire.genotypes)
            num_exploiters = args.players - 1
            exploit_indices = [HOF_ARCHIVE_SIZE + (k % HOF_EXPLOIT_SIZE) for k in range(num_exploiters)]
            exploiters = [jax.tree_util.tree_map(lambda x, i=i: x[i], hof_loaded) for i in exploit_indices]
            exploit_params = jax.tree_util.tree_map(
                lambda c, *es: jnp.stack([c, *es], axis=0), champion, *exploiters
            )
            print(f"Exploit replay: champion vs exploiter slots {exploit_indices}")
        except Exception as e:
            print(f"Failed to load HoF ({e}). Skipping exploit replay.")

if top4_params is None:
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

        ages_now = state.step - params_inner.comet_spawn_steps
        comet_active_now = params_inner.is_comet & (ages_now >= 0) & (ages_now < params_inner.body_lifespans)
        target_active = params_inner.is_static_planet | params_inner.is_orbiting_planet | comet_active_now
        target_mask_64 = jnp.concatenate([target_active, jnp.ones(4, dtype=bool)], axis=-1)

        def run_player(pid, actor):
            planets, fleets, p_mask, f_mask = build_obs_arrays(state, params_inner, pid, num_players)
            logits = actor(planets, fleets, planet_mask=p_mask, fleet_mask=f_mask)
            ships, ds_angles = logits_to_action(logits, state.planet_ships)
            ships = jnp.where(target_mask_64[None, :], ships, 0.0)
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            params_b = jax.tree_util.tree_map(lambda x: x[None, ...], params_inner)
            ships_b = ships[None, ..., :60]
            intercept_angles = calculate_intercept_angle(state_b, params_b, ships_b)[0]
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


os.makedirs("public", exist_ok=True)

# --- Archive replay (top 4 agents) ---
print("Simulating archive replay (500 steps)...")
history, params = rollout(top4_params, dummy_key, args.players)
states_list = [tree_map(lambda x: x[t], history) for t in range(500)]
env = jax_states_to_kaggle_env(states_list, params)
print("Rendering archive HTML...")
with open("public/orbit_wars_replay.html", "w") as f:
    f.write(env.render(mode="html", width=800, height=800))
print("Written: public/orbit_wars_replay.html")

# --- Exploit replay (champion vs exploiters) ---
has_exploit = exploit_params is not None
if has_exploit:
    dummy_key2 = jax.random.PRNGKey(int(_time.time()) + 1)
    print("Simulating exploit replay (500 steps)...")
    history_ex, params_ex = rollout(exploit_params, dummy_key2, args.players)
    states_list_ex = [tree_map(lambda x: x[t], history_ex) for t in range(500)]
    env_ex = jax_states_to_kaggle_env(states_list_ex, params_ex)
    print("Rendering exploit HTML...")
    with open("public/orbit_wars_replay_exploit.html", "w") as f:
        f.write(env_ex.render(mode="html", width=800, height=800))
    print("Written: public/orbit_wars_replay_exploit.html")

# --- Index wrapper with toggle + legend ---
def legend_html(labels):
    items = ''.join(
        f'<div class="legend-item">'
        f'<span class="swatch" style="background:{PLAYER_COLORS[i]};'
        f'{"border:1px solid #777;" if i == 3 else ""}"></span>'
        f'{label}</div>'
        for i, label in enumerate(labels)
    )
    return f'<div class="legend">{items}</div>'

archive_labels = [f'P{i+1} · Archive #{i+1}' for i in range(args.players)]
exploit_labels = ['P1 · Champion'] + [f'P{i+1} · Exploiter' for i in range(1, args.players)]

exploit_btn_disabled = '' if has_exploit else 'disabled title="No exploiter checkpoint found"'
exploit_btn_class = 'btn' if has_exploit else 'btn disabled'

index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>OrbitWars Replay</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0e0e0e; color: #ddd; font-family: 'Courier New', monospace; height: 100vh; display: flex; flex-direction: column; }}
  .bar {{ display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #161616; border-bottom: 1px solid #2a2a2a; flex-shrink: 0; }}
  .btn {{ padding: 6px 14px; background: #1e1e1e; color: #aaa; border: 1px solid #333; cursor: pointer; border-radius: 3px; font-family: inherit; font-size: 13px; transition: all 0.15s; }}
  .btn:hover:not(.disabled) {{ border-color: #555; color: #eee; }}
  .btn.active {{ border-color: #0072B2; color: #fff; background: #0a2d4a; }}
  .btn.disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .sep {{ color: #333; font-size: 18px; }}
  .legend {{ display: flex; gap: 14px; align-items: center; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: #bbb; white-space: nowrap; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }}
  .frame-wrap {{ flex: 1; position: relative; }}
  iframe {{ position: absolute; inset: 0; width: 100%; height: 100%; border: none; }}
</style>
</head>
<body>
<div class="bar">
  <button class="btn active" id="btn-archive" onclick="switchTo('archive')">Top {args.players} Archive Agents</button>
  <button class="{exploit_btn_class}" id="btn-exploit" onclick="switchTo('exploit')" {exploit_btn_disabled}>Champion vs Exploiters</button>
  <span class="sep">|</span>
  <div id="leg-archive">{legend_html(archive_labels)}</div>
  <div id="leg-exploit" style="display:none">{legend_html(exploit_labels)}</div>
</div>
<div class="frame-wrap">
  <iframe id="frame-archive" src="orbit_wars_replay.html"></iframe>
  <iframe id="frame-exploit" src="" style="display:none"></iframe>
</div>
<script>
  let exploitLoaded = false;
  function switchTo(mode) {{
    if (mode === 'exploit' && document.getElementById('btn-exploit').classList.contains('disabled')) return;
    const isArchive = mode === 'archive';
    document.getElementById('frame-archive').style.display = isArchive ? 'block' : 'none';
    document.getElementById('frame-exploit').style.display = isArchive ? 'none' : 'block';
    document.getElementById('leg-archive').style.display = isArchive ? 'flex' : 'none';
    document.getElementById('leg-exploit').style.display = isArchive ? 'none' : 'flex';
    document.getElementById('btn-archive').className = 'btn' + (isArchive ? ' active' : '');
    document.getElementById('btn-exploit').className = '{exploit_btn_class}' + (isArchive ? '' : ' active');
    if (!isArchive && !exploitLoaded) {{
      document.getElementById('frame-exploit').src = 'orbit_wars_replay_exploit.html';
      exploitLoaded = true;
    }}
  }}
</script>
</body>
</html>"""

with open("public/index.html", "w") as f:
    f.write(index_html)
print("Written: public/index.html")
print("Done. Open public/index.html to view.")
