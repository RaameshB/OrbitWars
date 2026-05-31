import os, glob, sys, argparse, functools, json
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

CLOUDFLARE_WORKER_URL = 'https://orbit-wars-gateway.raameshb.workers.dev/'

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


def compute_final_stats(states_list, num_players, player_labels):
    """Return list of dicts sorted by final ship count (rank 1 = most ships)."""
    final = states_list[-1]
    results = []
    for pid in range(num_players):
        p_ships = int(jnp.sum(jnp.where(final.planet_owners == pid, final.planet_ships, 0)))
        f_ships = int(jnp.sum(jnp.where(final.fleet_owners == pid, final.fleet_ship_count, 0)))
        results.append({
            'player': pid,
            'name': player_labels[pid],
            'color': PLAYER_COLORS[pid],
            'planet_ships': p_ships,
            'fleet_ships': f_ships,
            'total': p_ships + f_ships,
        })
    results.sort(key=lambda x: -x['total'])
    for rank, r in enumerate(results):
        r['rank'] = rank + 1
    return results


os.makedirs("public", exist_ok=True)

has_exploit = exploit_params is not None

PLAYER_LABELS = {
    'ffa-archive':  [f'Archive #{i+1}' for i in range(args.players)],
    'ffa-exploit':  ['Champion'] + [f'Exploiter {i+1}' for i in range(args.players - 1)],
    'duel-archive': ['Archive #1', 'Archive #2'],
    'duel-exploit': ['Champion', 'Exploiter'],
}

all_stats = {}

# --- FFA archive replay ---
print("Simulating FFA archive replay (500 steps)...")
key_ffa_arch = dummy_key
history, params = rollout(top4_params, key_ffa_arch, args.players)
states_list = [tree_map(lambda x: x[t], history) for t in range(500)]
env = jax_states_to_kaggle_env(states_list, params)
all_stats['ffa-archive'] = compute_final_stats(states_list, args.players, PLAYER_LABELS['ffa-archive'])
print("Rendering FFA archive HTML...")
with open("public/orbit_wars_replay.html", "w") as f:
    f.write(env.render(mode="html", width=800, height=800))
print("Written: public/orbit_wars_replay.html")

# --- Duel archive replay ---
print("Simulating Duel archive replay (500 steps)...")
key_duel_arch = jax.random.PRNGKey(int(_time.time()) + 2)
history_da, params_da = rollout(top4_params, key_duel_arch, 2)
states_list_da = [tree_map(lambda x: x[t], history_da) for t in range(500)]
env_da = jax_states_to_kaggle_env(states_list_da, params_da)
all_stats['duel-archive'] = compute_final_stats(states_list_da, 2, PLAYER_LABELS['duel-archive'])
print("Rendering Duel archive HTML...")
with open("public/orbit_wars_replay_duel.html", "w") as f:
    f.write(env_da.render(mode="html", width=800, height=800))
print("Written: public/orbit_wars_replay_duel.html")

# --- FFA exploit replay ---
if has_exploit:
    key_ffa_ex = jax.random.PRNGKey(int(_time.time()) + 1)
    print("Simulating FFA exploit replay (500 steps)...")
    history_ex, params_ex = rollout(exploit_params, key_ffa_ex, args.players)
    states_list_ex = [tree_map(lambda x: x[t], history_ex) for t in range(500)]
    env_ex = jax_states_to_kaggle_env(states_list_ex, params_ex)
    all_stats['ffa-exploit'] = compute_final_stats(states_list_ex, args.players, PLAYER_LABELS['ffa-exploit'])
    print("Rendering FFA exploit HTML...")
    with open("public/orbit_wars_replay_exploit.html", "w") as f:
        f.write(env_ex.render(mode="html", width=800, height=800))
    print("Written: public/orbit_wars_replay_exploit.html")

    # --- Duel exploit replay ---
    key_duel_ex = jax.random.PRNGKey(int(_time.time()) + 3)
    print("Simulating Duel exploit replay (500 steps)...")
    history_de, params_de = rollout(exploit_params, key_duel_ex, 2)
    states_list_de = [tree_map(lambda x: x[t], history_de) for t in range(500)]
    env_de = jax_states_to_kaggle_env(states_list_de, params_de)
    all_stats['duel-exploit'] = compute_final_stats(states_list_de, 2, PLAYER_LABELS['duel-exploit'])
    print("Rendering Duel exploit HTML...")
    with open("public/orbit_wars_replay_duel_exploit.html", "w") as f:
        f.write(env_de.render(mode="html", width=800, height=800))
    print("Written: public/orbit_wars_replay_duel_exploit.html")


# --- Legend helpers ---
def swatch(color, i):
    border = 'border:1.5px solid #777;' if i == 3 else ''
    return f'<span class="swatch" style="background:{color};{border}"></span>'


def legend_html(labels, num):
    items = ''.join(
        f'<div class="legend-item">{swatch(PLAYER_COLORS[i], i)}{label}</div>'
        for i, label in enumerate(labels[:num])
    )
    return f'<div class="legend">{items}</div>'


ALL_LEGEND_LABELS = {
    'ffa-archive':  [f'P{i+1} · Archive #{i+1}' for i in range(args.players)],
    'ffa-exploit':  ['P1 · Champion'] + [f'P{i+2} · Exploiter' for i in range(args.players - 1)],
    'duel-archive': ['P1 · Archive #1', 'P2 · Archive #2'],
    'duel-exploit': ['P1 · Champion', 'P2 · Exploiter'],
}

SRCS = {
    'ffa-archive':  'orbit_wars_replay.html',
    'ffa-exploit':  'orbit_wars_replay_exploit.html',
    'duel-archive': 'orbit_wars_replay_duel.html',
    'duel-exploit': 'orbit_wars_replay_duel_exploit.html',
}

exploit_disabled_attr = '' if has_exploit else 'disabled title="No exploiter checkpoint found"'

legends_js = {
    k: legend_html(v, 4 if k.startswith('ffa') else 2)
    for k, v in ALL_LEGEND_LABELS.items()
}


def js_obj(d):
    pairs = ', '.join(f'"{k}": `{v}`' for k, v in d.items())
    return '{' + pairs + '}'


srcs_js = js_obj(SRCS)
legends_js_str = js_obj(legends_js)
stats_js = json.dumps(all_stats)

index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<title>OrbitWars AI Dashboard</title>
<style>
  :root {{
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg-dark: #0f172a;
    --glass-bg: rgba(30, 41, 59, 0.7);
    --glass-border: rgba(255, 255, 255, 0.1);
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-dark);
    color: white;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    background-image: radial-gradient(circle at 50% 0%, #1e1b4b 0%, var(--bg-dark) 50%);
    padding: 0 env(safe-area-inset-right) 0 env(safe-area-inset-left);
  }}

  header {{
    margin-top: clamp(1rem, 4vw, 2.5rem);
    margin-bottom: clamp(0.75rem, 3vw, 1.5rem);
    text-align: center;
    padding: 0 1rem;
  }}

  h1 {{
    font-size: clamp(1.4rem, 5vw, 2.5rem);
    font-weight: 800;
    margin: 0;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.05em;
  }}

  p.subtitle {{
    color: #94a3b8;
    margin-top: 0.4rem;
    font-size: clamp(0.75rem, 2.5vw, 1rem);
  }}

  @media (max-width: 480px) {{
    p.subtitle {{ display: none; }}
  }}

  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    align-items: center;
    justify-content: center;
    z-index: 10;
    padding: 0 0.75rem;
    width: 100%;
  }}

  .toggle-group {{
    display: flex;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--glass-border);
    border-radius: 9999px;
    padding: 3px;
    backdrop-filter: blur(10px);
  }}

  .toggle-btn {{
    background: transparent;
    border: none;
    color: #94a3b8;
    padding: clamp(0.3rem, 1.5vw, 0.5rem) clamp(0.6rem, 2.5vw, 1.1rem);
    border-radius: 9999px;
    font-size: clamp(0.75rem, 2.5vw, 0.9rem);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    white-space: nowrap;
    -webkit-tap-highlight-color: transparent;
  }}

  .toggle-btn:hover:not(:disabled) {{ color: #e2e8f0; }}

  .toggle-btn.active {{
    background: var(--glass-bg);
    color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}

  .toggle-btn:disabled {{ opacity: 0.35; cursor: not-allowed; }}

  button.primary-btn {{
    background: var(--primary);
    border: none;
    color: white;
    padding: clamp(0.4rem, 1.5vw, 0.6rem) clamp(0.9rem, 3vw, 1.4rem);
    border-radius: 9999px;
    font-size: clamp(0.75rem, 2.5vw, 0.9rem);
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);
    -webkit-tap-highlight-color: transparent;
  }}

  button.primary-btn:hover {{
    background: var(--primary-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 15px -3px rgba(99,102,241,0.4);
  }}

  button.primary-btn:active {{ transform: translateY(0); }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.6rem;
    min-height: 20px;
    padding: 0 0.75rem;
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: clamp(0.7rem, 2vw, 0.8rem);
    color: #94a3b8;
    white-space: nowrap;
  }}

  .swatch {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* Glass container: square, capped at 700px, responsive on mobile */
  .glass-container {{
    width: min(700px, calc(100vw - 1.5rem));
    max-width: 900px;
    aspect-ratio: 1;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: clamp(12px, 3vw, 24px);
    overflow: hidden;
    backdrop-filter: blur(12px);
    box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5);
    position: relative;
    margin-bottom: 1rem;
  }}

  /* Iframes are fixed 800×800 (Kaggle renderer); JS scales them to fit */
  iframe {{
    position: absolute;
    top: 0;
    left: 0;
    width: 800px;
    height: 800px;
    border: none;
    transform-origin: top left;
  }}

  .loading-overlay {{
    position: absolute;
    inset: 0;
    background: rgba(15, 23, 42, 0.88);
    backdrop-filter: blur(8px);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 20;
    border-radius: inherit;
    padding: 1rem;
    text-align: center;
  }}

  .loading-overlay.active {{ opacity: 1; pointer-events: all; }}

  .spinner {{
    width: 44px;
    height: 44px;
    border: 4px solid rgba(99, 102, 241, 0.25);
    border-radius: 50%;
    border-top-color: var(--primary);
    animation: spin 1s ease-in-out infinite;
    margin-bottom: 1.25rem;
    flex-shrink: 0;
  }}

  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

  .loading-text {{
    font-size: clamp(0.85rem, 3vw, 1.1rem);
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1rem;
  }}

  .progress-bar-container {{
    width: min(55%, 240px);
    height: 6px;
    background: rgba(255,255,255,0.1);
    border-radius: 9999px;
    overflow: hidden;
  }}

  .progress-bar {{
    height: 100%;
    width: 0%;
    background: linear-gradient(to right, #6366f1, #a855f7);
    border-radius: 9999px;
    transition: width 1s linear;
  }}

  .progress-hint {{
    color: #64748b;
    font-size: 0.75rem;
    margin-top: 0.75rem;
  }}

  /* Results panel */
  .results-panel {{
    width: min(90%, calc(100vw - 1.5rem));
    max-width: 900px;
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: clamp(10px, 2.5vw, 16px);
    backdrop-filter: blur(12px);
    padding: 0.85rem 1rem;
    margin-bottom: 2rem;
  }}

  .results-title {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.65rem;
  }}

  .results-row {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}

  .results-row:last-child {{ border-bottom: none; }}

  .rank-badge {{
    width: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #64748b;
    flex-shrink: 0;
    text-align: center;
  }}

  .rank-badge.gold   {{ color: #fbbf24; }}
  .rank-badge.silver {{ color: #94a3b8; }}
  .rank-badge.bronze {{ color: #b45309; }}

  .result-name {{
    flex: 1;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: clamp(0.75rem, 2.5vw, 0.85rem);
    font-weight: 600;
    color: #e2e8f0;
    min-width: 0;
  }}

  .result-name span:last-child {{
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .ship-bar-wrap {{
    flex: 2;
    height: 5px;
    background: rgba(255,255,255,0.06);
    border-radius: 9999px;
    overflow: hidden;
    min-width: 40px;
  }}

  @media (max-width: 380px) {{
    .ship-bar-wrap {{ display: none; }}
  }}

  .ship-bar {{
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
  }}

  .ship-count {{
    font-size: clamp(0.7rem, 2vw, 0.8rem);
    color: #94a3b8;
    white-space: nowrap;
    flex-shrink: 0;
  }}
</style>
</head>
<body>

<header>
  <h1>OrbitWars AI Dashboard</h1>
  <p class="subtitle">Live continuous monitoring of the TPU Reinforcement Learning Cluster</p>
</header>

<div class="controls">
  <div class="toggle-group">
    <button class="toggle-btn active" id="btn-ffa"  onclick="setMode('ffa')">FFA (4-way)</button>
    <button class="toggle-btn"        id="btn-duel" onclick="setMode('duel')">Duel (2-way)</button>
  </div>
  <div class="toggle-group">
    <button class="toggle-btn active" id="btn-archive" onclick="setType('archive')">Archive</button>
    <button class="toggle-btn" id="btn-exploit" onclick="setType('exploit')" {exploit_disabled_attr}>Champion vs Exploiters</button>
  </div>
  <button class="primary-btn" onclick="triggerSimulation()">&#128640; New Game</button>
</div>

<div id="legend-wrapper">{legend_html(ALL_LEGEND_LABELS['ffa-archive'], 4)}</div>

<div class="glass-container" id="glass-container">
  <div class="loading-overlay" id="loadingOverlay">
    <div class="spinner"></div>
    <div class="loading-text" id="loadingText">Compiling JAX XLA Graphs...</div>
    <div class="progress-bar-container">
      <div class="progress-bar" id="progressBar"></div>
    </div>
    <p class="progress-hint">This process takes approximately 3 minutes.</p>
  </div>
  <iframe id="frame-ffa-archive"  src="orbit_wars_replay.html" style="display:block"></iframe>
  <iframe id="frame-ffa-exploit"  src="" style="display:none"></iframe>
  <iframe id="frame-duel-archive" src="" style="display:none"></iframe>
  <iframe id="frame-duel-exploit" src="" style="display:none"></iframe>
</div>

<div class="results-panel" id="results-panel">
  <div class="results-title">Final Results · 500 Steps</div>
  <div id="results-rows"></div>
</div>

<script>
  const EXPLOIT_AVAILABLE = {'true' if has_exploit else 'false'};
  const SRCS    = {srcs_js};
  const LEGENDS = {legends_js_str};
  const STATS   = {stats_js};

  let currentMode = 'ffa';
  let currentType = 'archive';
  const loaded = {{'ffa-archive': true, 'ffa-exploit': false, 'duel-archive': false, 'duel-exploit': false}};

  const RANK_CLASSES = ['gold', 'silver', 'bronze', ''];
  const RANK_SYMBOLS = ['#1', '#2', '#3', '#4'];

  function currentKey() {{ return currentMode + '-' + currentType; }}

  // Scale all iframes so the 800×800 Kaggle render fills the container
  const IFRAME_SIZE = 800;
  function scaleIframes() {{
    const c = document.getElementById('glass-container');
    const scale = Math.min(c.clientWidth / IFRAME_SIZE, c.clientHeight / IFRAME_SIZE);
    const ox = (c.clientWidth  - IFRAME_SIZE * scale) / 2;
    const oy = (c.clientHeight - IFRAME_SIZE * scale) / 2;
    document.querySelectorAll('iframe').forEach(f => {{
      f.style.transform = `scale(${{scale}})`;
      f.style.left = ox + 'px';
      f.style.top  = oy + 'px';
    }});
  }}

  window.addEventListener('resize', scaleIframes);
  scaleIframes(); // script is at end of body — DOM and CSS are already applied

  function renderStats(key) {{
    const rows = STATS[key];
    if (!rows || !rows.length) {{
      document.getElementById('results-panel').style.display = 'none';
      return;
    }}
    document.getElementById('results-panel').style.display = '';
    const maxTotal = Math.max(...rows.map(r => r.total), 1);
    document.getElementById('results-rows').innerHTML = rows.map(r => {{
      const pct = Math.round(r.total / maxTotal * 100);
      const rankClass = RANK_CLASSES[r.rank - 1] || '';
      const symbol = RANK_SYMBOLS[r.rank - 1] || '#' + r.rank;
      const border = r.player === 3 ? 'border:1.5px solid #777;' : '';
      return `<div class="results-row">
        <span class="rank-badge ${{rankClass}}">${{symbol}}</span>
        <span class="result-name">
          <span style="background:${{r.color}};${{border}}width:9px;height:9px;border-radius:50%;flex-shrink:0;display:inline-block"></span>
          <span>P${{r.player + 1}} · ${{r.name}}</span>
        </span>
        <div class="ship-bar-wrap">
          <div class="ship-bar" style="width:${{pct}}%;background:${{r.color}}"></div>
        </div>
        <span class="ship-count">${{r.total.toLocaleString()}}</span>
      </div>`;
    }}).join('');
  }}

  function updateView() {{
    const key = currentKey();
    ['ffa-archive','ffa-exploit','duel-archive','duel-exploit'].forEach(k => {{
      document.getElementById('frame-' + k).style.display = k === key ? 'block' : 'none';
    }});
    if (!loaded[key]) {{
      document.getElementById('frame-' + key).src = SRCS[key];
      loaded[key] = true;
    }}
    document.getElementById('legend-wrapper').innerHTML = LEGENDS[key];
    renderStats(key);
    scaleIframes();

    document.getElementById('btn-ffa').className    = 'toggle-btn' + (currentMode === 'ffa'   ? ' active' : '');
    document.getElementById('btn-duel').className   = 'toggle-btn' + (currentMode === 'duel'  ? ' active' : '');
    document.getElementById('btn-archive').className = 'toggle-btn' + (currentType === 'archive' ? ' active' : '');
    document.getElementById('btn-exploit').className = 'toggle-btn'
      + (!EXPLOIT_AVAILABLE ? '' : currentType === 'exploit' ? ' active' : '');
  }}

  function setMode(mode) {{
    currentMode = mode;
    if (!EXPLOIT_AVAILABLE && currentType === 'exploit') currentType = 'archive';
    updateView();
  }}

  function setType(type) {{
    if (type === 'exploit' && !EXPLOIT_AVAILABLE) return;
    currentType = type;
    updateView();
  }}

  const CLOUDFLARE_WORKER_URL = '{CLOUDFLARE_WORKER_URL}';

  async function triggerSimulation() {{
    const overlay    = document.getElementById('loadingOverlay');
    const progressBar = document.getElementById('progressBar');
    const loadingText = document.getElementById('loadingText');

    overlay.classList.add('active');
    progressBar.style.width = '0%';
    loadingText.textContent = 'Triggering simulation...';

    try {{
      const response = await fetch(CLOUDFLARE_WORKER_URL, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ players: 4 }})
      }});

      if (!response.ok) throw new Error('API error ' + response.status);

      let progress = 0;
      loadingText.textContent = 'Simulating 4-Way Free-For-All...';

      const interval = setInterval(() => {{
        progress += 100 / 180;
        if (progress >= 30  && progress < 60)  loadingText.textContent = 'Downloading latest R2 checkpoints...';
        if (progress >= 60  && progress < 90)  loadingText.textContent = 'Executing physics engine rollout...';
        if (progress >= 90  && progress < 120) loadingText.textContent = 'Rendering interactive HTML...';
        if (progress >= 120)                   loadingText.textContent = 'Deploying to GitHub Pages...';

        if (progress >= 100) {{
          clearInterval(interval);
          progressBar.style.width = '100%';
          setTimeout(() => {{
            const ts = '?t=' + Date.now();
            ['ffa-archive','ffa-exploit','duel-archive','duel-exploit'].forEach(k => loaded[k] = false);
            loaded['ffa-archive'] = true;
            document.getElementById('frame-ffa-archive').src = SRCS['ffa-archive'] + ts;
            if (currentKey() !== 'ffa-archive') {{
              loaded[currentKey()] = true;
              document.getElementById('frame-' + currentKey()).src = SRCS[currentKey()] + ts;
            }}
            overlay.classList.remove('active');
          }}, 5000);
        }} else {{
          progressBar.style.width = progress + '%';
        }}
      }}, 1000);

    }} catch (err) {{
      console.error(err);
      loadingText.textContent = 'Error: Failed to connect to API Gateway.';
      setTimeout(() => overlay.classList.remove('active'), 3000);
    }}
  }}

  renderStats('ffa-archive');
</script>
</body>
</html>"""

with open("public/index.html", "w") as f:
    f.write(index_html)
print("Written: public/index.html")
print("Done. Open public/index.html to view.")
