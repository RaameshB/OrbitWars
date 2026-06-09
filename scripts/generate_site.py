"""Generate an HTML replay for a custom agent matchup.

Usage (from project root):
    python scripts/generate_site.py --players 4 --agents bc_init iter0099 sniper random

Agent identifiers:
    sniper   — rule-based nearest-sniper (pure JAX)
    random   — random agent (pure JAX)
    <tag>    — HoF agent tag (downloaded from R2 as rl_v25/hof/{mode}/{tag}.pkl)
"""
import argparse, os, sys, pickle, functools, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jax.tree_util import tree_map
import boto3

from core.networks import Actor, logits_to_action
from core.orbit_wars_jax import setup, step, EnvAction, MAX_BODIES, MAX_FLEETS
from core.rollout_utils import calculate_intercept_angle, build_obs_arrays, active_body_mask
from scripts.jax_visualizer import jax_states_to_kaggle_env

# ── Constants ──────────────────────────────────────────────────────────────────
HIDDEN_DIM    = 48
NUM_SA_LAYERS = 6
MAX_TARGETS   = 64          # 60 planet/comet slots + 4 DS slots
R2_PREFIX     = 'rl_v25'
PLAYER_COLORS = ['#0072B2', '#D55E00', '#009E73', '#F0E442']
BUILTIN_TAGS  = {'sniper', 'random'}

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--players', type=int, default=4)
parser.add_argument('--agents',  nargs='+', required=True)
args = parser.parse_args()

num_players = args.players
assert len(args.agents) == num_players, \
    f"Expected {num_players} agents, got {len(args.agents)}: {args.agents}"
print(f"Generating {num_players}-player replay: {args.agents}")

# ── R2 client (uses env vars set by GitHub Actions secrets) ───────────────────
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['R2_ENDPOINT_URL'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    region_name='auto',
)
bucket = os.environ['R2_BUCKET_NAME']

# ── Actor graph template ───────────────────────────────────────────────────────
_proto = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
actor_graph, _ = nnx.split(_proto)

def np_to_params(np_dict):
    return jax.tree_util.tree_map(jnp.array, np_dict)

# ── Load a HoF agent from R2 ───────────────────────────────────────────────────
def load_hof_actor(tag):
    modes = ['4p', '2p'] if num_players == 4 else ['2p', '4p']
    
    # Legacy tags on R2 don't actually have the 'legacy_' prefix in their filenames
    # because the migration script preserved the original v24 filenames.
    file_tag = tag[7:] if tag.startswith('legacy_') else tag
    
    keys_to_try = [f'{R2_PREFIX}/hof/{mode}/{file_tag}.pkl' for mode in modes] + [f'{R2_PREFIX}/hof/{file_tag}.pkl']
    for key in keys_to_try:
        local = f'/tmp/hof_agent_{tag}.pkl'
        try:
            print(f'  Downloading {key} ...')
            s3.download_file(bucket, key, local)
            with open(local, 'rb') as f:
                raw = pickle.load(f)
            params_np = raw['params'] if isinstance(raw, dict) and 'params' in raw else raw
            return nnx.merge(actor_graph, np_to_params(params_np))
        except Exception as e:
            print(f'  Not found at {key}: {e}')
    raise RuntimeError(f'Could not load agent "{tag}" from R2 (checked flat and mode dirs).')

# ── Build per-player agent list ────────────────────────────────────────────────
actors = []
agent_labels = []
for tag in args.agents:
    if tag in BUILTIN_TAGS:
        actors.append(tag)
        agent_labels.append(tag.title())
    else:
        actors.append(load_hof_actor(tag))
        agent_labels.append(tag)
print("Agents ready:", agent_labels)

# ── Rule-based JAX agents ──────────────────────────────────────────────────────
# These are called inside lax.scan so must be pure JAX.
# `pid` is a static Python int (loop variable), so partial-application is fine.

def _sniper_step(state, ep, pid):
    """Send half ships from each owned planet to nearest active non-owned body."""
    active = active_body_mask(state, ep)           # [MAX_BODIES]
    owned  = (state.planet_owners == pid) & active # [MAX_BODIES]
    ships  = state.planet_ships                    # [MAX_BODIES]

    coords = state.planet_coords                   # [MAX_BODIES, 2]
    x, y   = coords[:, 0], coords[:, 1]
    dx = x[:, None] - x[None, :]                   # [MAX_BODIES, MAX_BODIES]
    dy = y[:, None] - y[None, :]
    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)

    not_mine = (state.planet_owners != pid) & active
    dist_masked = jnp.where(not_mine[None, :], dist, 1e9)
    target_idx  = jnp.argmin(dist_masked, axis=1) # [MAX_BODIES]

    ships_to_send = (ships // 2).astype(jnp.int32)
    target_oh = jax.nn.one_hot(target_idx, MAX_TARGETS)          # [MAX_BODIES, 64]
    s = jnp.where(owned[:, None], ships_to_send[:, None] * target_oh, 0).astype(jnp.int32)

    tgt_x = x[target_idx]
    tgt_y = y[target_idx]
    angle_to_tgt = jnp.arctan2(tgt_y - y, tgt_x - x)            # [MAX_BODIES]
    a = jnp.broadcast_to(angle_to_tgt[:, None], (MAX_BODIES, MAX_TARGETS))
    a = jnp.where(owned[:, None], a, 0.0)

    return s, a

def _random_step(state, ep, pid, key):
    """Each owned planet sends random ships to a random active planet."""
    active = active_body_mask(state, ep)
    owned  = (state.planet_owners == pid) & active
    ships  = state.planet_ships

    k1, k2, k3 = jr.split(key, 3)
    target_idx  = jr.randint(k1, (MAX_BODIES,), 0, MAX_BODIES)
    ship_frac   = jr.uniform(k2, (MAX_BODIES,), minval=0.3, maxval=0.7)
    ships_send  = (ships * ship_frac).astype(jnp.int32)

    target_oh = jax.nn.one_hot(target_idx, MAX_TARGETS)
    s = jnp.where(owned[:, None], ships_send[:, None] * target_oh, 0).astype(jnp.int32)

    angles_raw = jr.uniform(k3, (MAX_BODIES,), minval=-jnp.pi, maxval=jnp.pi)
    a = jnp.broadcast_to(angles_raw[:, None], (MAX_BODIES, MAX_TARGETS))
    a = jnp.where(owned[:, None], a, 0.0)

    return s, a

# ── Rollout ────────────────────────────────────────────────────────────────────
def make_rollout(actors, num_players):
    @functools.partial(jax.jit, static_argnames=('num_players',))
    def rollout(key, num_players=num_players):
        def scan_step(carry, _):
            state, ep, key = carry
            key, sk = jr.split(key)
            subkeys = jr.split(sk, num_players)

            ages         = state.step - ep.comet_spawn_steps
            comet_active = ep.is_comet & (ages >= 0) & (ages < ep.body_lifespans)
            target_active = ep.is_static_planet | ep.is_orbiting_planet | comet_active
            target_mask_64 = jnp.concatenate([target_active, jnp.ones(4, dtype=bool)], axis=-1)

            total_ships  = jnp.zeros((MAX_BODIES, MAX_TARGETS), dtype=jnp.int32)
            total_angles = jnp.zeros((MAX_BODIES, MAX_TARGETS), dtype=jnp.float32)

            for pid, actor in enumerate(actors):
                is_owner = (state.planet_owners == pid)[..., None]  # [MAX_BODIES, 1]

                if actor == 'sniper':
                    s, a = _sniper_step(state, ep, pid)
                elif actor == 'random':
                    s, a = _random_step(state, ep, pid, subkeys[pid])
                else:
                    # NN actor (nnx.Module captured as closure constant)
                    planets, fleets, p_mask, f_mask = build_obs_arrays(state, ep, pid, num_players)
                    logits   = actor(planets, fleets, planet_mask=p_mask, fleet_mask=f_mask)
                    nn_ships, ds_angles = logits_to_action(logits, state.planet_ships.astype(jnp.float32))
                    nn_ships = jnp.where(target_mask_64[None, :], nn_ships, 0.0)
                    state_b  = tree_map(lambda x: x[None], state)
                    ep_b     = tree_map(lambda x: x[None], ep)
                    intercept = calculate_intercept_angle(state_b, ep_b, nn_ships[None, ..., :60])[0]
                    ds_world  = ds_angles + pid * 2.0 * jnp.pi / num_players
                    a = jnp.concatenate([intercept, ds_world], axis=-1)
                    s = jnp.where(nn_ships < 1.0, 0.0, nn_ships).astype(jnp.int32)

                # mask inactive targets
                s = jnp.where(target_mask_64[None, :], s, 0)

                total_ships  = total_ships  + jnp.where(is_owner, s,              0)
                total_angles = total_angles + jnp.where(is_owner, a.astype(jnp.float32), 0.0)

            action = EnvAction(ships=total_ships, angle=total_angles)
            next_state, _, _, _ = step(state, ep, action, num_players)
            return (next_state, ep, key), state

        init_state, ep = setup(key, num_players)
        _, history = jax.lax.scan(scan_step, (init_state, ep, key), None, length=500)
        return history, ep

    return rollout

print("Compiling rollout (may take 1–3 min on CPU)...")
t0 = time.time()
rollout_fn = make_rollout(actors, num_players)
key = jr.PRNGKey(int(time.time()))
history, ep = rollout_fn(key, num_players)
print(f"Rollout complete in {time.time() - t0:.1f}s")

states_list = [tree_map(lambda x: x[t], history) for t in range(500)]
env = jax_states_to_kaggle_env(states_list, ep)

# ── Final stats ────────────────────────────────────────────────────────────────
def compute_final_stats(states_list, num_players, labels):
    final = states_list[-1]
    results = []
    for pid in range(num_players):
        p_ships = int(jnp.sum(jnp.where(final.planet_owners == pid, final.planet_ships, 0)))
        f_ships = int(jnp.sum(jnp.where(final.fleet_owners  == pid, final.fleet_ship_count, 0)))
        results.append({
            'player': pid, 'name': labels[pid], 'color': PLAYER_COLORS[pid],
            'planet_ships': p_ships, 'fleet_ships': f_ships, 'total': p_ships + f_ships,
        })
    results.sort(key=lambda x: -x['total'])
    for rank, r in enumerate(results):
        r['rank'] = rank + 1
    return results

stats = compute_final_stats(states_list, num_players, agent_labels)
print("Final standings:", [(r['name'], r['rank'], r['total']) for r in stats])

# ── Write outputs ──────────────────────────────────────────────────────────────
os.makedirs("public", exist_ok=True)

with open("public/orbit_wars_replay.html", "w") as f:
    f.write(env.render(mode="html", width=800, height=800))
print("Written: public/orbit_wars_replay.html")

meta = {
    'players':      num_players,
    'agents':       args.agents,
    'labels':       agent_labels,
    'stats':        stats,
    'generated_at': time.time(),
}
with open("public/replay_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Written: public/replay_meta.json")
print("Done.")
