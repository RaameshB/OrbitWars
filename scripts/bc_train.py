"""
Behavioral Cloning from leaderboard replay data.

Usage:
  uv run python scripts/bc_train.py --preprocess --data-dir /tmp/orbit-wars-parquet
  uv run python scripts/bc_train.py --train --bc-data /tmp/bc_data.npz --out weights_bc.pkl
  uv run python scripts/bc_train.py --pretrain-critic --bc-data /tmp/bc_data.npz
  uv run python scripts/bc_train.py --preprocess --train --pretrain-critic
"""

import os, sys, math, pickle, argparse, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None

def _bar(iterable, **kwargs):
    if _tqdm is not None:
        return _tqdm(iterable, dynamic_ncols=True, leave=False, unit='batch', **kwargs)
    return iterable

import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--preprocess',     action='store_true')
parser.add_argument('--train',          action='store_true')
parser.add_argument('--pretrain-critic',action='store_true')
parser.add_argument('--data-dir',       default='/tmp/orbit-wars-parquet')
parser.add_argument('--bc-data',        default='/tmp/bc_data.npz')
parser.add_argument('--out',            default='/tmp/weights_bc.pkl')
parser.add_argument('--critic-out',     default='/tmp/weights_bc_critic.pkl')
parser.add_argument('--top-k',          type=int,   default=10)
parser.add_argument('--min-games',      type=int,   default=20)
parser.add_argument('--epochs',         type=int,   default=50)
parser.add_argument('--critic-epochs',  type=int,   default=30)
parser.add_argument('--batch-size',     type=int,   default=1024)
parser.add_argument('--lr',             type=float, default=3e-3)
parser.add_argument('--weight-decay',   type=float, default=1e-2,
                    help='AdamW weight decay (grokking-inspired; applied to 1D params via Muon)')
parser.add_argument('--gamma',          type=float, default=0.99,
                    help='Discount factor for Monte Carlo returns in critic pretraining')
parser.add_argument('--val-frac',       type=float, default=0.1,
                    help='Fraction of data held out for validation loss tracking')
parser.add_argument('--warmup-epochs',  type=int,   default=5,
                    help='Linear LR warmup from lr/100 to lr over this many epochs (one-cycle phase 1)')
parser.add_argument('--alpha-lr-scale', type=float, default=0.1,
                    help='ReZero α LR as fraction of peak --lr; kept constant throughout (paper §E.2)')
parser.add_argument('--resume',         action='store_true',
                    help='Resume from local checkpoint saved alongside --out')
parser.add_argument('--checkpoint-every',type=int,   default=30,
                    help='Upload BC weights to R2 every N epochs (0 = disabled)')
parser.add_argument('--r2-prefix',       type=str,   default='bc_v9',
                    help='R2 key prefix for checkpoint uploads (e.g. bc_v9)')
parser.add_argument('--eval-every',     type=int,   default=1,
                    help='Run val loss every N epochs (set >1 to save compute)')
args = parser.parse_args()

HIDDEN_DIM    = 48
NUM_SA_LAYERS = 6
SHIP_SPEED    = 6.0
MAX_FLEET_OBS = 64

# ---------------------------------------------------------------------------
# ReZero alpha logging — all scalars (ndim==0) in the params pytree are alphas;
# every other param in this architecture is at least 1-D.
# Logs each layer by name plus a summary line.
# ---------------------------------------------------------------------------
def _parse_rezero(params):
    """Return list of (name, value) for all ReZero alpha scalars."""
    import jax
    named = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if not (hasattr(leaf, 'shape') and leaf.ndim == 0):
            continue
        parts = []
        for key in path:
            s = str(key)
            for wrapper in ("DictKey(key='", "DictKey('", "FlattenedIndexKey(",
                            "SequenceKey(idx=", "GetitemKey(key="):
                if s.startswith(wrapper):
                    s = s[len(wrapper):].rstrip(")'")
                    break
            parts.append(s)
        named.append(('.'.join(parts), float(leaf)))
    return named


def _rezero_summary(params) -> str:
    """Compact one-line ReZero summary: one α per block, grouped by CA vs SA."""
    named = _parse_rezero(params)
    if not named:
        return ''

    groups = {'ca': [], 'sa': []}
    for name, v in named:
        if 'ca_block' in name or 'cross_attention_block' in name:
            groups['ca'].append(v)
        else:
            groups['sa'].append(v)

    def fmt(vals):
        if not vals:
            return '—'
        if len(vals) == 1:
            return f'{vals[0]:+.3f}'
        a = np.array(vals)
        return f'{a.mean():+.3f}±{a.std():.3f}'

    all_vals = np.array([v for _, v in named])
    threshold = 0.1 * np.abs(all_vals).max() if np.abs(all_vals).max() > 0 else 0.01
    dead = (np.abs(all_vals) < threshold).sum()
    return (f"α dead={dead}/{len(named)} "
            f"ca={fmt(groups['ca'])} sa={fmt(groups['sa'])}")


def _log_rezero(params, label: str = ''):
    """Full per-layer ReZero log with bar chart (used every 10 epochs)."""
    named = _parse_rezero(params)
    if not named:
        return
    tag = f"[{label}] " if label else ""
    vals = np.array([v for _, v in named])
    threshold = 0.1 * np.abs(vals).max() if np.abs(vals).max() > 0 else 0.01
    dead = (np.abs(vals) < threshold).sum()
    print(f"  {tag}ReZero α  dead={dead}/{len(vals)}  "
          f"mean={vals.mean():.4f}  max={vals.max():.4f}")
    for name, v in named:
        bar = '█' * int(min(abs(v) * 200, 30))
        sign = '+' if v >= 0 else '-'
        print(f"    {name:<55s} {sign}{abs(v):.5f}  {bar}")


def _upload_to_r2(*local_paths, prefix='bc'):
    """Upload files to R2 under the given prefix. Silently skips if creds missing."""
    endpoint = os.environ.get('R2_ENDPOINT_URL', '')
    bucket   = os.environ.get('R2_BUCKET_NAME', '')
    if not endpoint or not bucket:
        print('  [R2] skipping upload — R2_ENDPOINT_URL / R2_BUCKET_NAME not set')
        return
    aws_env = {
        **os.environ,
        'AWS_ACCESS_KEY_ID':     os.environ.get('R2_ACCESS_KEY_ID', ''),
        'AWS_SECRET_ACCESS_KEY': os.environ.get('R2_SECRET_ACCESS_KEY', ''),
        'AWS_DEFAULT_REGION':    'auto',
    }
    for path in local_paths:
        if not os.path.exists(path):
            continue
        key = f's3://{bucket}/{prefix}/{os.path.basename(path)}'
        result = subprocess.run(
            ['aws', 's3', 'cp', path, key, '--endpoint-url', endpoint],
            capture_output=True, text=True, env=aws_env
        )
        if result.returncode == 0:
            print(f'  [R2] uploaded {os.path.basename(path)} → {key}')
        else:
            print(f'  [R2] upload failed for {path}: {result.stderr.strip()}')


# ---------------------------------------------------------------------------
# Fleet speed formula — matches orbit_wars_jax.py line 730 exactly
# ---------------------------------------------------------------------------
def _fleet_speed_batch(n_ships_arr):
    # Truncate to int before speed formula — matches game's int-cast behaviour
    n = np.maximum(np.floor(np.asarray(n_ships_arr, dtype=np.float32)), 1.0)
    raw = 1.0 + (SHIP_SPEED - 1.0) * (np.log(n) / np.log(1000.0)) ** 1.5
    return np.minimum(raw, SHIP_SPEED)


# ---------------------------------------------------------------------------
# Vectorized ray-cast: classify all actions in a tick simultaneously
# ---------------------------------------------------------------------------
def _classify_batch(src_xs, src_ys, angles, planet_xs, planet_ys, planet_rs, src_idxs):
    """[n_acts] → [n_acts] target slot indices, -1 for deep space."""
    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]

    dx = planet_xs[None, :] - src_xs[:, None]
    dy = planet_ys[None, :] - src_ys[:, None]

    proj    = dx * cos_a + dy * sin_a
    perp_sq = dx*dx + dy*dy - proj*proj

    n_pl      = len(planet_xs)
    self_mask = np.arange(n_pl)[None, :] == src_idxs[:, None]
    hits      = (proj > 0) & (perp_sq < planet_rs[None, :]**2) & ~self_mask

    proj_masked = np.where(hits, proj, np.inf)
    best = np.argmin(proj_masked, axis=1)
    return np.where(hits.any(axis=1), best, -1)


# ---------------------------------------------------------------------------
# PREPROCESS  (Polars + vectorised numpy — no iterrows)
# ---------------------------------------------------------------------------
def preprocess():
    import polars as pl

    data_dir = args.data_dir
    print(f"Loading parquet tables from {data_dir}...")
    t0 = time.time()
    episodes     = pl.read_parquet(f'{data_dir}/episodes.parquet')
    player_eps   = pl.read_parquet(f'{data_dir}/player_episodes.parquet')
    actions_df   = pl.read_parquet(f'{data_dir}/actions.parquet')
    ep_planets   = pl.read_parquet(f'{data_dir}/episode_planets.parquet')
    planet_state = pl.read_parquet(f'{data_dir}/planet_state.parquet')
    print(f"  loaded {len(planet_state):,} planet-state rows in {time.time()-t0:.1f}s")

    # Top-K players by win rate
    stats = (player_eps
        .group_by('name')
        .agg(games=pl.len(), wins=pl.col('is_winner').sum())
        .with_columns(win_rate=pl.col('wins') / pl.col('games'))
        .filter(pl.col('games') >= args.min_games)
        .sort('win_rate', descending=True)
        .head(args.top_k))
    top_players = stats['name'].to_list()
    print(f"Top-{args.top_k} players: {top_players}")

    top_pe = (player_eps
        .filter(pl.col('name').is_in(top_players))
        .join(episodes.select(['episode_id', 'n_players', 'angular_velocity']), on='episode_id'))

    eid_list     = top_pe['episode_id'].unique().to_list()
    planet_state = planet_state.filter(pl.col('episode_id').is_in(eid_list))
    actions_df   = actions_df.filter(pl.col('episode_id').is_in(eid_list))

    # Normalize angles in bulk
    TAU = 2 * math.pi
    actions_df = actions_df.with_columns(
        angle=((pl.col('angle') + math.pi) % TAU) - math.pi
    )

    # Join planet topology into planet_state once
    join_on = ['episode_id', 'planet_id'] if 'episode_id' in ep_planets.columns else ['planet_id']
    ps_full = (planet_state
        .join(ep_planets.select(join_on + ['radius', 'production']), on=join_on, how='left')
        .with_columns([
            pl.col('radius').fill_null(0.0).cast(pl.Float32),
            pl.col('production').fill_null(0.0).cast(pl.Float32),
            pl.col('x').cast(pl.Float32),
            pl.col('y').cast(pl.Float32),
            pl.col('ships').cast(pl.Float32),
            pl.col('owner').cast(pl.Int32),
            pl.col('planet_id').cast(pl.Int32),
        ]))

    # Build (episode_id, tick) → numpy arrays dict
    print("Indexing planet state by (episode, tick)...")
    t1 = time.time()
    ps_grouped = (ps_full
        .sort(['episode_id', 'tick', 'planet_id'])
        .group_by(['episode_id', 'tick'], maintain_order=True)
        .agg([
            pl.col('planet_id'), pl.col('owner'),
            pl.col('x'), pl.col('y'), pl.col('ships'),
            pl.col('radius'), pl.col('production'),
        ]))

    ps_dict = {}
    for row in ps_grouped.iter_rows(named=True):
        ps_dict[(row['episode_id'], row['tick'])] = (
            np.asarray(row['planet_id'], dtype=np.int32),
            np.asarray(row['owner'],     dtype=np.int32),
            np.asarray(row['x'],         dtype=np.float32),
            np.asarray(row['y'],         dtype=np.float32),
            np.asarray(row['ships'],     dtype=np.float32),
            np.asarray(row['radius'],    dtype=np.float32),
            np.asarray(row['production'],dtype=np.float32),
        )
    print(f"  {len(ps_dict):,} (episode, tick) groups in {time.time()-t1:.1f}s")

    # Episode max tick (for computing discounted returns)
    max_tick_by_ep = {}
    for (eid, tick) in ps_dict:
        if eid not in max_tick_by_ep or tick > max_tick_by_ep[eid]:
            max_tick_by_ep[eid] = tick

    # Outcome lookup: (episode_id, slot) → is_winner float
    outcome_lookup = {
        (row[0], row[1]): float(row[2])
        for row in top_pe.select(['episode_id', 'slot', 'is_winner']).iter_rows()
    }

    # Build (episode_id, slot, tick) → action arrays dict
    print("Indexing actions by (episode, slot, tick)...")
    t2 = time.time()
    acts_grouped = (actions_df
        .with_columns([
            pl.col('src_planet_id').cast(pl.Int32),
            pl.col('n_ships').cast(pl.Float32),
            pl.col('angle').cast(pl.Float32),
            pl.col('slot').cast(pl.Int32),
        ])
        .group_by(['episode_id', 'slot', 'tick'])
        .agg([pl.col('src_planet_id'), pl.col('angle'), pl.col('n_ships')]))

    acts_dict  = {}
    tick_index = {}
    for row in acts_grouped.iter_rows(named=True):
        key = (row['episode_id'], row['slot'], row['tick'])
        acts_dict[key] = (
            np.asarray(row['src_planet_id'], dtype=np.int32),
            np.asarray(row['angle'],         dtype=np.float32),
            np.asarray(row['n_ships'],       dtype=np.float32),
        )
        ep_slot = (row['episode_id'], row['slot'])
        if ep_slot not in tick_index:
            tick_index[ep_slot] = []
        tick_index[ep_slot].append(row['tick'])
    print(f"  {len(acts_dict):,} (episode, slot, tick) groups in {time.time()-t2:.1f}s")

    # Build per-episode fleet transit event table (all players, all ticks)
    # Each event: [launch_tick, arrival_tick, owner_slot, launch_x, launch_y, angle, n_ships]
    print("Building fleet transit events...")
    t_fleet = time.time()
    all_ep_acts = (actions_df
        .with_columns([
            pl.col('src_planet_id').cast(pl.Int32),
            pl.col('n_ships').cast(pl.Float32),
            pl.col('slot').cast(pl.Int32),
        ])
        .group_by(['episode_id', 'tick', 'slot'])
        .agg([pl.col('src_planet_id'), pl.col('angle'), pl.col('n_ships')]))

    _ep_events = {}
    for row in all_ep_acts.iter_rows(named=True):
        eid, tick, fslot = row['episode_id'], row['tick'], int(row['slot'])
        if (eid, tick) not in ps_dict:
            continue
        pids, owners, xs, ys, _, radii, _ = ps_dict[(eid, tick)]
        pid_to_idx = {int(p): i for i, p in enumerate(pids)}
        src_pids = np.asarray(row['src_planet_id'], dtype=np.int32)
        angles_a = np.asarray(row['angle'],          dtype=np.float32)
        nships_a = np.asarray(row['n_ships'],         dtype=np.float32)
        src_idxs = np.fromiter((pid_to_idx.get(int(p), -1) for p in src_pids),
                               dtype=np.int32, count=len(src_pids))
        valid = src_idxs >= 0
        if not valid.any():
            continue
        sv, av, nv = src_idxs[valid], angles_a[valid], nships_a[valid]
        tgt_slots = _classify_batch(xs[sv], ys[sv], av, xs, ys, radii, sv)
        hit = tgt_slots >= 0
        if not hit.any():
            continue
        sv, av, nv, tv = sv[hit], av[hit], nv[hit], tgt_slots[hit]
        # Subtract both radii: fleet travels from source surface to target surface
        dist  = np.sqrt((xs[sv]-xs[tv])**2 + (ys[sv]-ys[tv])**2) - radii[sv] - radii[tv]
        speed = _fleet_speed_batch(nv)
        arr_dt = (np.ceil(np.maximum(dist, 0.0) / speed) + 1).astype(np.int32)
        if eid not in _ep_events:
            _ep_events[eid] = []
        for i in range(len(sv)):
            # columns: launch_tick, arrival_tick, owner_slot, launch_x, launch_y, angle, n_ships, src_radius
            _ep_events[eid].append((
                float(tick), float(tick + arr_dt[i]), float(fslot),
                float(xs[sv[i]]), float(ys[sv[i]]), float(av[i]), float(nv[i]),
                float(radii[sv[i]]),
            ))

    fleet_arrays_by_ep = {eid: np.array(evs, dtype=np.float32)
                          for eid, evs in _ep_events.items()}
    del _ep_events
    n_launches = sum(len(v) for v in fleet_arrays_by_ep.values())
    print(f"  {n_launches:,} fleet launches indexed in {time.time()-t_fleet:.1f}s")

    # Main loop — vectorised per tick
    print("Processing ticks (vectorised)...")
    t3 = time.time()
    all_planet_obs   = []
    all_ships_target = []
    all_owner_mask   = []
    all_returns      = []
    all_fleet_obs    = []
    all_fleet_mask   = []
    processed_ticks  = 0

    for eid, slot, n_p in top_pe.select(['episode_id', 'slot', 'n_players']).iter_rows():
        ep_slot = (eid, int(slot))
        if ep_slot not in tick_index:
            continue

        theta = -slot * (2.0 * math.pi / n_p)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        outcome   = outcome_lookup.get(ep_slot, 0.0)
        max_tick  = max_tick_by_ep.get(eid, 400)

        for tick in tick_index[ep_slot]:
            if (eid, tick) not in ps_dict:
                continue

            pids, owners, xs, ys, ships_arr, radii, prods = ps_dict[(eid, tick)]
            n_pl = len(pids)
            if n_pl == 0:
                continue

            # Planet obs — vectorised
            slots_arr = np.arange(n_pl)
            dx = xs - 50.0;  dy = ys - 50.0
            rot_x = dx * cos_t - dy * sin_t + 50.0
            rot_y = dx * sin_t + dy * cos_t + 50.0
            rel_owners = np.where(owners == slot, 1.0,
                         np.where(owners < 0,    0.0, -1.0)).astype(np.float32)

            p_arr = np.zeros((60, 7), dtype=np.float32)
            p_arr[slots_arr, 0] = slots_arr.astype(np.float32)
            p_arr[slots_arr, 1] = rel_owners
            p_arr[slots_arr, 2] = rot_x
            p_arr[slots_arr, 3] = rot_y
            p_arr[slots_arr, 4] = radii
            p_arr[slots_arr, 5] = ships_arr
            p_arr[slots_arr, 6] = prods

            owned_mask = owners == slot
            if not owned_mask.any():
                continue
            owned_slots_arr = slots_arr[owned_mask]

            src_pids_act, angles_act, nships_act = acts_dict[(eid, slot, tick)]
            pid_to_idx = {int(p): i for i, p in enumerate(pids)}
            src_idxs = np.fromiter(
                (pid_to_idx.get(int(p), -1) for p in src_pids_act),
                dtype=np.int32, count=len(src_pids_act)
            )

            valid = (src_idxs >= 0) & np.isin(src_idxs, owned_slots_arr)
            if not valid.any():
                continue

            src_v    = src_idxs[valid]
            angles_v = angles_act[valid]
            nships_v = nships_act[valid]

            tgt_slots = _classify_batch(
                xs[src_v], ys[src_v], angles_v,
                xs[:n_pl], ys[:n_pl], radii[:n_pl], src_v
            )

            hit_mask = tgt_slots >= 0
            if not hit_mask.any():
                continue

            ships_target = np.zeros((60, 60), dtype=np.float32)
            np.add.at(ships_target, (src_v[hit_mask], tgt_slots[hit_mask]), nships_v[hit_mask])

            if ships_target.sum() == 0:
                continue

            owner_mask = np.zeros(60, dtype=bool)
            owner_mask[owned_slots_arr] = True

            # Monte Carlo return: γ^(T-t) * outcome
            ret = (args.gamma ** (max_tick - tick)) * outcome

            # Fleet obs — in-transit fleets at this tick across all players
            fleet_arr      = np.zeros((MAX_FLEET_OBS, 6), dtype=np.float32)
            fleet_mask_arr = np.zeros(MAX_FLEET_OBS,      dtype=bool)
            if eid in fleet_arrays_by_ep:
                fa   = fleet_arrays_by_ep[eid]
                in_t = fa[(fa[:, 0] <= tick) & (fa[:, 1] > tick)]
                if len(in_t) > MAX_FLEET_OBS:
                    in_t = in_t[:MAX_FLEET_OBS]
                K = len(in_t)
                if K > 0:
                    dt    = tick - in_t[:, 0]
                    speed = _fleet_speed_batch(in_t[:, 6])
                    # Start from planet surface (launch_x/y are planet centers)
                    r_src = in_t[:, 7]
                    cx    = in_t[:, 3] + np.cos(in_t[:, 5]) * (r_src + speed * dt)
                    cy    = in_t[:, 4] + np.sin(in_t[:, 5]) * (r_src + speed * dt)
                    fdx   = cx - 50.0;  fdy = cy - 50.0
                    fleet_arr[:K, 0] = np.arange(K, dtype=np.float32)
                    fleet_arr[:K, 1] = np.where(in_t[:, 2] == slot, 1.0, -1.0)
                    fleet_arr[:K, 2] = in_t[:, 5] + theta
                    fleet_arr[:K, 3] = fdx * cos_t - fdy * sin_t + 50.0
                    fleet_arr[:K, 4] = fdx * sin_t + fdy * cos_t + 50.0
                    fleet_arr[:K, 5] = in_t[:, 6]
                    fleet_mask_arr[:K] = True

            all_planet_obs.append(p_arr.astype(np.float16))
            all_ships_target.append(ships_target.astype(np.float16))
            all_owner_mask.append(owner_mask)
            all_returns.append(np.float32(ret))
            all_fleet_obs.append(fleet_arr.astype(np.float16))
            all_fleet_mask.append(fleet_mask_arr)
            processed_ticks += 1

            if processed_ticks % 50_000 == 0 and processed_ticks > 0:
                print(f"  {processed_ticks:,} ticks | {time.time()-t3:.0f}s elapsed")

    print(f"\nProcessed {processed_ticks:,} ticks in {time.time()-t3:.1f}s")

    planet_obs_arr   = np.stack(all_planet_obs,   axis=0)
    ships_target_arr = np.stack(all_ships_target, axis=0)
    owner_mask_arr   = np.stack(all_owner_mask,   axis=0)
    returns_arr      = np.array(all_returns,      dtype=np.float32)
    fleet_obs_arr    = np.stack(all_fleet_obs,    axis=0)
    fleet_mask_arr   = np.stack(all_fleet_mask,   axis=0)

    active_fleets = fleet_mask_arr.sum(axis=1)
    print(f"  Fleet obs: mean={active_fleets.mean():.1f}  max={active_fleets.max()}  "
          f"zero={( active_fleets==0).sum()} ticks with no fleets")

    print(f"Saving to {args.bc_data}...")
    np.savez_compressed(
        args.bc_data,
        planet_obs=planet_obs_arr,
        ships_target=ships_target_arr,
        owner_mask=owner_mask_arr,
        returns=returns_arr,
        fleet_obs=fleet_obs_arr,
        fleet_mask=fleet_mask_arr,
    )
    size_mb = os.path.getsize(args.bc_data) / 1e6
    print(f"Saved {args.bc_data} ({size_mb:.1f} MB)")
    return planet_obs_arr, ships_target_arr, owner_mask_arr, returns_arr, fleet_obs_arr, fleet_mask_arr


# ---------------------------------------------------------------------------
# TRAIN ACTOR (BC)
# ---------------------------------------------------------------------------
def train(planet_obs=None, ships_target=None, owner_mask=None, returns=None,
          fleet_obs=None, fleet_mask=None):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from core.networks import Actor

    if planet_obs is None:
        print(f"Loading {args.bc_data}...")
        data = np.load(args.bc_data)
        planet_obs   = data['planet_obs'].astype(np.float32)
        ships_target = data['ships_target'].astype(np.float32)
        owner_mask   = data['owner_mask']
        if 'fleet_obs' in data:
            fleet_obs  = data['fleet_obs'].astype(np.float32)
            fleet_mask = data['fleet_mask']

    N = len(planet_obs)
    if fleet_obs is None:
        fleet_obs  = np.zeros((N, MAX_FLEET_OBS, 6), dtype=np.float32)
        fleet_mask = np.zeros((N, MAX_FLEET_OBS),    dtype=bool)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_val      = max(1, int(N * args.val_frac))
    val_idx    = perm[:n_val]
    train_idx  = perm[n_val:]
    N_train    = len(train_idx)
    print(f"Training actor on {N_train:,} train / {n_val:,} val ticks | "
          f"batch={args.batch_size} | epochs={args.epochs} | wd={args.weight_decay}")

    actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
    actor_graph, params = nnx.split(actor)

    steps_per_epoch = N_train // args.batch_size
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    decay_steps     = max(total_steps - warmup_steps, 1)
    min_lr          = args.lr * 0.01
    alpha_lr        = args.lr * args.alpha_lr_scale

    # One-cycle schedule for main weights: linear warmup → cosine decay
    main_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(min_lr, args.lr, warmup_steps),
            optax.cosine_decay_schedule(args.lr, decay_steps, alpha=min_lr / args.lr),
        ],
        boundaries=[warmup_steps],
    )
    # α (ReZero residual weights) get a fixed low LR — they cannot tolerate large LR swings
    main_opt  = optax.contrib.muon(learning_rate=main_schedule, weight_decay=args.weight_decay)
    alpha_opt = optax.adam(learning_rate=alpha_lr)

    def _param_labels(p):
        return jax.tree_util.tree_map(lambda x: 'alpha' if x.ndim == 0 else 'main', p)

    opt       = optax.multi_transform({'main': main_opt, 'alpha': alpha_opt}, _param_labels)
    opt_state = opt.init(params)

    resume_path = args.out.replace('.pkl', '_resume.pkl')
    start_epoch = 0
    history     = []
    best_val    = float('inf')
    hist_path   = args.out.replace('.pkl', '_history.json')

    if args.resume and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = pickle.load(open(resume_path, 'rb'))
        start_epoch = ckpt['epoch']
        best_val    = ckpt['best_val']
        history     = ckpt['history']
        params      = jax.tree_util.tree_map(jnp.array, ckpt['params'])
        opt_state   = jax.tree_util.tree_map(jnp.array, ckpt['opt_state'])
        print(f"  Resumed at epoch {start_epoch} | best_val={best_val:.4f}")

    def _xent(planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b, actor_inst):
        pmask       = planets_b[..., 4] > 0   # radius > 0 — excludes zero-padded ghost slots
        logits      = actor_inst(planets_b, fleets_b, planet_mask=pmask, fleet_mask=fmask_b)
        total_sent  = ships_tgt_b.sum(axis=-1, keepdims=True)
        target_dist = ships_tgt_b / (total_sent + 1e-8)
        log_probs   = jax.nn.log_softmax(logits[..., :60], axis=-1)
        xent        = -jnp.sum(target_dist * log_probs, axis=-1)
        has_action  = (total_sent[..., 0] > 0) & owner_mask_b
        return jnp.sum(xent * has_action) / (jnp.sum(has_action) + 1e-8)

    @jax.jit
    def train_step(params, opt_state, planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b):
        def loss_fn(params):
            return _xent(planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b,
                         nnx.merge(actor_graph, params))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    @jax.jit
    def eval_step(params, planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b):
        return _xent(planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b,
                     nnx.merge(actor_graph, params))

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        shuffled = rng.permutation(N_train)

        train_losses = []
        batch_iter = _bar(range(0, N_train - args.batch_size, args.batch_size),
                          desc=f'E{epoch+1}/{args.epochs}')
        for start in batch_iter:
            bi = train_idx[shuffled[start:start + args.batch_size]]
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(planet_obs[bi]),
                jnp.array(ships_target[bi]),
                jnp.array(owner_mask[bi]),
                jnp.array(fleet_obs[bi]),
                jnp.array(fleet_mask[bi]),
            )
            loss_val = float(loss)
            train_losses.append(loss_val)
            if _tqdm is not None:
                batch_iter.set_postfix(loss=f'{loss_val:.4f}')

        train_loss = float(np.mean(train_losses))
        do_eval    = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1

        if do_eval:
            val_losses = []
            for start in range(0, n_val - args.batch_size, args.batch_size):
                bi = val_idx[start:start + args.batch_size]
                val_losses.append(float(eval_step(
                    params,
                    jnp.array(planet_obs[bi]),
                    jnp.array(ships_target[bi]),
                    jnp.array(owner_mask[bi]),
                    jnp.array(fleet_obs[bi]),
                    jnp.array(fleet_mask[bi]),
                )))
            val_loss = float(np.mean(val_losses))
            print(f"Epoch {epoch+1:>3d}/{args.epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f} | {time.time()-t0:.1f}s  {_rezero_summary(params)}")
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
            with open(hist_path, 'w') as f:
                json.dump({'actor': history}, f)
        else:
            val_loss = best_val  # use last known for checkpoint gating
            print(f"Epoch {epoch+1:>3d}/{args.epochs} | train={train_loss:.4f} | {time.time()-t0:.1f}s  {_rezero_summary(params)}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log_rezero(params, label=f'actor epoch {epoch+1}')

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            ckpt = {
                'epoch':     epoch + 1,
                'best_val':  best_val,
                'history':   history,
                'params':    jax.tree_util.tree_map(np.array, params),
                'opt_state': jax.tree_util.tree_map(np.array, opt_state),
            }
            with open(resume_path, 'wb') as f:
                pickle.dump(ckpt, f)
            _upload_to_r2(args.out, hist_path, resume_path, prefix=args.r2_prefix)

        if val_loss < best_val:
            best_val = val_loss
            final_actor = nnx.merge(actor_graph, params)
            _, best_params = nnx.split(final_actor)
            np_params = jax.tree_util.tree_map(np.array, best_params)
            with open(args.out, 'wb') as f:
                pickle.dump(np_params, f)

    _log_rezero(params, label='actor final')
    print(f"\nDone. Best val loss: {best_val:.4f} | Weights → {args.out} | History → {hist_path}")


# ---------------------------------------------------------------------------
# PRETRAIN CRITIC  (supervised Monte Carlo returns)
# ---------------------------------------------------------------------------
def pretrain_critic(planet_obs=None, returns=None, fleet_obs=None, fleet_mask=None):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from core.networks import Critic

    if planet_obs is None:
        print(f"Loading {args.bc_data}...")
        data = np.load(args.bc_data)
        planet_obs = data['planet_obs'].astype(np.float32)
        returns    = data['returns'].astype(np.float32)
        if 'fleet_obs' in data:
            fleet_obs  = data['fleet_obs'].astype(np.float32)
            fleet_mask = data['fleet_mask']

    N = len(planet_obs)
    if fleet_obs is None:
        fleet_obs  = np.zeros((N, MAX_FLEET_OBS, 6), dtype=np.float32)
        fleet_mask = np.zeros((N, MAX_FLEET_OBS),    dtype=bool)
    rng = np.random.default_rng(43)
    perm    = rng.permutation(N)
    n_val   = max(1, int(N * args.val_frac))
    val_idx = perm[:n_val]
    tr_idx  = perm[n_val:]
    N_train = len(tr_idx)
    print(f"Pretraining critic on {N_train:,} train / {n_val:,} val ticks | "
          f"batch={args.batch_size} | epochs={args.critic_epochs}")
    print(f"  Return stats: mean={returns.mean():.3f}  std={returns.std():.3f}  "
          f"min={returns.min():.3f}  max={returns.max():.3f}")

    critic = Critic(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(1))
    critic_graph, params = nnx.split(critic)

    steps_per_epoch = N_train // args.batch_size
    warmup_steps    = args.warmup_epochs * steps_per_epoch
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(args.lr * 0.1, args.lr, warmup_steps),
            optax.constant_schedule(args.lr),
        ],
        boundaries=[warmup_steps],
    )
    opt         = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    opt_state   = opt.init(params)

    actions_b = jnp.zeros((args.batch_size, 60, 72), dtype=jnp.float32)

    def _mse(planets_b, returns_b, fleets_b, fmask_b, critic_inst):
        pmask = planets_b[..., 5] >= 0
        q_val = critic_inst(planets_b, fleets_b, actions_b,
                            planet_mask=pmask, fleet_mask=fmask_b)
        return jnp.mean((q_val[..., 0] - returns_b) ** 2)

    @jax.jit
    def train_step(params, opt_state, planets_b, returns_b, fleets_b, fmask_b):
        def loss_fn(params):
            return _mse(planets_b, returns_b, fleets_b, fmask_b, nnx.merge(critic_graph, params))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    @jax.jit
    def eval_step(params, planets_b, returns_b, fleets_b, fmask_b):
        return _mse(planets_b, returns_b, fleets_b, fmask_b, nnx.merge(critic_graph, params))

    history   = []
    best_val  = float('inf')
    hist_path = args.out.replace('.pkl', '_history.json')

    for epoch in range(args.critic_epochs):
        t0 = time.time()
        shuffled = rng.permutation(N_train)

        train_losses = []
        batch_iter = _bar(range(0, N_train - args.batch_size, args.batch_size),
                          desc=f'C{epoch+1}/{args.critic_epochs}')
        for start in batch_iter:
            bi = tr_idx[shuffled[start:start + args.batch_size]]
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(planet_obs[bi]),
                jnp.array(returns[bi]),
                jnp.array(fleet_obs[bi]),
                jnp.array(fleet_mask[bi]),
            )
            loss_val = float(loss)
            train_losses.append(loss_val)
            if _tqdm is not None:
                batch_iter.set_postfix(loss=f'{loss_val:.4f}')

        train_loss = float(np.mean(train_losses))
        do_eval    = (epoch + 1) % args.eval_every == 0 or epoch == args.critic_epochs - 1

        if do_eval:
            val_losses = []
            for start in range(0, n_val - args.batch_size, args.batch_size):
                bi = val_idx[start:start + args.batch_size]
                val_losses.append(float(eval_step(
                    params,
                    jnp.array(planet_obs[bi]),
                    jnp.array(returns[bi]),
                    jnp.array(fleet_obs[bi]),
                    jnp.array(fleet_mask[bi]),
                )))
            val_loss = float(np.mean(val_losses))
            print(f"Epoch {epoch+1:>3d}/{args.critic_epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f} | {time.time()-t0:.1f}s  {_rezero_summary(params)}")
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
            existing = {}
            if os.path.exists(hist_path):
                with open(hist_path) as f:
                    existing = json.load(f)
            existing['critic'] = history
            with open(hist_path, 'w') as f:
                json.dump(existing, f)
        else:
            val_loss = best_val
            print(f"Epoch {epoch+1:>3d}/{args.critic_epochs} | train={train_loss:.4f} | {time.time()-t0:.1f}s  {_rezero_summary(params)}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log_rezero(params, label=f'critic epoch {epoch+1}')

        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            _upload_to_r2(args.critic_out, prefix=args.r2_prefix)

        if val_loss < best_val:
            best_val = val_loss
            final_critic = nnx.merge(critic_graph, params)
            _, best_params = nnx.split(final_critic)
            np_params = jax.tree_util.tree_map(np.array, best_params)
            with open(args.critic_out, 'wb') as f:
                pickle.dump(np_params, f)

    _log_rezero(params, label='critic final')
    print(f"\nDone. Best critic val MSE: {best_val:.4f} | Weights → {args.critic_out} | History → {hist_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if not args.preprocess and not args.train and not args.pretrain_critic:
        parser.print_help()
        sys.exit(1)

    preprocess_result = None
    if args.preprocess:
        preprocess_result = preprocess()

    # Load data once and share between actor + critic to avoid double I/O and double JAX init
    if (args.train or args.pretrain_critic) and preprocess_result is None:
        print(f"Loading {args.bc_data}...")
        _d = np.load(args.bc_data)
        _fo = _d['fleet_obs'].astype(np.float32)  if 'fleet_obs'  in _d else None
        _fm = _d['fleet_mask']                     if 'fleet_mask' in _d else None
        preprocess_result = (
            _d['planet_obs'].astype(np.float32),
            _d['ships_target'].astype(np.float32),
            _d['owner_mask'],
            _d['returns'].astype(np.float32),
            _fo,
            _fm,
        )
        print(f"  Loaded {len(preprocess_result[0]):,} samples")

    if args.train:
        train(*preprocess_result)

    if args.pretrain_critic:
        pretrain_critic(preprocess_result[0], preprocess_result[3],
                        preprocess_result[4], preprocess_result[5])
