"""
Behavioral Cloning from leaderboard replay data (v13.1).

Usage:
  uv run python scripts/bc_train.py --preprocess --data-dir /tmp/orbit-wars-parquet
  uv run python scripts/bc_train.py --train --bc-data /tmp/bc_data.npz --out weights_bc.pkl
  uv run python scripts/bc_train.py --pretrain-critic --bc-data /tmp/bc_data.npz
  uv run python scripts/bc_train.py --preprocess --train --pretrain-critic

v13.1 preprocessing changes vs v13:
  - ALL ticks per player (wait ticks included; diagonal = ships held = fire-control signal)
  - Enemy encoding: ranked distinct negative values by current ship strength
  - MAX_FLEET_OBS=128 (no pruning at 64)
  - top-k = 10 (kept low to avoid strategy poisoning; 4× augmentation gives ample data)
  - FFA oversampled 3× at training time (--ffa-weight 3) to balance 3:1 duel:ffa ratio
  - Hold-only ticks subsampled to 10% at training time (--hold-rate 0.1) to prevent degenerate hold policy
  - n_players saved per sample for weighted training sampler
  - On-the-fly 4-fold rotation augmentation during training
  - Training loss: diagonal included as valid target, has_ships mask replaces has_action
"""

import os, sys, math, pickle, argparse, time, json, subprocess, multiprocessing as mp
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
parser.add_argument('--ffa-weight',     type=int,   default=3,
                    help='Oversample 4-player FFA games by this factor at training time (balances ~3:1 duel:ffa ratio)')
parser.add_argument('--hold-rate',      type=float, default=0.1,
                    help='Fraction of hold-only ticks (no off-diagonal sends) to keep at training time (1.0 = keep all)')
parser.add_argument('--epochs',         type=int,   default=50)
parser.add_argument('--critic-epochs',  type=int,   default=30)
parser.add_argument('--batch-size',     type=int,   default=512)
parser.add_argument('--lr',             type=float, default=3e-4)
parser.add_argument('--weight-decay',   type=float, default=0.0)
parser.add_argument('--gamma',          type=float, default=0.99,
                    help='Discount factor for Monte Carlo returns in critic pretraining')
parser.add_argument('--val-frac',       type=float, default=0.1,
                    help='Fraction of data held out for validation loss tracking')
parser.add_argument('--warmup-epochs',  type=int,   default=5,
                    help='Linear LR warmup epochs')
parser.add_argument('--resume',         action='store_true',
                    help='Resume from local checkpoint saved alongside --out')
parser.add_argument('--checkpoint-every',type=int,   default=30,
                    help='Upload BC weights to R2 every N epochs (0 = disabled)')
parser.add_argument('--r2-prefix',       type=str,   default='bc_v13_1',
                    help='R2 key prefix for checkpoint uploads')
parser.add_argument('--transfer-from',   type=str,   default='',
                    help='Path to v13 weights.pkl to warm-start from (reinits action head + fleet MLPs)')
parser.add_argument('--eval-every',     type=int,   default=1,
                    help='Run val loss every N epochs (set >1 to save compute)')
parser.add_argument('--workers',        type=int,   default=max(1, mp.cpu_count() // 2),
                    help='Parallel workers for preprocessing (fork, COW shared arrays)')
parser.add_argument('--flush-every',    type=int,   default=50_000,
                    help='Flush partial results to disk every N samples per worker (caps peak RAM)')
parser.add_argument('--upload-to-kaggle', type=str, default='',
                    help='Kaggle dataset slug (owner/dataset-name) to upload bc_data npz as a new version')
args = parser.parse_args()

HIDDEN_DIM    = 48
NUM_SA_LAYERS = 6
SHIP_SPEED    = 6.0
MAX_FLEET_OBS = 128

# ---------------------------------------------------------------------------
# Shared state for forked preprocessing workers (populated before Pool creation;
# fork() gives each worker a COW view — no copying of the large dicts).
# ---------------------------------------------------------------------------
_G: dict = {}

# ---------------------------------------------------------------------------
# ReZero alpha logging — all scalars (ndim==0) in the params pytree are alphas;
# every other param in this architecture is at least 1-D.
# Logs each layer by name plus a summary line.
# ---------------------------------------------------------------------------
def _rezero_summary(params) -> str:
    return ''

def _log_rezero(params, label: str = ''):
    pass


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
# Preprocessing worker — called in forked subprocess, reads _G via COW.
# Processes a chunk of (eid, slot, n_p) tuples and writes results to a temp npz.
# ---------------------------------------------------------------------------
def _flush_chunk(lists, out_dir, worker_id, flush_idx):
    """Write accumulated lists to a numbered sub-chunk file and return its path."""
    out_path = os.path.join(out_dir, f'chunk_{worker_id:04d}_{flush_idx:04d}.npz')
    np.savez_compressed(
        out_path,
        planet_obs  = np.stack(lists[0]),
        ships_target= np.stack(lists[1]),
        owner_mask  = np.stack(lists[2]),
        returns     = np.array(lists[3], dtype=np.float32),
        fleet_obs   = np.stack(lists[4]),
        fleet_mask  = np.stack(lists[5]),
        n_players   = np.array(lists[6], dtype=np.int8),
    )
    return out_path


def _preprocess_chunk(args_tuple):
    chunk, worker_id, out_dir, shared_dir, gamma, mfo, flush_every = args_tuple

    # Load shared arrays via mmap — OS shares physical pages across all spawned workers
    def _m(name): return np.load(os.path.join(shared_dir, f'{name}.npy'), mmap_mode='r')

    grp_ep        = _m('grp_ep');    grp_tick   = _m('grp_tick')
    grp_starts    = _m('grp_starts')
    ps_pid        = _m('ps_pid');    ps_own     = _m('ps_own')
    ps_x          = _m('ps_x');      ps_y       = _m('ps_y')
    ps_shp        = _m('ps_shp');    ps_rad     = _m('ps_rad');   ps_prd = _m('ps_prd')
    unique_eps    = _m('unique_eps'); max_tick_vals = _m('max_tick_vals')
    act_grp_ep    = _m('act_grp_ep'); act_grp_slot = _m('act_grp_slot')
    act_grp_tick  = _m('act_grp_tick'); act_grp_starts = _m('act_grp_starts')
    act_pid = _m('act_pid'); act_ang = _m('act_ang'); act_nsh = _m('act_nsh')
    fleet_ep_ids  = _m('fleet_ep_ids');  fleet_offsets = _m('fleet_offsets')
    fleet_flat    = _m('fleet_flat')
    out_ep        = _m('out_ep');    out_slot = _m('out_slot');   out_val = _m('out_val')

    planet_obs_list   = []
    ships_target_list = []
    owner_mask_list   = []
    returns_list      = []
    fleet_obs_list    = []
    fleet_mask_list   = []
    n_players_list    = []

    out_paths   = []
    flush_idx   = 0
    total_count = 0

    def _maybe_flush():
        nonlocal flush_idx, total_count
        lists = [planet_obs_list, ships_target_list, owner_mask_list,
                 returns_list, fleet_obs_list, fleet_mask_list, n_players_list]
        n = len(planet_obs_list)
        if n >= flush_every:
            out_paths.append(_flush_chunk(lists, out_dir, worker_id, flush_idx))
            flush_idx   += 1
            total_count += n
            for lst in lists:
                lst.clear()

    for eid, slot, n_p in chunk:
        # Episode range in planet CSR
        ep_lo = int(np.searchsorted(grp_ep, eid, side='left'))
        ep_hi = int(np.searchsorted(grp_ep, eid, side='right'))
        if ep_lo == ep_hi:
            continue

        # max_tick via numpy lookup
        mt_idx   = int(np.searchsorted(unique_eps, eid))
        max_tick = int(max_tick_vals[mt_idx]) if (mt_idx < len(unique_eps) and unique_eps[mt_idx] == eid) else 400

        # (ep, slot) range in acts CSR — constant for this (eid, slot), cache outside tick loop
        ae_lo  = int(np.searchsorted(act_grp_ep, eid, side='left'))
        ae_hi  = int(np.searchsorted(act_grp_ep, eid, side='right'))
        asl_lo = ae_lo + int(np.searchsorted(act_grp_slot[ae_lo:ae_hi], slot, side='left'))
        asl_hi = ae_lo + int(np.searchsorted(act_grp_slot[ae_lo:ae_hi], slot, side='right'))

        theta    = -slot * (2.0 * math.pi / n_p)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # outcome: CSR lookup in sorted (out_ep, out_slot) arrays
        _oe_lo = int(np.searchsorted(out_ep, eid, side='left'))
        _oe_hi = int(np.searchsorted(out_ep, eid, side='right'))
        _oi = _oe_lo + int(np.searchsorted(out_slot[_oe_lo:_oe_hi], slot, side='left'))
        outcome = float(out_val[_oi]) if (_oi < _oe_hi and out_ep[_oi] == eid and out_slot[_oi] == slot) else 0.0

        for gi in range(ep_lo, ep_hi):
            tick = int(grp_tick[gi])
            lo   = int(grp_starts[gi])
            hi   = int(grp_starts[gi + 1])
            pids      = ps_pid[lo:hi];   owners    = ps_own[lo:hi]
            xs        = ps_x[lo:hi];     ys        = ps_y[lo:hi]
            ships_arr = ps_shp[lo:hi];   radii     = ps_rad[lo:hi];  prods = ps_prd[lo:hi]
            n_pl = len(pids)
            if n_pl == 0:
                continue

            owned_mask = owners == slot
            if not owned_mask.any():
                continue
            owned_slots_arr = np.where(owned_mask)[0]

            # Enemy encoding: rank by current ship count
            enemy_players = sorted(set(int(o) for o in owners if o >= 0 and o != slot))
            enemy_ships_total = [(p, float(ships_arr[owners == p].sum())) for p in enemy_players]
            enemy_ships_total.sort(key=lambda x: -x[1])
            k = len(enemy_ships_total)
            enemy_rel_map = {p: -(k - rank) / k for rank, (p, _) in enumerate(enemy_ships_total)}

            slots_arr = np.arange(n_pl)
            dx = xs - 50.0;  dy = ys - 50.0
            rot_x = dx * cos_t - dy * sin_t + 50.0
            rot_y = dx * sin_t + dy * cos_t + 50.0
            rel_owners = np.zeros(n_pl, dtype=np.float32)
            rel_owners[owners == slot] = 1.0
            for ep, rel_val in enemy_rel_map.items():
                rel_owners[owners == ep] = rel_val

            p_arr = np.zeros((60, 7), dtype=np.float32)
            p_arr[slots_arr, 0] = slots_arr.astype(np.float32)
            p_arr[slots_arr, 1] = rel_owners
            p_arr[slots_arr, 2] = rot_x
            p_arr[slots_arr, 3] = rot_y
            p_arr[slots_arr, 4] = radii
            p_arr[slots_arr, 5] = ships_arr
            p_arr[slots_arr, 6] = prods

            ships_target = np.zeros((60, 60), dtype=np.float32)

            # Acts CSR lookup: find tick within the cached (ep, slot) range
            if asl_lo < asl_hi:
                atk = asl_lo + int(np.searchsorted(act_grp_tick[asl_lo:asl_hi], tick, side='left'))
                if atk < asl_hi and act_grp_tick[atk] == tick:
                    alo = int(act_grp_starts[atk])
                    ahi = int(act_grp_starts[atk + 1])
                    src_pids_act = act_pid[alo:ahi]
                    angles_act   = act_ang[alo:ahi]
                    nships_act   = act_nsh[alo:ahi]
                    pid_to_idx = {int(p): i for i, p in enumerate(pids)}
                    src_idxs = np.fromiter(
                        (pid_to_idx.get(int(p), -1) for p in src_pids_act),
                        dtype=np.int32, count=len(src_pids_act)
                    )
                    valid = (src_idxs >= 0) & np.isin(src_idxs, owned_slots_arr)
                    if valid.any():
                        src_v    = src_idxs[valid]
                        angles_v = angles_act[valid]
                        nships_v = nships_act[valid]
                        tgt_slots = _classify_batch(
                            xs[src_v], ys[src_v], angles_v,
                            xs[:n_pl], ys[:n_pl], radii[:n_pl], src_v
                        )
                        hit_mask = tgt_slots >= 0
                        if hit_mask.any():
                            np.add.at(ships_target,
                                      (src_v[hit_mask], tgt_slots[hit_mask]),
                                      nships_v[hit_mask])

            for i in owned_slots_arr:
                launched = float(ships_target[i, :].sum())
                held = max(float(ships_arr[i]) - launched, 0.0)
                ships_target[i, i] = held

            if ships_target[owned_slots_arr, :].sum() == 0:
                continue

            owner_mask_out = np.zeros(60, dtype=bool)
            owner_mask_out[owned_slots_arr] = True

            ret = (gamma ** (max_tick - tick)) * outcome

            fleet_arr      = np.zeros((mfo, 6), dtype=np.float32)
            fleet_mask_arr = np.zeros(mfo, dtype=bool)
            _fi = int(np.searchsorted(fleet_ep_ids, eid, side='left'))
            if _fi < len(fleet_ep_ids) and fleet_ep_ids[_fi] == eid:
                _flo = int(fleet_offsets[_fi]); _fhi = int(fleet_offsets[_fi + 1])
                fa   = fleet_flat[_flo:_fhi]   # view, no copy
                in_t = fa[(fa[:, 0] <= tick) & (fa[:, 1] > tick)]
                if len(in_t) > mfo:
                    in_t = in_t[:mfo]
                K = len(in_t)
                if K > 0:
                    dt    = tick - in_t[:, 0]
                    speed = _fleet_speed_batch(in_t[:, 6])
                    r_src = in_t[:, 7]
                    cx    = in_t[:, 3] + np.cos(in_t[:, 5]) * (r_src + speed * dt)
                    cy    = in_t[:, 4] + np.sin(in_t[:, 5]) * (r_src + speed * dt)
                    fdx   = cx - 50.0;  fdy = cy - 50.0
                    fleet_owner_slots = in_t[:, 2].astype(np.int32)
                    fleet_rel = np.zeros(K, dtype=np.float32)
                    fleet_rel[fleet_owner_slots == slot] = 1.0
                    for ep, rel_val in enemy_rel_map.items():
                        fleet_rel[fleet_owner_slots == ep] = rel_val
                    fleet_arr[:K, 0] = np.arange(K, dtype=np.float32)
                    fleet_arr[:K, 1] = fleet_rel
                    fleet_arr[:K, 2] = in_t[:, 5] + theta
                    fleet_arr[:K, 3] = fdx * cos_t - fdy * sin_t + 50.0
                    fleet_arr[:K, 4] = fdx * sin_t + fdy * cos_t + 50.0
                    fleet_arr[:K, 5] = in_t[:, 6]
                    fleet_mask_arr[:K] = True

            planet_obs_list.append(p_arr.astype(np.float16))
            ships_target_list.append(ships_target.astype(np.float16))
            owner_mask_list.append(owner_mask_out)
            returns_list.append(np.float32(ret))
            fleet_obs_list.append(fleet_arr.astype(np.float16))
            fleet_mask_list.append(fleet_mask_arr)
            n_players_list.append(np.int8(n_p))
            _maybe_flush()

    # Final partial flush
    if planet_obs_list:
        lists = [planet_obs_list, ships_target_list, owner_mask_list,
                 returns_list, fleet_obs_list, fleet_mask_list, n_players_list]
        out_paths.append(_flush_chunk(lists, out_dir, worker_id, flush_idx))
        total_count += len(planet_obs_list)

    return out_paths, total_count


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

    # Build CSR planet-state arrays (replaces ps_dict + all_ticks_by_ep).
    # Numpy flat arrays bypass Python refcounting in forked workers → no COW per lookup.
    print("Indexing planet state (CSR arrays)...")
    t1 = time.time()
    ps_full_sorted = ps_full.sort(['episode_id', 'tick', 'planet_id'])
    ps_ep   = ps_full_sorted['episode_id'].to_numpy().astype(np.int64)
    ps_tick = ps_full_sorted['tick'].to_numpy().astype(np.int32)
    ps_pid  = ps_full_sorted['planet_id'].to_numpy().astype(np.int32)
    ps_own  = ps_full_sorted['owner'].to_numpy().astype(np.int32)
    ps_x    = ps_full_sorted['x'].to_numpy().astype(np.float32)
    ps_y    = ps_full_sorted['y'].to_numpy().astype(np.float32)
    ps_shp  = ps_full_sorted['ships'].to_numpy().astype(np.float32)
    ps_rad  = ps_full_sorted['radius'].to_numpy().astype(np.float32)
    ps_prd  = ps_full_sorted['production'].to_numpy().astype(np.float32)
    del ps_full_sorted, ps_full

    is_new     = np.concatenate([[True],
        (ps_ep[1:] != ps_ep[:-1]) | (ps_tick[1:] != ps_tick[:-1]), [True]])
    grp_starts = np.where(is_new)[0].astype(np.int64)  # (N_groups+1,)
    grp_ep     = ps_ep[grp_starts[:-1]]                # (N_groups,) non-decreasing
    grp_tick   = ps_tick[grp_starts[:-1]]              # sorted within each episode
    del is_new

    # max_tick per episode as a numpy lookup (avoids Python dict)
    unique_eps    = np.unique(grp_ep)
    max_tick_vals = np.array(
        [grp_tick[grp_ep == e].max() for e in unique_eps], dtype=np.int32)

    # Outcome lookup stays as a small Python dict (~16K entries, minimal COW cost)
    outcome_lookup = {
        (row[0], row[1]): float(row[2])
        for row in top_pe.select(['episode_id', 'slot', 'is_winner']).iter_rows()
    }
    print(f"  {len(grp_ep):,} (episode, tick) groups in {time.time()-t1:.1f}s")

    # Build CSR action arrays (replaces acts_dict).
    print("Indexing actions (CSR arrays)...")
    t2 = time.time()
    acts_sorted = (actions_df
        .with_columns([
            pl.col('src_planet_id').cast(pl.Int32),
            pl.col('n_ships').cast(pl.Float32),
            pl.col('angle').cast(pl.Float32),
            pl.col('slot').cast(pl.Int32),
        ])
        .sort(['episode_id', 'slot', 'tick']))
    act_ep   = acts_sorted['episode_id'].to_numpy().astype(np.int64)
    act_slot = acts_sorted['slot'].to_numpy().astype(np.int32)
    act_tick = acts_sorted['tick'].to_numpy().astype(np.int32)
    act_pid  = acts_sorted['src_planet_id'].to_numpy().astype(np.int32)
    act_ang  = acts_sorted['angle'].to_numpy().astype(np.float32)
    act_nsh  = acts_sorted['n_ships'].to_numpy().astype(np.float32)
    del acts_sorted

    is_new_act = np.concatenate([[True],
        (act_ep[1:] != act_ep[:-1]) |
        (act_slot[1:] != act_slot[:-1]) |
        (act_tick[1:] != act_tick[:-1]), [True]])
    act_grp_starts = np.where(is_new_act)[0].astype(np.int64)
    act_grp_ep     = act_ep[act_grp_starts[:-1]]
    act_grp_slot   = act_slot[act_grp_starts[:-1]]
    act_grp_tick   = act_tick[act_grp_starts[:-1]]
    del is_new_act
    print(f"  {len(act_grp_ep):,} (episode, slot, tick) action groups in {time.time()-t2:.1f}s")

    # Helper: CSR lookup for planet state (used in fleet event building below)
    def _csr_ps(eid, tick):
        ep_lo = int(np.searchsorted(grp_ep, eid, side='left'))
        ep_hi = int(np.searchsorted(grp_ep, eid, side='right'))
        if ep_lo == ep_hi:
            return None
        rel = int(np.searchsorted(grp_tick[ep_lo:ep_hi], tick, side='left'))
        gi  = ep_lo + rel
        if gi >= ep_hi or grp_tick[gi] != tick:
            return None
        lo, hi = int(grp_starts[gi]), int(grp_starts[gi + 1])
        return (ps_pid[lo:hi], ps_own[lo:hi], ps_x[lo:hi], ps_y[lo:hi],
                ps_shp[lo:hi], ps_rad[lo:hi], ps_prd[lo:hi])

    # Build per-episode fleet transit event table
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
        ps_row = _csr_ps(eid, tick)
        if ps_row is None:
            continue
        pids, owners, xs, ys, _, radii, _ = ps_row
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
        dist  = np.sqrt((xs[sv]-xs[tv])**2 + (ys[sv]-ys[tv])**2) - radii[sv] - radii[tv]
        speed = _fleet_speed_batch(nv)
        arr_dt = (np.ceil(np.maximum(dist, 0.0) / speed) + 1).astype(np.int32)
        if eid not in _ep_events:
            _ep_events[eid] = []
        for i in range(len(sv)):
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

    # Convert fleet_arrays_by_ep (Python dict) → CSR numpy arrays for mmap sharing
    _fleet_ep_ids = np.array(sorted(fleet_arrays_by_ep.keys()), dtype=np.int64)
    _fleet_parts  = [fleet_arrays_by_ep[e] for e in _fleet_ep_ids]
    _fleet_offs   = np.concatenate([[0], np.cumsum([len(f) for f in _fleet_parts])]).astype(np.int64)
    _fleet_flat   = np.concatenate(_fleet_parts, axis=0) if _fleet_parts else np.empty((0, 8), dtype=np.float32)

    # Convert outcome_lookup (Python dict) → sorted numpy arrays for mmap sharing
    _out_items = sorted(outcome_lookup.items())
    _out_ep    = np.array([k[0] for k, _ in _out_items], dtype=np.int64)
    _out_slot  = np.array([k[1] for k, _ in _out_items], dtype=np.int32)
    _out_val   = np.array([v   for _, v  in _out_items], dtype=np.float32)

    # Split work across workers
    work_items = [(int(eid), int(slot), int(n_p))
                  for eid, slot, n_p in top_pe.select(['episode_id', 'slot', 'n_players']).iter_rows()]

    n_workers  = min(args.workers, len(work_items))
    chunk_size = max(1, len(work_items) // n_workers)
    chunks     = [work_items[i:i + chunk_size] for i in range(0, len(work_items), chunk_size)]
    chunk_dir  = os.path.join(os.path.dirname(args.bc_data), '_preprocess_chunks')
    shared_dir = os.path.join(chunk_dir, 'shared')
    os.makedirs(shared_dir, exist_ok=True)

    # Save all shared arrays to .npy files; spawned workers mmap_mode='r' → OS shares pages
    print(f"Saving shared arrays to {shared_dir}/ ...")
    t_save = time.time()
    for name, arr in [
        ('grp_ep', grp_ep), ('grp_tick', grp_tick), ('grp_starts', grp_starts),
        ('ps_pid', ps_pid), ('ps_own', ps_own), ('ps_x', ps_x), ('ps_y', ps_y),
        ('ps_shp', ps_shp), ('ps_rad', ps_rad), ('ps_prd', ps_prd),
        ('unique_eps', unique_eps), ('max_tick_vals', max_tick_vals),
        ('act_grp_ep', act_grp_ep), ('act_grp_slot', act_grp_slot),
        ('act_grp_tick', act_grp_tick), ('act_grp_starts', act_grp_starts),
        ('act_pid', act_pid), ('act_ang', act_ang), ('act_nsh', act_nsh),
        ('fleet_ep_ids', _fleet_ep_ids), ('fleet_offsets', _fleet_offs),
        ('fleet_flat', _fleet_flat),
        ('out_ep', _out_ep), ('out_slot', _out_slot), ('out_val', _out_val),
    ]:
        np.save(os.path.join(shared_dir, f'{name}.npy'), arr)
    print(f"  saved in {time.time()-t_save:.1f}s")

    task_args = [(chunk, i, chunk_dir, shared_dir, args.gamma, MAX_FLEET_OBS, args.flush_every)
                 for i, chunk in enumerate(chunks)]

    print(f"Processing ticks with {n_workers} workers ({len(work_items)} ep-slots, "
          f"{len(chunks)} chunks)...")
    t3 = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        chunk_results = pool.map(_preprocess_chunk, task_args)

    processed_ticks = sum(n for _, n in chunk_results)
    print(f"\nProcessed {processed_ticks:,} ticks in {time.time()-t3:.1f}s "
          f"across {n_workers} workers")

    # Concatenate all sub-chunk files (each worker may have produced several)
    print("Concatenating chunks...")
    t4 = time.time()
    arrays = {k: [] for k in ['planet_obs','ships_target','owner_mask',
                               'returns','fleet_obs','fleet_mask','n_players']}
    for out_paths, n in chunk_results:
        if n == 0:
            continue
        for out_path in out_paths:
            if not os.path.exists(out_path):
                continue
            d = np.load(out_path)
            for k in arrays:
                arrays[k].append(d[k])
            os.remove(out_path)
    import shutil as _shutil
    try:
        _shutil.rmtree(shared_dir, ignore_errors=True)
        os.rmdir(chunk_dir)
    except OSError:
        pass
    print(f"  Concatenated in {time.time()-t4:.1f}s")

    planet_obs_arr   = np.concatenate(arrays['planet_obs'],   axis=0)
    ships_target_arr = np.concatenate(arrays['ships_target'], axis=0)
    owner_mask_arr   = np.concatenate(arrays['owner_mask'],   axis=0)
    returns_arr      = np.concatenate(arrays['returns'],      axis=0).astype(np.float32)
    fleet_obs_arr    = np.concatenate(arrays['fleet_obs'],    axis=0)
    fleet_mask_arr   = np.concatenate(arrays['fleet_mask'],   axis=0)
    n_players_arr    = np.concatenate(arrays['n_players'],    axis=0).astype(np.int8)

    active_fleets = fleet_mask_arr.sum(axis=1)
    print(f"  Fleet obs: mean={active_fleets.mean():.1f}  max={active_fleets.max()}  "
          f"  capped (={MAX_FLEET_OBS}): {(active_fleets==MAX_FLEET_OBS).sum()} ticks")
    duels = (n_players_arr == 2).sum()
    print(f"  n_players: {duels:,} duel ticks / {processed_ticks - duels:,} FFA ticks")

    # Diagonal coverage (fraction of owned planets with non-zero hold signal)
    diag = np.array([ships_target_arr[:, i, i] for i in range(60)]).T
    own  = owner_mask_arr
    held_frac = (diag[own] > 0).mean()
    print(f"  Hold signal (diagonal): {held_frac*100:.1f}% of owned planet slots are non-zero")

    print(f"Saving to {args.bc_data}...")
    np.savez_compressed(
        args.bc_data,
        planet_obs=planet_obs_arr,
        ships_target=ships_target_arr,
        owner_mask=owner_mask_arr,
        returns=returns_arr,
        fleet_obs=fleet_obs_arr,
        fleet_mask=fleet_mask_arr,
        n_players=n_players_arr,
    )
    size_mb = os.path.getsize(args.bc_data) / 1e6
    print(f"Saved {args.bc_data} ({size_mb:.1f} MB)")

    if args.upload_to_kaggle:
        _upload_to_kaggle(args.bc_data, args.upload_to_kaggle)

    return planet_obs_arr, ships_target_arr, owner_mask_arr, returns_arr, fleet_obs_arr, fleet_mask_arr, n_players_arr


def _upload_to_kaggle(npz_path: str, dataset_slug: str):
    """Upload npz_path as a new version of a Kaggle dataset (owner/name)."""
    import shutil, tempfile
    print(f"Uploading {npz_path} to Kaggle dataset {dataset_slug}...")
    owner, name = dataset_slug.split('/', 1)
    with tempfile.TemporaryDirectory() as tmp:
        fname = os.path.basename(npz_path)
        shutil.copy(npz_path, os.path.join(tmp, fname))
        meta = {
            "title": name,
            "id": f"{owner}/{name}",
            "licenses": [{"name": "other"}]
        }
        with open(os.path.join(tmp, 'dataset-metadata.json'), 'w') as f:
            json.dump(meta, f)
        result = subprocess.run(
            ['kaggle', 'datasets', 'version', '-p', tmp,
             '-m', f'bc_data update {time.strftime("%Y-%m-%d %H:%M")}', '--dir-mode', 'tar'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Kaggle upload succeeded: {result.stdout.strip()}")
        else:
            print(f"  Kaggle upload failed: {result.stderr.strip()}")
            print("  (Ensure KAGGLE_USERNAME / KAGGLE_KEY are set and the dataset already exists)")


# ---------------------------------------------------------------------------
# TRAIN ACTOR (BC)
# ---------------------------------------------------------------------------
def train(planet_obs=None, ships_target=None, owner_mask=None, returns=None,
          fleet_obs=None, fleet_mask=None, n_players=None):
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
        n_players    = data['n_players'] if 'n_players' in data else None
        if 'fleet_obs' in data:
            fleet_obs  = data['fleet_obs'].astype(np.float32)
            fleet_mask = data['fleet_mask']

    N = len(planet_obs)
    if fleet_obs is None:
        fleet_obs  = np.zeros((N, MAX_FLEET_OBS, 6), dtype=np.float32)
        fleet_mask = np.zeros((N, MAX_FLEET_OBS),    dtype=bool)

    rng = np.random.default_rng(42)
    perm   = rng.permutation(N)
    n_val  = max(1, int(N * args.val_frac))
    val_idx     = perm[:n_val]
    val_idx_nat = val_idx.copy()   # natural distribution val — no subsampling, for diagnostic only
    train_idx   = perm[n_val:]
    N_train     = len(train_idx)

    # Hold subsampling: applied identically to train and val to keep same distribution
    if args.hold_rate < 1.0:
        diag_mask_2d   = np.eye(60, dtype=bool)
        ships_off_diag = ships_target * (~diag_mask_2d)[None, :, :]
        off_per_sample = (ships_off_diag * owner_mask[:, :, None]).sum(axis=(1, 2))
        is_action      = off_per_sample > 0.5
        del ships_off_diag, off_per_sample

        action_tr   = train_idx[is_action[train_idx]]
        hold_tr     = train_idx[~is_action[train_idx]]
        n_hold_keep = max(1, int(len(hold_tr) * args.hold_rate))
        train_idx   = np.concatenate([action_tr, rng.choice(hold_tr, size=n_hold_keep, replace=False)])
        print(f"  Hold subsampling (train): {len(action_tr):,} action + {len(hold_tr):,} hold "
              f"→ kept {n_hold_keep:,} hold → {len(train_idx):,} total "
              f"({len(action_tr)/len(train_idx)*100:.1f}% action)")
        N_train = len(train_idx)

        action_v     = val_idx[is_action[val_idx]]
        hold_v       = val_idx[~is_action[val_idx]]
        n_hold_keep_v = max(1, int(len(hold_v) * args.hold_rate))
        val_idx      = np.concatenate([action_v, rng.choice(hold_v, size=n_hold_keep_v, replace=False)])
        print(f"  Hold subsampling (val):   {len(action_v):,} action + {len(hold_v):,} hold "
              f"→ kept {n_hold_keep_v:,} hold → {len(val_idx):,} total "
              f"({len(action_v)/len(val_idx)*100:.1f}% action)")

    # FFA oversampling: applied identically to train and val
    if n_players is not None and args.ffa_weight > 1:
        is_ffa_tr   = n_players[train_idx] == 4
        train_idx_w = np.concatenate([train_idx, np.tile(train_idx[is_ffa_tr], args.ffa_weight - 1)])
        n_duel = (~is_ffa_tr).sum(); n_ffa = is_ffa_tr.sum()
        print(f"  FFA weighting (train): {n_duel:,} duel / {n_ffa:,} ffa → ffa ×{args.ffa_weight} "
              f"→ train size {len(train_idx_w):,}")

        is_ffa_v  = n_players[val_idx] == 4
        val_idx_w = np.concatenate([val_idx, np.tile(val_idx[is_ffa_v], args.ffa_weight - 1)])
        print(f"  FFA weighting (val):   val size {len(val_idx_w):,}")
    else:
        train_idx_w = train_idx
        val_idx_w   = val_idx
    N_train_w = len(train_idx_w)
    n_val_w   = len(val_idx_w)

    print(f"Training actor on {N_train:,} train / {len(val_idx):,} val ticks "
          f"(effective {N_train_w:,} train / {n_val_w:,} val with hold+FFA weighting) | "
          f"batch={args.batch_size} | epochs={args.epochs} | wd={args.weight_decay}")

    actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
    actor_graph, params = nnx.split(actor)

    # Transfer learning: load v13 weights, reinit action head + fleet MLPs
    if args.transfer_from:
        print(f"Transfer learning from {args.transfer_from}...")
        import pickle as _pkl
        with open(args.transfer_from, 'rb') as _f:
            _src = _pkl.load(_f)
        _src_jax = jax.tree_util.tree_map(jnp.array, _src)

        # Flat-key transfer via jax pytree paths
        src_leaves, src_tdef = jax.tree_util.tree_flatten_with_path(_src_jax)
        dst_leaves, dst_tdef = jax.tree_util.tree_flatten_with_path(params)
        REINIT_SUBSTRINGS = [
            'q_action', 'k_action', 'temperature_head',
            'deep_space_prob', 'deep_space_sincos',
            'ca_block_0.fleet_mlp', 'ca_block_0.planet_mlp', 'ca_block_0.rel_bias_mlp',
            'ca_block_1.fleet_mlp', 'ca_block_1.rel_bias_mlp',
        ]
        new_leaves = []
        transferred, reinitialized = 0, 0
        for (dst_path, dst_leaf), (_, src_leaf) in zip(dst_leaves, src_leaves):
            path_str = '.'.join(
                str(k.key if hasattr(k, 'key') else k) for k in dst_path
            )
            if any(sub in path_str for sub in REINIT_SUBSTRINGS):
                new_leaves.append(dst_leaf)  # keep fresh init
                reinitialized += 1
            else:
                new_leaves.append(src_leaf)  # transfer from v13
                transferred += 1
        params = jax.tree_util.tree_unflatten(dst_tdef, new_leaves)
        print(f"  Transferred {transferred} / reinitialized {reinitialized} parameter leaves")

    steps_per_epoch = N_train_w // args.batch_size
    warmup_steps    = args.warmup_epochs * steps_per_epoch

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, args.lr, max(warmup_steps, 1)),
            optax.constant_schedule(args.lr),
        ],
        boundaries=[warmup_steps],
    )
    opt       = optax.contrib.muon(learning_rate=schedule, weight_decay=args.weight_decay)
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

    def _rotate_batch(planets_b, fleets_b, k):
        """Apply k*90° rotation (0=identity, 1=90°, 2=180°, 3=270°). Pure JAX — no Python branches."""
        angle = k * jnp.pi / 2.0
        c, s = jnp.cos(angle), jnp.sin(angle)

        # Planet: cols 2,3 = rot_x, rot_y
        px = planets_b[..., 2] - 50.0
        py = planets_b[..., 3] - 50.0
        planets_r = jnp.concatenate([
            planets_b[..., :2],
            (px * c - py * s + 50.0)[..., None],
            (px * s + py * c + 50.0)[..., None],
            planets_b[..., 4:],
        ], axis=-1)

        # Fleet: col 2 = angle, cols 3,4 = rot_x, rot_y
        fx = fleets_b[..., 3] - 50.0
        fy = fleets_b[..., 4] - 50.0
        fleets_r = jnp.concatenate([
            fleets_b[..., :2],
            (fleets_b[..., 2] + angle)[..., None],
            (fx * c - fy * s + 50.0)[..., None],
            (fx * s + fy * c + 50.0)[..., None],
            fleets_b[..., 5:],
        ], axis=-1)

        return planets_r, fleets_r

    def _xent(planets_b, ships_tgt_b, owner_mask_b, fleets_b, fmask_b, actor_inst):
        pmask = planets_b[..., 4] > 0   # radius > 0 — excludes zero-padded ghost slots
        logits = actor_inst(planets_b, fleets_b, planet_mask=pmask, fleet_mask=fmask_b)
        # Diagonal now included as "hold" signal — normalize by total ships per planet
        total_ships = ships_tgt_b.sum(axis=-1, keepdims=True)
        target_dist = ships_tgt_b / (total_ships + 1e-8)
        log_probs   = jax.nn.log_softmax(logits[..., :60], axis=-1)
        xent        = -jnp.sum(target_dist * log_probs, axis=-1)
        # has_ships: owned planets with any ships (diagonal or off-diagonal)
        has_ships   = (total_ships[..., 0] > 0) & owner_mask_b
        return jnp.sum(xent * has_ships) / (jnp.sum(has_ships) + 1e-8)

    @jax.jit
    def train_step(params, opt_state, planets_b, ships_tgt_b, owner_mask_b,
                   fleets_b, fmask_b, rot_k):
        # On-the-fly rotation augmentation (rot_k ∈ {0,1,2,3})
        planets_aug, fleets_aug = _rotate_batch(planets_b, fleets_b, rot_k)

        def loss_fn(params):
            return _xent(planets_aug, ships_tgt_b, owner_mask_b, fleets_aug, fmask_b,
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
        shuffled = rng.permutation(N_train_w)

        train_losses = []
        batch_iter = _bar(range(0, N_train_w - args.batch_size, args.batch_size),
                          desc=f'E{epoch+1}/{args.epochs}')
        for start in batch_iter:
            bi = train_idx_w[shuffled[start:start + args.batch_size]]
            rot_k = int(rng.integers(0, 4))
            params, opt_state, loss = train_step(
                params, opt_state,
                jnp.array(planet_obs[bi]),
                jnp.array(ships_target[bi]),
                jnp.array(owner_mask[bi]),
                jnp.array(fleet_obs[bi]),
                jnp.array(fleet_mask[bi]),
                jnp.array(rot_k),
            )
            loss_val = float(loss)
            train_losses.append(loss_val)
            if _tqdm is not None:
                batch_iter.set_postfix(loss=f'{loss_val:.4f}')

        train_loss = float(np.mean(train_losses))
        do_eval    = (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1

        if do_eval:
            val_losses = []
            val_perm = rng.permutation(n_val_w)
            for start in range(0, n_val_w - args.batch_size, args.batch_size):
                bi = val_idx_w[val_perm[start:start + args.batch_size]]
                val_losses.append(float(eval_step(
                    params,
                    jnp.array(planet_obs[bi]),
                    jnp.array(ships_target[bi]),
                    jnp.array(owner_mask[bi]),
                    jnp.array(fleet_obs[bi]),
                    jnp.array(fleet_mask[bi]),
                )))
            val_loss = float(np.mean(val_losses))

            # Natural distribution diagnostic (no hold subsampling / FFA weighting) — every 25 epochs
            nat_val_str = ''
            if (epoch + 1) % 25 == 0:
                nat_losses = []
                nat_perm = rng.permutation(len(val_idx_nat))
                for start in range(0, len(val_idx_nat) - args.batch_size, args.batch_size):
                    bi = val_idx_nat[nat_perm[start:start + args.batch_size]]
                    nat_losses.append(float(eval_step(
                        params,
                        jnp.array(planet_obs[bi]),
                        jnp.array(ships_target[bi]),
                        jnp.array(owner_mask[bi]),
                        jnp.array(fleet_obs[bi]),
                        jnp.array(fleet_mask[bi]),
                    )))
                nat_val_str = f'  nat_val={float(np.mean(nat_losses)):.4f}'

            print(f"Epoch {epoch+1:>3d}/{args.epochs} | "
                  f"train={train_loss:.4f}  val={val_loss:.4f}{nat_val_str} | {time.time()-t0:.1f}s  {_rezero_summary(params)}")
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
        _fo  = _d['fleet_obs'].astype(np.float32) if 'fleet_obs'  in _d else None
        _fm  = _d['fleet_mask']                    if 'fleet_mask' in _d else None
        _np  = _d['n_players']                     if 'n_players'  in _d else None
        preprocess_result = (
            _d['planet_obs'].astype(np.float32),
            _d['ships_target'].astype(np.float32),
            _d['owner_mask'],
            _d['returns'].astype(np.float32),
            _fo,
            _fm,
            _np,
        )
        print(f"  Loaded {len(preprocess_result[0]):,} samples")

    if args.train:
        train(*preprocess_result)

    if args.pretrain_critic:
        pretrain_critic(preprocess_result[0], preprocess_result[3],
                        preprocess_result[4], preprocess_result[5])
