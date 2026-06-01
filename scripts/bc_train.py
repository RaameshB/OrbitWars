"""
Behavioral Cloning from leaderboard replay data.

Usage:
  uv run python scripts/bc_train.py --preprocess --data-dir /tmp/orbit-wars-parquet
  uv run python scripts/bc_train.py --train --bc-data /tmp/bc_data.npz --out weights_bc.pkl
  uv run python scripts/bc_train.py --preprocess --train  # run both back to back
"""

import os, sys, math, pickle, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--train',      action='store_true')
parser.add_argument('--data-dir',   default='/tmp/orbit-wars-parquet')
parser.add_argument('--bc-data',    default='/tmp/bc_data.npz')
parser.add_argument('--out',        default='/tmp/weights_bc.pkl')
parser.add_argument('--top-k',      type=int, default=10,   help='Top-K players by win rate')
parser.add_argument('--min-games',  type=int, default=20)
parser.add_argument('--epochs',     type=int, default=50)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--lr',         type=float, default=3e-4)
args = parser.parse_args()

HIDDEN_DIM    = 32
NUM_SA_LAYERS = 6

# ---------------------------------------------------------------------------
# Target classification — ray-cast angle → target planet slot
# ---------------------------------------------------------------------------
def classify_target(sx, sy, angle, planet_xs, planet_ys, planet_rs, src_slot):
    """Return target slot index or -1 if deep space."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    best_slot, best_t = -1, 1e18
    for slot, (px, py, pr) in enumerate(zip(planet_xs, planet_ys, planet_rs)):
        if slot == src_slot:
            continue
        dx, dy = px - sx, py - sy
        proj = dx * cos_a + dy * sin_a
        if proj <= 0:
            continue
        perp_sq = dx*dx + dy*dy - proj*proj
        if perp_sq < pr*pr and proj < best_t:
            best_slot, best_t = slot, proj
    return best_slot


# ---------------------------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------------------------
def preprocess():
    import pandas as pd

    data_dir = args.data_dir
    print(f"Loading parquet tables from {data_dir}...")
    episodes        = pd.read_parquet(f'{data_dir}/episodes.parquet')
    player_eps      = pd.read_parquet(f'{data_dir}/player_episodes.parquet')
    actions_df      = pd.read_parquet(f'{data_dir}/actions.parquet')
    ep_planets      = pd.read_parquet(f'{data_dir}/episode_planets.parquet')
    planet_state    = pd.read_parquet(f'{data_dir}/planet_state.parquet')
    print(f"  planet_state: {len(planet_state):,} rows")

    # Top-K players by win rate
    stats = player_eps.groupby('name').agg(
        games=('is_winner', 'count'), wins=('is_winner', 'sum')
    ).reset_index()
    stats['win_rate'] = stats['wins'] / stats['games']
    top_players = (stats[stats['games'] >= args.min_games]
                   .sort_values('win_rate', ascending=False)
                   .head(args.top_k)['name'].tolist())
    print(f"Top-{args.top_k} players: {top_players}")

    # Filter player_eps to top players, join episode metadata
    top_pe = player_eps[player_eps['name'].isin(top_players)].merge(
        episodes[['episode_id', 'n_players', 'angular_velocity']], on='episode_id'
    )

    # Normalize angles to [-π, π]
    def norm_angle(a):
        return ((a + math.pi) % (2 * math.pi)) - math.pi

    # Index planet_state and ep_planets by episode for fast lookup
    print("Indexing planet_state by episode (this may take a moment)...")
    ps_by_ep  = {eid: grp for eid, grp in planet_state.groupby('episode_id')}
    epp_by_ep = {eid: grp for eid, grp in ep_planets.groupby('episode_id')}
    act_by_ep_slot = {
        (eid, slot): grp
        for (eid, slot), grp in actions_df.groupby(['episode_id', 'slot'])
    }

    all_planet_obs  = []  # [N_ticks, 60, 7]  float16
    all_ships_target= []  # [N_ticks, 60, 60] float16
    all_owner_mask  = []  # [N_ticks, 60]     bool

    skipped_eps = 0
    processed_ticks = 0

    for _, row in top_pe.iterrows():
        eid      = row['episode_id']
        slot     = row['slot']
        n_p      = int(row['n_players'])

        # Actions for this player in this episode
        key = (eid, slot)
        if key not in act_by_ep_slot:
            continue
        ep_acts = act_by_ep_slot[key]

        # Planet topology for this episode
        if eid not in epp_by_ep or eid not in ps_by_ep:
            skipped_eps += 1
            continue

        topo  = epp_by_ep[eid].set_index('planet_id')  # radius, production per planet_id
        ps_ep = ps_by_ep[eid]                           # tick-level state

        # Player rotation: rotate coords to player's perspective
        theta  = -slot * (2.0 * math.pi / n_p)
        cos_t  = math.cos(theta)
        sin_t  = math.sin(theta)

        # Group state and actions by tick
        ps_by_tick  = {t: g for t, g in ps_ep.groupby('tick')}
        act_by_tick = {t: g for t, g in ep_acts.groupby('tick')}

        ticks_with_actions = set(act_by_tick.keys())

        for tick in ticks_with_actions:
            if tick not in ps_by_tick:
                continue

            ps_tick = ps_by_tick[tick]      # planet states at this tick
            acts    = act_by_tick[tick]     # actions this tick

            # Build planet_id → slot mapping from planet_state at this tick
            # planet_state may have fewer than 60 planets
            pids_in_tick = ps_tick['planet_id'].values
            n_planets = len(pids_in_tick)
            if n_planets == 0:
                continue

            # planet_id → slot index (0-based in order they appear)
            pid_to_slot = {int(pid): i for i, pid in enumerate(pids_in_tick)}

            # Build observation arrays — planet array [60, 7]
            p_arr  = np.zeros((60, 7), dtype=np.float32)
            p_mask = np.zeros(60, dtype=bool)

            # Pre-extract arrays for classify_target
            planet_xs = np.zeros(60, dtype=np.float32)
            planet_ys = np.zeros(60, dtype=np.float32)
            planet_rs = np.zeros(60, dtype=np.float32)

            owned_slots = set()

            for _, pr in ps_tick.iterrows():
                pid   = int(pr['planet_id'])
                pslot = pid_to_slot[pid]
                owner = int(pr['owner'])
                x, y  = float(pr['x']), float(pr['y'])
                ships = float(pr['ships'])

                # Topology
                if pid not in topo.index:
                    continue
                radius = float(topo.loc[pid, 'radius'])
                prod   = float(topo.loc[pid, 'production'])

                # Rotate to player frame
                dx, dy  = x - 50.0, y - 50.0
                rot_x   = dx * cos_t - dy * sin_t + 50.0
                rot_y   = dx * sin_t + dy * cos_t + 50.0

                rel_owner = 1.0 if owner == slot else (0.0 if owner < 0 else -1.0)

                p_arr[pslot]   = [pslot, rel_owner, rot_x, rot_y, radius, ships, prod]
                p_mask[pslot]  = True
                planet_xs[pslot] = x   # world coords for ray-cast
                planet_ys[pslot] = y
                planet_rs[pslot] = radius

                if owner == slot:
                    owned_slots.add(pslot)

            if not owned_slots:
                continue

            # Build target ships matrix [60, 60]
            ships_target = np.zeros((60, 60), dtype=np.float32)

            for _, act in acts.iterrows():
                src_pid = int(act['src_planet_id'])
                if src_pid not in pid_to_slot:
                    continue
                src_slot_idx = pid_to_slot[src_pid]
                if src_slot_idx not in owned_slots:
                    continue

                angle   = norm_angle(float(act['angle']))
                n_ships = float(act['n_ships'])
                sx      = planet_xs[src_slot_idx]
                sy      = planet_ys[src_slot_idx]

                tgt_slot = classify_target(
                    sx, sy, angle,
                    planet_xs[:n_planets], planet_ys[:n_planets], planet_rs[:n_planets],
                    src_slot_idx
                )
                if tgt_slot >= 0:
                    ships_target[src_slot_idx, tgt_slot] += n_ships
                # deep space: skip for now (angle doesn't hit any planet)

            # Skip ticks where no classifiable actions
            if ships_target.sum() == 0:
                continue

            owner_mask = np.zeros(60, dtype=bool)
            for s in owned_slots:
                owner_mask[s] = True

            all_planet_obs.append(p_arr.astype(np.float16))
            all_ships_target.append(ships_target.astype(np.float16))
            all_owner_mask.append(owner_mask)
            processed_ticks += 1

    print(f"\nProcessed {processed_ticks:,} ticks ({skipped_eps} episodes skipped)")

    planet_obs_arr  = np.stack(all_planet_obs,   axis=0)  # [N, 60, 7]
    ships_target_arr= np.stack(all_ships_target, axis=0)  # [N, 60, 60]
    owner_mask_arr  = np.stack(all_owner_mask,   axis=0)  # [N, 60]

    print(f"Saving to {args.bc_data}...")
    np.savez_compressed(
        args.bc_data,
        planet_obs=planet_obs_arr,
        ships_target=ships_target_arr,
        owner_mask=owner_mask_arr,
    )
    size_mb = os.path.getsize(args.bc_data) / 1e6
    print(f"Saved {args.bc_data} ({size_mb:.1f} MB)")
    return planet_obs_arr, ships_target_arr, owner_mask_arr


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------
def train(planet_obs=None, ships_target=None, owner_mask=None):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from core.networks import Actor

    # Load data if not passed directly from preprocess
    if planet_obs is None:
        print(f"Loading {args.bc_data}...")
        data = np.load(args.bc_data)
        planet_obs   = data['planet_obs'].astype(np.float32)    # [N, 60, 7]
        ships_target = data['ships_target'].astype(np.float32)  # [N, 60, 60]
        owner_mask   = data['owner_mask']                        # [N, 60]

    N = len(planet_obs)
    print(f"Training on {N:,} ticks | batch={args.batch_size} | epochs={args.epochs}")

    # Dummy fleet input (no fleet data in parquet)
    dummy_fleets     = np.zeros((1, 1, 6), dtype=np.float32)
    dummy_fleet_mask = np.zeros((1, 1),    dtype=bool)

    # Init network
    actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
    actor_graph, params = nnx.split(actor)

    # Muon optimizer with cosine LR decay
    steps_per_epoch = N // args.batch_size
    total_steps     = steps_per_epoch * args.epochs
    schedule = optax.cosine_decay_schedule(args.lr, total_steps, alpha=0.1)
    opt      = optax.contrib.muon(learning_rate=schedule)
    opt_state = opt.init(params)

    # Tile dummy fleets to match batch size
    fleets_b     = jnp.zeros((args.batch_size, 1, 6), dtype=jnp.float32)
    fmask_b      = jnp.zeros((args.batch_size, 1),    dtype=bool)

    @jax.jit
    def train_step(params, opt_state, planets_b, ships_tgt_b, owner_mask_b):
        def loss_fn(params):
            actor = nnx.merge(actor_graph, params)
            # planet_mask: True for filled slots (ships > 0 or owned)
            pmask = planets_b[..., 5] >= 0  # all valid slots
            logits = actor(planets_b, fleets_b, planet_mask=pmask, fleet_mask=fmask_b)
            # logits: [B, 60, 72] — first 64 are action logits (60 planets + 4 ds)
            action_logits = logits[..., :60]  # [B, 60, 60] planet-to-planet only

            # Target distribution: normalize ships sent per source planet
            total_sent = ships_tgt_b.sum(axis=-1, keepdims=True)  # [B, 60, 1]
            target_dist = ships_tgt_b / (total_sent + 1e-8)       # [B, 60, 60]

            # Cross-entropy: -sum(target * log_softmax(logits)) per owned planet with actions
            log_probs = jax.nn.log_softmax(action_logits, axis=-1)  # [B, 60, 60]
            xent = -jnp.sum(target_dist * log_probs, axis=-1)       # [B, 60]

            # Mask: only owned planets that actually sent ships this tick
            has_action = (total_sent[..., 0] > 0) & owner_mask_b    # [B, 60]
            loss = jnp.sum(xent * has_action) / (jnp.sum(has_action) + 1e-8)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    rng = np.random.default_rng(42)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        idx = rng.permutation(N)
        epoch_losses = []
        t0 = time.time()

        for start in range(0, N - args.batch_size, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            planets_b  = jnp.array(planet_obs[batch_idx])    # [B, 60, 7]
            ships_tgt_b= jnp.array(ships_target[batch_idx])  # [B, 60, 60]
            omask_b    = jnp.array(owner_mask[batch_idx])     # [B, 60]

            params, opt_state, loss = train_step(params, opt_state, planets_b, ships_tgt_b, omask_b)
            epoch_losses.append(float(loss))

        mean_loss = np.mean(epoch_losses)
        elapsed   = time.time() - t0
        print(f"Epoch {epoch+1:>3d}/{args.epochs} | loss={mean_loss:.4f} | {elapsed:.1f}s")

        if mean_loss < best_loss:
            best_loss = mean_loss
            # Save weights as numpy dict (same format as submission/main.py)
            final_actor = nnx.merge(actor_graph, params)
            _, best_params = nnx.split(final_actor)
            np_params = jax.tree_util.tree_map(np.array, best_params)
            with open(args.out, 'wb') as f:
                pickle.dump(np_params, f)

    print(f"\nDone. Best loss: {best_loss:.4f} | Weights saved to {args.out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    if not args.preprocess and not args.train:
        parser.print_help()
        sys.exit(1)

    data = None
    if args.preprocess:
        result = preprocess()
        if args.train:
            data = result  # pass directly, skip disk round-trip

    if args.train:
        if data:
            train(*data)
        else:
            train()
