"""Generate an HTML replay of the v13 ep480 BC agent vs random.

Usage (from project root):
    uv run python scripts/eval_v13.py [--out /tmp/bc_v13_vs_random.html]
"""
import argparse
import math
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from kaggle_environments import make

from core.networks import Actor, logits_to_action

HIDDEN_DIM    = 48
NUM_SA_LAYERS = 6
MAX_SPEED     = 6.0
ROTATION_RADIUS_LIMIT = 50.0
SUN_RADIUS    = 10.0
_CENTER       = 50.0
_BOARD_MIN    = 0.0
_BOARD_MAX    = 100.0


def _fleet_speed(num_ships):
    safe = max(int(num_ships), 1)
    return min(1.0 + (MAX_SPEED - 1.0) * (math.log(safe) / math.log(1000)) ** 1.5, MAX_SPEED)


def _ray_hits_sun(sx, sy, angle):
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = sx - _CENTER, sy - _CENTER
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_RADIUS * SUN_RADIUS
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return False
    return (-b + math.sqrt(disc)) / 2.0 >= 0.0


def _ray_exits_board(sx, sy, angle):
    if not (_BOARD_MIN <= sx <= _BOARD_MAX and _BOARD_MIN <= sy <= _BOARD_MAX):
        return False
    ex = sx + math.cos(angle) * 141.0
    ey = sy + math.sin(angle) * 141.0
    return not (_BOARD_MIN <= ex <= _BOARD_MAX and _BOARD_MIN <= ey <= _BOARD_MAX)


def _comet_intercept_angle(sx, sy, sr, tr, path, path_index, n_ships):
    speed = _fleet_speed(n_ships)
    for t in range(1, 151):
        future_idx = path_index + t
        if future_idx >= len(path):
            return None
        fx, fy = path[future_idx]
        reach_dist = math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
        if speed * t >= reach_dist - sr - tr:
            return math.atan2(fy - sy, fx - sx)
    return None


def _intercept_angle(sx, sy, sr, tx, ty, tr, ang_vel, n_ships):
    t_dist = math.sqrt((tx - 50.0) ** 2 + (ty - 50.0) ** 2)
    if ang_vel == 0.0 or t_dist + tr >= ROTATION_RADIUS_LIMIT:
        return math.atan2(ty - sy, tx - sx)
    t_angle = math.atan2(ty - 50.0, tx - 50.0)
    speed   = _fleet_speed(n_ships)
    for t in range(1, 151):
        fa = t_angle + ang_vel * t
        fx = 50.0 + t_dist * math.cos(fa)
        fy = 50.0 + t_dist * math.sin(fa)
        reach_dist = math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
        if speed * t >= reach_dist - sr - tr:
            return math.atan2(fy - sy, fx - sx)
    return math.atan2(ty - sy, tx - sx)


def load_actor(weights_path: str):
    actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
    actor_graph, _ = nnx.split(actor)

    with open(weights_path, 'rb') as f:
        np_params = pickle.load(f)
    jax_params = jax.tree_util.tree_map(jnp.array, np_params)

    @jax.jit
    def forward(params, planets, fleets, planet_mask, fleet_mask):
        a = nnx.merge(actor_graph, params)
        return a(planets, fleets, planet_mask=planet_mask, fleet_mask=fleet_mask)

    return jax_params, forward


def make_bc_agent(jax_params, forward_fn):
    num_players_state = [None]

    def bc_agent(obs):
        if isinstance(obs, dict):
            player      = obs['player']
            raw_planets = obs['planets']
            raw_fleets  = obs['fleets']
            ang_vel     = float(obs.get('angular_velocity', 0.0))
            raw_comets  = obs.get('comets', [])
        else:
            player      = obs.player
            raw_planets = obs.planets
            raw_fleets  = obs.fleets
            ang_vel     = float(obs.angular_velocity)
            raw_comets  = getattr(obs, 'comets', [])

        comet_path_info = {}
        for group in raw_comets:
            idx = group['path_index']
            for i, pid in enumerate(group['planet_ids']):
                comet_path_info[pid] = (group['paths'][i], idx)

        if num_players_state[0] is None:
            owners = set(int(p[1]) for p in raw_planets if int(p[1]) >= 0)
            num_players_state[0] = max(len(owners), 2)

        theta = -player * (2.0 * math.pi / num_players_state[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        p_arr  = np.zeros((60, 7), dtype=np.float32)
        p_mask = np.zeros(60, dtype=bool)
        for slot, p in enumerate(raw_planets[:60]):
            pid, owner, x, y, radius, ships, prod = (
                int(p[0]), int(p[1]), float(p[2]), float(p[3]),
                float(p[4]), float(p[5]), float(p[6])
            )
            if x < 0 or y < 0:
                continue
            dx, dy = x - 50.0, y - 50.0
            rot_x = dx * cos_t - dy * sin_t + 50.0
            rot_y = dx * sin_t + dy * cos_t + 50.0
            rel_owner = 1.0 if owner == player else (0.0 if owner < 0 else -1.0)
            p_arr[slot]  = [slot, rel_owner, rot_x, rot_y, radius, ships, prod]
            p_mask[slot] = True

        n_fleets = len(raw_fleets)
        f_arr  = np.zeros((max(n_fleets, 1), 6), dtype=np.float32)
        f_mask = np.zeros(max(n_fleets, 1), dtype=bool)
        for slot, fl in enumerate(raw_fleets):
            fid, owner, x, y, angle, from_pid, ships = (
                int(fl[0]), int(fl[1]), float(fl[2]), float(fl[3]),
                float(fl[4]), int(fl[5]), float(fl[6])
            )
            dx, dy = x - 50.0, y - 50.0
            rot_x = dx * cos_t - dy * sin_t + 50.0
            rot_y = dx * sin_t + dy * cos_t + 50.0
            rel_owner = 1.0 if owner == player else (0.0 if owner < 0 else -1.0)
            f_arr[slot]  = [slot, rel_owner, angle + theta, rot_x, rot_y, ships]
            f_mask[slot] = True

        planets_jax = jnp.array(p_arr[None])
        fleets_jax  = jnp.array(f_arr[None])
        pmask_jax   = jnp.array(p_mask[None])
        fmask_jax   = jnp.array(f_mask[None])

        logits = forward_fn(jax_params, planets_jax, fleets_jax, pmask_jax, fmask_jax)

        p_ships_jax = planets_jax[..., 5]
        ships, ds_angle = logits_to_action(logits, p_ships_jax)

        ships    = np.array(ships[0])     # [60, 64]
        ds_angle = np.array(ds_angle[0])  # [60, 4]

        moves = []
        for src_slot, p in enumerate(raw_planets[:60]):
            pid, owner, sx, sy = int(p[0]), int(p[1]), float(p[2]), float(p[3])
            if owner != player:
                continue
            sr = float(p[4])

            for tgt_slot, tp in enumerate(raw_planets[:60]):
                if tgt_slot == src_slot:
                    continue
                n_ships = int(ships[src_slot, tgt_slot])
                if n_ships < 1:
                    continue
                tx, ty, tr = float(tp[2]), float(tp[3]), float(tp[4])
                if tx < 0 or ty < 0:
                    continue
                tpid = int(tp[0])
                if tpid in comet_path_info:
                    path, path_index = comet_path_info[tpid]
                    angle = _comet_intercept_angle(sx, sy, sr, tr, path, path_index, n_ships)
                    if angle is None:
                        continue
                else:
                    angle = _intercept_angle(sx, sy, sr, tx, ty, tr, ang_vel, n_ships)
                if _ray_hits_sun(sx, sy, angle):
                    continue
                moves.append([pid, angle, n_ships])

            for b in range(4):
                n_ships = int(ships[src_slot, 60 + b])
                if n_ships < 1:
                    continue
                world_angle = float(ds_angle[src_slot, b]) - theta
                if _ray_hits_sun(sx, sy, world_angle) or _ray_exits_board(sx, sy, world_angle):
                    continue
                moves.append([pid, world_angle, n_ships])

        return moves

    return bc_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(_HERE), 'models', 'weights_bc_v13_e480.pkl'))
    parser.add_argument('--out', default='/tmp/bc_v13_vs_random.html')
    args = parser.parse_args()

    print(f'Loading weights from {args.weights}...')
    jax_params, forward_fn = load_actor(args.weights)
    bc_agent = make_bc_agent(jax_params, forward_fn)
    print('Model loaded. Running game...')

    env = make('orbit_wars', debug=False)
    env.run([bc_agent, 'random'])
    reward = env.steps[-1][0].reward
    result = 'WIN' if reward == 1 else ('LOSS' if reward == -1 else 'DRAW')
    print(f'Result: {result} in {len(env.steps)} steps')

    with open(args.out, 'w') as f:
        f.write(env.render(mode='html', width=800, height=600))
    print(f'Replay saved → {args.out}')


if __name__ == '__main__':
    main()
