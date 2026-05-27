from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import kaggle_environments.envs.orbit_wars.orbit_wars as orbit_wars


@dataclass(frozen=True)
class JAXOrbitWarsConfig:
    num_envs: int = 128
    num_players: int = 2
    max_planets: int = 48
    max_fleets: int = 512
    episode_steps: int = 500
    ship_speed: float = 6.0
    use_swept_collision: bool = False
    use_segment_sun_check: bool = False


def default_config(
    num_envs: int = 128,
    num_players: int = 2,
    max_fleets: int = 512,
    use_swept_collision: bool = False,
    use_segment_sun_check: bool = False,
) -> JAXOrbitWarsConfig:
    return JAXOrbitWarsConfig(
        num_envs=num_envs,
        num_players=num_players,
        max_fleets=max_fleets,
        use_swept_collision=use_swept_collision,
        use_segment_sun_check=use_segment_sun_check,
    )


def _pad_planets(planets: list[list[Any]], max_planets: int) -> tuple[np.ndarray, ...]:
    n = min(len(planets), max_planets)
    owner = np.full((max_planets,), -1, dtype=np.int32)
    x = np.zeros((max_planets,), dtype=np.float32)
    y = np.zeros((max_planets,), dtype=np.float32)
    radius = np.zeros((max_planets,), dtype=np.float32)
    ships = np.zeros((max_planets,), dtype=np.float32)
    prod = np.zeros((max_planets,), dtype=np.float32)
    alive = np.zeros((max_planets,), dtype=np.bool_)

    for i in range(n):
        p = planets[i]
        owner[i] = int(p[1])
        x[i] = float(p[2])
        y[i] = float(p[3])
        radius[i] = float(p[4])
        ships[i] = float(p[5])
        prod[i] = float(p[6])
        alive[i] = True

    return owner, x, y, radius, ships, prod, alive


def reset(key: jax.Array, cfg: JAXOrbitWarsConfig) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    seeds = np.asarray(jax.random.randint(key, (cfg.num_envs,), minval=0, maxval=2**31 - 1), dtype=np.int64)

    owner_l, x_l, y_l, r_l, ships_l, prod_l, alive_l, init_x_l, init_y_l, av_l = ([] for _ in range(10))
    for i in range(cfg.num_envs):
        rng = np.random.RandomState(int(seeds[i]))
        # mirror official seed behavior using Python Random for map generation
        import random

        prng = random.Random(int(seeds[i]))
        angular_velocity = prng.uniform(0.025, 0.05)
        planets = orbit_wars.generate_planets(prng)
        initial_planets = [p.copy() for p in planets]

        num_groups = len(planets) // 4
        if num_groups > 0:
            home_group = prng.randint(0, num_groups - 1)
            base = home_group * 4
            if cfg.num_players == 2:
                planets[base][1] = 0
                planets[base][5] = 10
                planets[base + 3][1] = 1
                planets[base + 3][5] = 10
            else:
                for j in range(4):
                    planets[base + j][1] = j
                    planets[base + j][5] = 10

        owner, x, y, radius, ships, prod, alive = _pad_planets(planets, cfg.max_planets)
        _, init_x, init_y, _, _, _, _ = _pad_planets(initial_planets, cfg.max_planets)

        owner_l.append(owner)
        x_l.append(x)
        y_l.append(y)
        r_l.append(radius)
        ships_l.append(ships)
        prod_l.append(prod)
        alive_l.append(alive)
        init_x_l.append(init_x)
        init_y_l.append(init_y)
        av_l.append(angular_velocity)

    state = {
        "planet_owner": jnp.asarray(np.stack(owner_l), dtype=jnp.int32),
        "planet_x": jnp.asarray(np.stack(x_l), dtype=jnp.float32),
        "planet_y": jnp.asarray(np.stack(y_l), dtype=jnp.float32),
        "planet_radius": jnp.asarray(np.stack(r_l), dtype=jnp.float32),
        "planet_ships": jnp.asarray(np.stack(ships_l), dtype=jnp.float32),
        "planet_prod": jnp.asarray(np.stack(prod_l), dtype=jnp.float32),
        "planet_alive": jnp.asarray(np.stack(alive_l), dtype=jnp.bool_),
        "init_planet_x": jnp.asarray(np.stack(init_x_l), dtype=jnp.float32),
        "init_planet_y": jnp.asarray(np.stack(init_y_l), dtype=jnp.float32),
        "angular_velocity": jnp.asarray(np.asarray(av_l), dtype=jnp.float32),
        "fleet_owner": jnp.full((cfg.num_envs, cfg.max_fleets), -1, dtype=jnp.int32),
        "fleet_x": jnp.zeros((cfg.num_envs, cfg.max_fleets), dtype=jnp.float32),
        "fleet_y": jnp.zeros((cfg.num_envs, cfg.max_fleets), dtype=jnp.float32),
        "fleet_angle": jnp.zeros((cfg.num_envs, cfg.max_fleets), dtype=jnp.float32),
        "fleet_ships": jnp.zeros((cfg.num_envs, cfg.max_fleets), dtype=jnp.float32),
        "fleet_alive": jnp.zeros((cfg.num_envs, cfg.max_fleets), dtype=jnp.bool_),
        "fleet_overflow_attempts": jnp.zeros((cfg.num_envs,), dtype=jnp.int32),
        "fleet_overflow_steps": jnp.zeros((cfg.num_envs,), dtype=jnp.int32),
        "fleet_active_max": jnp.zeros((cfg.num_envs,), dtype=jnp.int32),
        "step": jnp.zeros((cfg.num_envs,), dtype=jnp.int32),
        "done": jnp.zeros((cfg.num_envs,), dtype=jnp.bool_),
    }
    return state, observe(state, cfg)


def observe(state: dict[str, jax.Array], cfg: JAXOrbitWarsConfig) -> dict[str, jax.Array]:
    del cfg
    planets = jnp.stack(
        [
            state["planet_owner"].astype(jnp.float32),
            state["planet_x"],
            state["planet_y"],
            state["planet_radius"],
            state["planet_ships"],
            state["planet_prod"],
            state["planet_alive"].astype(jnp.float32),
        ],
        axis=-1,
    )
    fleets = jnp.stack(
        [
            state["fleet_owner"].astype(jnp.float32),
            state["fleet_x"],
            state["fleet_y"],
            state["fleet_angle"],
            state["fleet_ships"],
            state["fleet_alive"].astype(jnp.float32),
        ],
        axis=-1,
    )
    return {
        "planets": planets,
        "fleets": fleets,
        "done": state["done"],
        "step": state["step"],
    }


def _insert_fleet_for_player(
    state: dict[str, jax.Array],
    action: dict[str, jax.Array],
    player: int,
    cfg: JAXOrbitWarsConfig,
) -> tuple[dict[str, jax.Array], jax.Array, jax.Array]:
    src_logits = action["source_logits"][:, player, :]  # [B, P]
    angles = action["angle"][:, player]  # [B]
    frac = jnp.clip(action["send_fraction"][:, player], 0.0, 1.0)  # [B]

    valid_source = (
        state["planet_alive"]
        & (state["planet_owner"] == player)
        & (state["planet_ships"] >= 1.0)
    )
    masked_logits = jnp.where(valid_source, src_logits, -1e9)
    src = jnp.argmax(masked_logits, axis=1)  # [B]

    b_idx = jnp.arange(cfg.num_envs)
    src_ships = state["planet_ships"][b_idx, src]
    src_x = state["planet_x"][b_idx, src]
    src_y = state["planet_y"][b_idx, src]
    src_r = state["planet_radius"][b_idx, src]
    chosen_valid = valid_source[b_idx, src]

    send_ships = jnp.floor(src_ships * frac)
    send_ships = jnp.where((send_ships < 1.0) & chosen_valid, 1.0, send_ships)
    send_ships = jnp.where(chosen_valid, send_ships, 0.0)

    updated_src_ships = src_ships - send_ships
    planet_ships = state["planet_ships"].at[b_idx, src].set(updated_src_ships)

    empty_mask = ~state["fleet_alive"]
    insert_idx = jnp.argmax(empty_mask, axis=1)
    has_empty = jnp.any(empty_mask, axis=1)

    start_x = src_x + jnp.cos(angles) * (src_r + 0.1)
    start_y = src_y + jnp.sin(angles) * (src_r + 0.1)
    launch_attempt = send_ships > 0.0
    should_insert = has_empty & launch_attempt
    overflow_attempt = launch_attempt & (~has_empty)

    idx = (b_idx, insert_idx)
    fleet_owner = state["fleet_owner"].at[idx].set(jnp.where(should_insert, player, state["fleet_owner"][idx]))
    fleet_x = state["fleet_x"].at[idx].set(jnp.where(should_insert, start_x, state["fleet_x"][idx]))
    fleet_y = state["fleet_y"].at[idx].set(jnp.where(should_insert, start_y, state["fleet_y"][idx]))
    fleet_angle = state["fleet_angle"].at[idx].set(jnp.where(should_insert, angles, state["fleet_angle"][idx]))
    fleet_ships = state["fleet_ships"].at[idx].set(jnp.where(should_insert, send_ships, state["fleet_ships"][idx]))
    fleet_alive = state["fleet_alive"].at[idx].set(jnp.where(should_insert, True, state["fleet_alive"][idx]))

    state = dict(state)
    state["planet_ships"] = planet_ships
    state["fleet_owner"] = fleet_owner
    state["fleet_x"] = fleet_x
    state["fleet_y"] = fleet_y
    state["fleet_angle"] = fleet_angle
    state["fleet_ships"] = fleet_ships
    state["fleet_alive"] = fleet_alive
    return state, overflow_attempt.astype(jnp.int32), launch_attempt.astype(jnp.int32)


def _rotate_planets(state: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    dx = state["init_planet_x"] - orbit_wars.CENTER
    dy = state["init_planet_y"] - orbit_wars.CENTER
    orbital_r = jnp.sqrt(dx * dx + dy * dy)
    rotating = state["planet_alive"] & ((orbital_r + state["planet_radius"]) < orbit_wars.ROTATION_RADIUS_LIMIT)

    init_angle = jnp.arctan2(dy, dx)
    current_angle = init_angle + state["angular_velocity"][:, None] * state["step"][:, None].astype(jnp.float32)

    new_x = orbit_wars.CENTER + orbital_r * jnp.cos(current_angle)
    new_y = orbit_wars.CENTER + orbital_r * jnp.sin(current_angle)

    x = jnp.where(rotating, new_x, state["planet_x"])
    y = jnp.where(rotating, new_y, state["planet_y"])
    return x, y


def _point_to_segment_distance_batch(
    px: float,
    py: float,
    x1: jax.Array,
    y1: jax.Array,
    x2: jax.Array,
    y2: jax.Array,
) -> jax.Array:
    vx = x2 - x1
    vy = y2 - y1
    wx = px - x1
    wy = py - y1
    l2 = vx * vx + vy * vy
    t = jnp.where(
        l2 > 0,
        jnp.clip((wx * vx + wy * vy) / jnp.where(l2 > 0, l2, 1.0), 0.0, 1.0),
        0.0,
    )
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    dx = proj_x - px
    dy = proj_y - py
    return jnp.sqrt(dx * dx + dy * dy)


def _swept_pair_hit_tensor(
    ax: jax.Array,
    ay: jax.Array,
    bx: jax.Array,
    by: jax.Array,
    p0x: jax.Array,
    p0y: jax.Array,
    p1x: jax.Array,
    p1y: jax.Array,
    r: jax.Array,
) -> jax.Array:
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r

    disc = b * b - 4.0 * a * c
    safe_a = jnp.where(a > 1e-12, a, 1.0)
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)

    linear_case = a < 1e-12
    linear_hit = c <= 0.0
    quad_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return jnp.where(linear_case, linear_hit, quad_hit)


def step_fn(state: dict[str, jax.Array], action: dict[str, jax.Array], cfg: JAXOrbitWarsConfig) -> tuple[dict[str, jax.Array], dict[str, jax.Array], jax.Array, jax.Array, dict[str, jax.Array]]:
    # production
    owner_active = (state["planet_owner"] != -1) & state["planet_alive"]
    planet_ships = state["planet_ships"] + jnp.where(owner_active, state["planet_prod"], 0.0)
    state = dict(state)
    state["planet_ships"] = planet_ships

    # launches (one action per player per env)
    overflow_attempts = jnp.zeros((cfg.num_envs,), dtype=jnp.int32)
    launch_attempts = jnp.zeros((cfg.num_envs,), dtype=jnp.int32)
    for player in range(cfg.num_players):
        state, overflow_p, launch_p = _insert_fleet_for_player(state, action, player, cfg)
        overflow_attempts = overflow_attempts + overflow_p
        launch_attempts = launch_attempts + launch_p

    state["fleet_overflow_attempts"] = state["fleet_overflow_attempts"] + overflow_attempts
    state["fleet_overflow_steps"] = state["fleet_overflow_steps"] + (
        overflow_attempts > 0
    ).astype(jnp.int32)

    # planet motion
    px, py = _rotate_planets(state)

    # fleet motion
    alive_f = state["fleet_alive"]
    ships = jnp.maximum(state["fleet_ships"], 1.0)
    speed = 1.0 + (cfg.ship_speed - 1.0) * (jnp.log(ships) / jnp.log(1000.0)) ** 1.5
    speed = jnp.minimum(speed, cfg.ship_speed)
    speed = jnp.where(alive_f, speed, 0.0)

    old_fx, old_fy = state["fleet_x"], state["fleet_y"]
    new_fx = old_fx + jnp.cos(state["fleet_angle"]) * speed
    new_fy = old_fy + jnp.sin(state["fleet_angle"]) * speed

    # bounds + sun checks
    in_bounds = (new_fx >= 0.0) & (new_fx <= orbit_wars.BOARD_SIZE) & (new_fy >= 0.0) & (new_fy <= orbit_wars.BOARD_SIZE)
    if cfg.use_segment_sun_check:
        sun_dist = _point_to_segment_distance_batch(
            float(orbit_wars.CENTER),
            float(orbit_wars.CENTER),
            old_fx,
            old_fy,
            new_fx,
            new_fy,
        )
        sun_hit = sun_dist < float(orbit_wars.SUN_RADIUS)
    else:
        sun_hit = ((new_fx - orbit_wars.CENTER) ** 2 + (new_fy - orbit_wars.CENTER) ** 2) < (orbit_wars.SUN_RADIUS ** 2)

    # collisions
    if cfg.use_swept_collision:
        ax = old_fx[:, :, None]
        ay = old_fy[:, :, None]
        bx = new_fx[:, :, None]
        by = new_fy[:, :, None]
        p0x = state["planet_x"][:, None, :]
        p0y = state["planet_y"][:, None, :]
        p1x = px[:, None, :]
        p1y = py[:, None, :]
        rr = state["planet_radius"][:, None, :]

        hit = _swept_pair_hit_tensor(ax, ay, bx, by, p0x, p0y, p1x, p1y, rr)
        hit = hit & state["planet_alive"][:, None, :] & alive_f[:, :, None]
    else:
        # approximate collisions (endpoint only for JIT throughput)
        dxp = new_fx[:, :, None] - px[:, None, :]
        dyp = new_fy[:, :, None] - py[:, None, :]
        dist2 = dxp * dxp + dyp * dyp
        hit = dist2 <= (state["planet_radius"][:, None, :] ** 2)
        hit = hit & state["planet_alive"][:, None, :] & alive_f[:, :, None]

    any_hit = jnp.any(hit, axis=2)
    hit_pid = jnp.argmax(hit, axis=2)

    # remove dead/consumed fleets
    fleet_alive_next = alive_f & in_bounds & (~sun_hit) & (~any_hit)

    # combat accumulation by owner and planet
    hit_ships = jnp.where(any_hit, state["fleet_ships"], 0.0)
    owner = jnp.where(any_hit, state["fleet_owner"], -1)

    per_owner = []
    for p in range(cfg.num_players):
        contrib = jnp.where(owner == p, hit_ships, 0.0)
        acc = jnp.zeros_like(state["planet_ships"])
        acc = acc.at[jnp.arange(cfg.num_envs)[:, None], hit_pid].add(contrib)
        per_owner.append(acc)
    ships_by_owner = jnp.stack(per_owner, axis=-1)  # [B, P, Players]

    top_idx = jnp.argmax(ships_by_owner, axis=-1)
    top_val = jnp.max(ships_by_owner, axis=-1)
    second_val = jnp.max(jnp.where(jax.nn.one_hot(top_idx, cfg.num_players, dtype=bool), -1.0, ships_by_owner), axis=-1)

    survivor = jnp.maximum(0.0, top_val - second_val)
    survivor_owner = jnp.where(survivor > 0.0, top_idx, -1)

    planet_owner = state["planet_owner"]
    planet_ships = state["planet_ships"]

    same_owner = survivor_owner == planet_owner
    planet_ships = jnp.where(same_owner, planet_ships + survivor, planet_ships - survivor)
    captured = (survivor > 0.0) & (~same_owner) & (planet_ships < 0.0)
    planet_owner = jnp.where(captured, survivor_owner, planet_owner)
    planet_ships = jnp.where(captured, -planet_ships, planet_ships)

    # write back
    state = dict(state)
    state["planet_owner"] = planet_owner
    state["planet_ships"] = planet_ships
    state["planet_x"] = px
    state["planet_y"] = py
    state["fleet_x"] = new_fx
    state["fleet_y"] = new_fy
    state["fleet_alive"] = fleet_alive_next
    state["step"] = state["step"] + 1

    active_fleets = jnp.sum(state["fleet_alive"].astype(jnp.int32), axis=1)
    state["fleet_active_max"] = jnp.maximum(state["fleet_active_max"], active_fleets)

    # termination + reward
    planet_owner_alive = jnp.where(state["planet_alive"], state["planet_owner"], -1)
    fleet_owner_alive = jnp.where(state["fleet_alive"], state["fleet_owner"], -1)

    owned_by = []
    for p in range(cfg.num_players):
        own_planet = jnp.any(planet_owner_alive == p, axis=1)
        own_fleet = jnp.any(fleet_owner_alive == p, axis=1)
        owned_by.append(own_planet | own_fleet)
    alive_players = jnp.stack(owned_by, axis=1)
    num_alive = jnp.sum(alive_players.astype(jnp.int32), axis=1)

    timeout_done = state["step"] >= (cfg.episode_steps - 1)
    done = timeout_done | (num_alive <= 1)
    state["done"] = done

    scores = []
    for p in range(cfg.num_players):
        ps = jnp.sum(jnp.where((state["planet_owner"] == p) & state["planet_alive"], state["planet_ships"], 0.0), axis=1)
        fs = jnp.sum(jnp.where((state["fleet_owner"] == p) & state["fleet_alive"], state["fleet_ships"], 0.0), axis=1)
        scores.append(ps + fs)
    scores = jnp.stack(scores, axis=1)
    winners = scores == jnp.max(scores, axis=1, keepdims=True)
    reward = winners.astype(jnp.float32) * 2.0 - 1.0  # [B, Players]

    obs = observe(state, cfg)
    info = {
        "scores": scores,
        "alive_players": alive_players.astype(jnp.int32),
        "fleet_overflow_attempts": state["fleet_overflow_attempts"],
        "fleet_overflow_steps": state["fleet_overflow_steps"],
        "fleet_active_max": state["fleet_active_max"],
        "fleet_capacity": jnp.full((cfg.num_envs,), cfg.max_fleets, dtype=jnp.int32),
        "launch_attempts_this_step": launch_attempts,
        "overflow_attempts_this_step": overflow_attempts,
    }
    return state, obs, reward, done, info


step = jax.jit(step_fn, static_argnames=("cfg",))
