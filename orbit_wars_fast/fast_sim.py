from __future__ import annotations

import importlib
import math
import random
from typing import Any

import numpy as np
import kaggle_environments.envs.orbit_wars.orbit_wars as orbit_wars

try:
    njit = importlib.import_module("numba").njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _swept_pair_hit_numba(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        p0x: float,
        p0y: float,
        p1x: float,
        p1y: float,
        r: float,
    ) -> bool:
        d0x = ax - p0x
        d0y = ay - p0y
        dvx = (bx - ax) - (p1x - p0x)
        dvy = (by - ay) - (p1y - p0y)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - r * r
        if a < 1e-12:
            return c <= 0.0
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return False
        sq = math.sqrt(disc)
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
        return t2 >= 0.0 and t1 <= 1.0

    @njit(cache=True)
    def _first_hit_index_numba(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        candidate_indices: np.ndarray,
        collision_old_x: np.ndarray,
        collision_old_y: np.ndarray,
        collision_new_x: np.ndarray,
        collision_new_y: np.ndarray,
        collision_r: np.ndarray,
    ) -> int:
        for k in range(candidate_indices.shape[0]):
            idx = int(candidate_indices[k])
            if _swept_pair_hit_numba(
                ax,
                ay,
                bx,
                by,
                float(collision_old_x[idx]),
                float(collision_old_y[idx]),
                float(collision_new_x[idx]),
                float(collision_new_y[idx]),
                float(collision_r[idx]),
            ):
                return idx
        return -1

    _numba_warmed = False

    def _warm_numba_kernels() -> None:
        global _numba_warmed
        if _numba_warmed:
            return
        dummy_idx = np.asarray([0], dtype=np.int64)
        dummy = np.asarray([0.0], dtype=np.float64)
        _ = _first_hit_index_numba(
            0.0,
            0.0,
            1.0,
            1.0,
            dummy_idx,
            dummy,
            dummy,
            dummy,
            dummy,
            np.asarray([1.0], dtype=np.float64),
        )
        _numba_warmed = True
else:

    def _warm_numba_kernels() -> None:
        return


def _get(d: Any, key: str, default: Any) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def random_agent_action(obs: dict[str, Any], rng: random.Random) -> list[list[Any]]:
    """Random legal-ish policy for fuzz/accuracy testing."""
    player = obs.get("player", 0)
    moves: list[list[Any]] = []
    for p in obs.get("planets", []):
        if p[1] != player:
            continue
        ships_avail = int(p[5])
        if ships_avail <= 0:
            continue
        if rng.random() < 0.30:
            send = int(rng.randint(1, ships_avail))
            angle = rng.random() * 2.0 * math.pi
            moves.append([int(p[0]), angle, send])
    return moves


class FastOrbitWarsSimulator:
    """High-speed, step-by-step Orbit Wars simulator.

    The mechanics are intentionally kept operation-equivalent to Kaggle's
    official `orbit_wars.interpreter` so states can be compared byte-accurately.
    """

    def __init__(
        self,
        num_agents: int = 2,
        seed: int | None = None,
        use_numba: bool = False,
    ):
        if num_agents not in (2, 4):
            raise ValueError("Orbit Wars supports only 2 or 4 agents")
        self.num_agents = num_agents
        cfg = dict(orbit_wars.specification.get("configuration", {}))

        def _cfg_value(key: str, default: Any) -> Any:
            value = cfg.get(key, default)
            if isinstance(value, dict):
                return value.get("default", default)
            return value

        self.episode_steps = int(_cfg_value("episodeSteps", 500))
        self.ship_speed = float(_cfg_value("shipSpeed", 6.0))
        self.comet_speed = float(_cfg_value("cometSpeed", 4.0))
        self.seed: int = 0
        self.use_numba = bool(use_numba and NUMBA_AVAILABLE)

        self.angular_velocity: float = 0.0
        self.planets: list[list[Any]] = []
        self.initial_planets: list[list[Any]] = []
        self.fleets: list[list[Any]] = []
        self.next_fleet_id: int = 0
        self.comets: list[dict[str, Any]] = []
        self.comet_planet_ids: list[int] = []

        self.status: list[str] = ["ACTIVE"] * self.num_agents
        self.reward: list[float] = [0.0] * self.num_agents
        self.step_count: int = 0

        if self.use_numba:
            _warm_numba_kernels()

        self.reset(seed=seed)

    @property
    def done(self) -> bool:
        return all(s == "DONE" for s in self.status)

    def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        if seed is None:
            seed = random.randrange(2**31)
        self.seed = int(seed)

        init_rng = random.Random(self.seed)
        self.angular_velocity = init_rng.uniform(0.025, 0.05)

        self.planets = orbit_wars.generate_planets(init_rng)
        self.initial_planets = [p.copy() for p in self.planets]
        self.fleets = []
        self.next_fleet_id = 0
        self.comets = []
        self.comet_planet_ids = []

        num_groups = len(self.planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4
            if self.num_agents == 2:
                self.planets[base][1] = 0
                self.planets[base][5] = 10
                self.planets[base + 3][1] = 1
                self.planets[base + 3][5] = 10
            else:
                for j in range(4):
                    self.planets[base + j][1] = j
                    self.planets[base + j][5] = 10

        self.status = ["ACTIVE"] * self.num_agents
        self.reward = [0.0] * self.num_agents
        self.step_count = 0
        return self.observations()

    def observations(self) -> list[dict[str, Any]]:
        obs: list[dict[str, Any]] = []
        for i in range(self.num_agents):
            obs.append(
                {
                    "planets": self.planets,
                    "fleets": self.fleets,
                    "player": i,
                    "angular_velocity": self.angular_velocity,
                    "initial_planets": self.initial_planets,
                    "next_fleet_id": self.next_fleet_id,
                    "comets": self.comets,
                    "comet_planet_ids": self.comet_planet_ids,
                    "step": self.step_count,
                    "remainingOverageTime": 60,
                }
            )
        return obs

    def _remove_expired_comets_prelaunch(self) -> None:
        expired_comet_pids: list[int] = []
        for group in self.comets:
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                if idx >= len(group["paths"][i]):
                    expired_comet_pids.append(pid)

        if not expired_comet_pids:
            return

        expired_set = set(expired_comet_pids)
        self.planets = [p for p in self.planets if p[0] not in expired_set]
        self.initial_planets = [
            p for p in self.initial_planets if p[0] not in expired_set
        ]
        self.comet_planet_ids = [
            pid for pid in self.comet_planet_ids if pid not in expired_set
        ]

        for group in self.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        self.comets = [g for g in self.comets if g["planet_ids"]]

    def _spawn_comets_if_needed(self) -> None:
        if (self.step_count + 1) not in orbit_wars.COMET_SPAWN_STEPS:
            return

        comet_rng = random.Random(f"orbit_wars-comet-{self.seed}-{self.step_count + 1}")
        comet_paths = orbit_wars.generate_comet_paths(
            self.initial_planets,
            self.angular_velocity,
            self.step_count + 1,
            self.comet_planet_ids,
            self.comet_speed,
            rng=comet_rng,
        )
        if not comet_paths:
            return

        next_id = max(p[0] for p in self.planets) + 1
        comet_ships = min(
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
            comet_rng.randint(1, 99),
        )
        group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
        for i, p_path in enumerate(comet_paths):
            _ = p_path
            pid = next_id + i
            group["planet_ids"].append(pid)
            self.comet_planet_ids.append(pid)
            planet = [
                pid,
                -1,
                -99,
                -99,
                orbit_wars.COMET_RADIUS,
                comet_ships,
                orbit_wars.COMET_PRODUCTION,
            ]
            self.planets.append(planet)
            self.initial_planets.append(planet[:])
        self.comets.append(group)

    def _process_moves(self, player_id: int, action: Any) -> None:
        if not action or not isinstance(action, list):
            return

        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            from_planet = next((p for p in self.planets if p[0] == from_id), None)
            if from_planet is None or from_planet[1] != player_id:
                continue
            if ships <= 0 or from_planet[5] < ships:
                continue

            from_planet[5] -= ships
            start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
            start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
            self.fleets.append(
                [
                    self.next_fleet_id,
                    player_id,
                    start_x,
                    start_y,
                    angle,
                    from_id,
                    ships,
                ]
            )
            self.next_fleet_id += 1

    def step(
        self, actions: list[list[list[Any]]]
    ) -> tuple[list[dict[str, Any]], bool, list[float]]:
        if self.done:
            return self.observations(), True, self.reward.copy()

        # Keep exact order from official interpreter.
        self._remove_expired_comets_prelaunch()
        self._spawn_comets_if_needed()

        for i in range(self.num_agents):
            self._process_moves(i, actions[i] if i < len(actions) else [])

        # 1) Production
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        # 2) Compute each planet's end-of-tick position
        comet_pid_set = set(self.comet_planet_ids)
        initial_by_id = {p[0]: p for p in self.initial_planets}

        planet_paths: dict[
            int, tuple[tuple[float, float], tuple[float, float], bool]
        ] = {}
        expired_comet_pids: list[int] = []

        for planet in self.planets:
            pid = planet[0]
            if pid in comet_pid_set:
                continue
            old_pos = (planet[2], planet[3])
            new_pos = old_pos
            initial_p = initial_by_id.get(pid)
            if initial_p is not None:
                dx = initial_p[2] - orbit_wars.CENTER
                dy = initial_p[3] - orbit_wars.CENTER
                r = math.sqrt(dx**2 + dy**2)
                if r + planet[4] < orbit_wars.ROTATION_RADIUS_LIMIT:
                    initial_angle = math.atan2(dy, dx)
                    current_angle = (
                        initial_angle + self.angular_velocity * self.step_count
                    )
                    new_pos = (
                        orbit_wars.CENTER + r * math.cos(current_angle),
                        orbit_wars.CENTER + r * math.sin(current_angle),
                    )
            planet_paths[pid] = (old_pos, new_pos, True)

        for group in self.comets:
            group["path_index"] += 1
            idx = group["path_index"]
            for i, pid in enumerate(group["planet_ids"]):
                planet = next((p for p in self.planets if p[0] == pid), None)
                if planet is None:
                    continue
                p_path = group["paths"][i]
                old_pos = (planet[2], planet[3])
                if idx >= len(p_path):
                    expired_comet_pids.append(pid)
                    planet_paths[pid] = (old_pos, old_pos, True)
                else:
                    new_pos = (p_path[idx][0], p_path[idx][1])
                    check = old_pos[0] >= 0
                    planet_paths[pid] = (old_pos, new_pos, check)

        # 3) Fleet movement and swept collisions
        fleets_to_remove_ids: set[int] = set()
        combat_lists: dict[int, list[list[Any]]] = {p[0]: [] for p in self.planets}

        planet_by_id = {p[0]: p for p in self.planets}
        collision_rows: list[tuple[int, float, float, float, float, float]] = []
        for planet in self.planets:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:
                continue
            p_old, p_new, _ = path
            collision_rows.append(
                (
                    int(planet[0]),
                    float(planet[4]),
                    float(p_old[0]),
                    float(p_old[1]),
                    float(p_new[0]),
                    float(p_new[1]),
                )
            )

        if collision_rows:
            collision_pid = np.asarray([row[0] for row in collision_rows], dtype=np.int32)
            collision_r = np.asarray([row[1] for row in collision_rows], dtype=np.float64)
            collision_old_x = np.asarray([row[2] for row in collision_rows], dtype=np.float64)
            collision_old_y = np.asarray([row[3] for row in collision_rows], dtype=np.float64)
            collision_new_x = np.asarray([row[4] for row in collision_rows], dtype=np.float64)
            collision_new_y = np.asarray([row[5] for row in collision_rows], dtype=np.float64)
            collision_min_x = np.minimum(collision_old_x, collision_new_x) - collision_r
            collision_max_x = np.maximum(collision_old_x, collision_new_x) + collision_r
            collision_min_y = np.minimum(collision_old_y, collision_new_y) - collision_r
            collision_max_y = np.maximum(collision_old_y, collision_new_y) + collision_r
        else:
            collision_pid = None
            collision_r = None
            collision_old_x = None
            collision_old_y = None
            collision_new_x = None
            collision_new_y = None
            collision_min_x = None
            collision_max_x = None
            collision_min_y = None
            collision_max_y = None

        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            speed = (
                1.0
                + (self.ship_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
            )
            speed = min(speed, self.ship_speed)

            old_pos = (fleet[2], fleet[3])
            fleet[2] += math.cos(angle) * speed
            fleet[3] += math.sin(angle) * speed
            new_pos = (fleet[2], fleet[3])

            hit_planet = False
            if collision_pid is not None:
                fx_min = min(old_pos[0], new_pos[0])
                fx_max = max(old_pos[0], new_pos[0])
                fy_min = min(old_pos[1], new_pos[1])
                fy_max = max(old_pos[1], new_pos[1])

                broad_mask = (
                    (fx_max >= collision_min_x)
                    & (fx_min <= collision_max_x)
                    & (fy_max >= collision_min_y)
                    & (fy_min <= collision_max_y)
                )
                candidate_indices = np.nonzero(broad_mask)[0]
            else:
                candidate_indices = np.asarray([], dtype=np.int32)

            if collision_pid is not None:
                assert collision_old_x is not None
                assert collision_old_y is not None
                assert collision_new_x is not None
                assert collision_new_y is not None
                assert collision_r is not None

                if self.use_numba and candidate_indices.size > 0:
                    first_idx = _first_hit_index_numba(
                        float(old_pos[0]),
                        float(old_pos[1]),
                        float(new_pos[0]),
                        float(new_pos[1]),
                        candidate_indices,
                        collision_old_x,
                        collision_old_y,
                        collision_new_x,
                        collision_new_y,
                        collision_r,
                    )
                    if first_idx >= 0:
                        pid = int(collision_pid[int(first_idx)])
                        planet = planet_by_id.get(pid)
                        if planet is not None:
                            combat_lists[planet[0]].append(fleet)
                            fleets_to_remove_ids.add(fleet[0])
                            hit_planet = True
                else:
                    for idx in candidate_indices:
                        pid = int(collision_pid[idx])
                        planet = planet_by_id.get(pid)
                        if planet is None:
                            continue
                        if orbit_wars.swept_pair_hit(
                            old_pos,
                            new_pos,
                            (float(collision_old_x[idx]), float(collision_old_y[idx])),
                            (float(collision_new_x[idx]), float(collision_new_y[idx])),
                            float(collision_r[idx]),
                        ):
                            combat_lists[planet[0]].append(fleet)
                            fleets_to_remove_ids.add(fleet[0])
                            hit_planet = True
                            break
            if hit_planet:
                continue

            if not (
                0 <= fleet[2] <= orbit_wars.BOARD_SIZE
                and 0 <= fleet[3] <= orbit_wars.BOARD_SIZE
            ):
                fleets_to_remove_ids.add(fleet[0])
                continue

            if (
                orbit_wars.point_to_segment_distance(
                    (orbit_wars.CENTER, orbit_wars.CENTER), old_pos, new_pos
                )
                < orbit_wars.SUN_RADIUS
            ):
                fleets_to_remove_ids.add(fleet[0])
                continue

        # 4) Apply planet movement
        for planet in self.planets:
            path = planet_paths.get(planet[0])
            if path is not None:
                planet[2], planet[3] = path[1]

        if expired_comet_pids:
            expired_set = set(expired_comet_pids)
            self.planets = [p for p in self.planets if p[0] not in expired_set]
            self.initial_planets = [
                p for p in self.initial_planets if p[0] not in expired_set
            ]
            self.comet_planet_ids = [
                pid for pid in self.comet_planet_ids if pid not in expired_set
            ]
            for group in self.comets:
                group["planet_ids"] = [
                    pid for pid in group["planet_ids"] if pid not in expired_set
                ]
            self.comets = [g for g in self.comets if g["planet_ids"]]

        if fleets_to_remove_ids:
            self.fleets = [f for f in self.fleets if f[0] not in fleets_to_remove_ids]

        # 5) Combat
        for pid, planet_fleets in combat_lists.items():
            planet = next((p for p in self.planets if p[0] == pid), None)
            if planet is None or not planet_fleets:
                continue

            player_ships: dict[int, int] = {}
            for fleet in planet_fleets:
                owner = fleet[1]
                player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

            if not player_ships:
                continue

            sorted_players = sorted(
                player_ships.items(), key=lambda item: item[1], reverse=True
            )
            top_player, top_ships = sorted_players[0]

            if len(sorted_players) > 1:
                second_ships = sorted_players[1][1]
                survivor_ships = top_ships - second_ships
                if sorted_players[0][1] == sorted_players[1][1]:
                    survivor_ships = 0
                survivor_owner = top_player if survivor_ships > 0 else -1
            else:
                survivor_owner = top_player
                survivor_ships = top_ships

            if survivor_ships > 0:
                if planet[1] == survivor_owner:
                    planet[5] += survivor_ships
                else:
                    planet[5] -= survivor_ships
                    if planet[5] < 0:
                        planet[1] = survivor_owner
                        planet[5] = abs(planet[5])

        # 6) Termination
        terminated = False
        if self.step_count >= self.episode_steps - 2:
            terminated = True

        alive_players: set[int] = set()
        for p in self.planets:
            if p[1] != -1:
                alive_players.add(p[1])
        for f in self.fleets:
            alive_players.add(f[1])

        if len(alive_players) <= 1:
            terminated = True

        if terminated:
            self.status = ["DONE"] * self.num_agents
            scores = [0] * self.num_agents
            for p in self.planets:
                if p[1] != -1:
                    scores[p[1]] += p[5]
            for f in self.fleets:
                scores[f[1]] += f[6]
            max_score = max(scores)
            for i in range(self.num_agents):
                if scores[i] == max_score and max_score > 0:
                    self.reward[i] = 1.0
                else:
                    self.reward[i] = -1.0

        self.step_count += 1
        return self.observations(), self.done, self.reward.copy()

    def snapshot(self) -> dict[str, Any]:
        return {
            "step": self.step_count,
            "done": self.done,
            "planets": self.planets,
            "fleets": self.fleets,
            "initial_planets": self.initial_planets,
            "next_fleet_id": self.next_fleet_id,
            "comets": self.comets,
            "comet_planet_ids": self.comet_planet_ids,
            "angular_velocity": self.angular_velocity,
            "status": self.status,
            "reward": self.reward,
        }
