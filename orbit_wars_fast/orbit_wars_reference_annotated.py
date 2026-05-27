import json
import math
from collections import namedtuple
from os import path
import random

# Named tuples for agent convenience to access specific items easily
# Planets and fleets share a common [id, owner, x, y, ...] prefix.
Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

# Core game constants dictating physics and generation
BOARD_SIZE = 100.0            # Total space size (100x100 grid)
CENTER = BOARD_SIZE / 2.0     # The Sun's location (50, 50)
SUN_RADIUS = 10.0             # Fleets passing this close to the center are destroyed
ROTATION_RADIUS_LIMIT = 50.0  # Planets inside this radius revolve. Outside this radius, planets are static
COMET_RADIUS = 1.0            # Comets are small
COMET_PRODUCTION = 1          # Comets produce 1 ship per turn
PLANET_CLEARANCE = 7          # Planets must spawn at least this far apart from each other's edges
MIN_PLANET_GROUPS = 5         # Minimum groups of 4 planets
MAX_PLANET_GROUPS = 10        # Maximum groups of 4 planets
MIN_STATIC_GROUPS = 3         # Minimum groups of 4 planets that are static (don't rotate)
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450] # Step indices where extra-solar objects (comets) appear

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    """
    Minimum distance from point `p` to line segment defined by `v` and `w`.
    Used to check if a fleet crossed the Sun's radius.
    """
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v) # v == w
    # t is the parameterized position on the segment projected from point p
    t = max(
        0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def swept_pair_hit(A, B, P0, P1, r):
    """
    Continuous swept-pair collision detection.
    A -> B is the fleet's line segment over the tick.
    P0 -> P1 is the planet's line segment over the tick.
    r is the planet's collision radius.
    Returns True if the fleet and planet come within radius `r` at any time `t` in [0, 1].
    Solves the quadratic equation: || (A + (B-A)t) - (P0 + (P1-P0)t) ||^2 <= r^2
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    
    # Quadratic coefficients a*t^2 + b*t + c <= 0
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    
    # If a is near 0, the objects are not moving relative to each other.
    if a < 1e-12:
        return c <= 0.0
        
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False # No real roots, they never intersect
        
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a) # Entering the radius
    t2 = (-b + sq) / (2.0 * a) # Exiting the radius
    
    # True if the intersection interval overlaps with the tick [0, 1]
    return t2 >= 0.0 and t1 <= 1.0

def generate_planets(rng=None):
    """
    Generates symmetrically placed planets.
    The board exhibits 4-fold rotational symmetry (every quadrant looks identical rotated 90 deg).
    """
    if rng is None:
        rng = random
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    # Phase 1: Generate guaranteed static planet groups (planets that do not revolve around the sun).
    # Sample within the circular region where orbital_radius + r >= ROTATION_RADIUS_LIMIT.
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)  # Q1 angle from center
        min_orbital = ROTATION_RADIUS_LIMIT - r
        # Max orbital radius constrained by board edges
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        # Verify board bounds for all four symmetric copies
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
        # Ensure symmetric copies don't overlap: Q1 must be far enough from axes
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        
        # Symmetrical placement across 4 quadrants
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        # Check overlap with existing planets
        valid = True
        for tp in temp_planets:
            for p in planets:
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            planets.extend(temp_planets)
            id_counter += 4
            static_groups += 1

    # Phase 2: Fill remaining planet groups (can be static or rotating).
    attempts = 0
    max_attempts = 5000
    has_orbiting = False

    while len(planets) < num_q1 * 4 or (not has_orbiting and attempts < max_attempts):
        attempts += 1
        if attempts >= max_attempts:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        x = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
        y = rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)

        orbital_radius = distance((x, y), (CENTER, CENTER))

        # Reject if too close to sun
        if orbital_radius < SUN_RADIUS + r + 10:
            continue

        # Reject if planet body would extend past board edge during rotation
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
            # Planet is static (won't rotate), just check it stays on board
            if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                continue

        valid = True
        ships = rng.randint(5, 30)
        temp_planets = [
            [id_counter, -1, y, x, r, ships, prod],
            [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
            [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
            [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
        ]

        # Check overlaps for rotating planets over the course of their full orbit
        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                # Standard initial distance check
                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break

                # Cross-check: one rotating, one static -> min distance over
                # full rotation is |orbital_radius_1 - orbital_radius_2|
                if tp_is_rotating != p_is_rotating:
                    if abs(tp_orbital - p_orbital) < tp[4] + p[4] + PLANET_CLEARANCE:
                        valid = False
                        break

            if not valid:
                break

        if valid:
            if orbital_radius + r < ROTATION_RADIUS_LIMIT:
                has_orbiting = True
            planets.extend(temp_planets)
            id_counter += 4

    return planets

def generate_comet_paths(
    initial_planets,
    angular_velocity,
    spawn_step,
    comet_planet_ids=None,
    comet_speed=4.0,
    rng=None,
):
    """Generate 4 symmetric elliptical orbit paths for extra-solar objects (comets)."""
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)
        
    for _ in range(300):
        # Highly eccentric ellipse with sun at one focus
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        # Orientation: perihelion direction from sun (keep in Q4 quadrant)
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        # Dense sample around perihelion half of orbit
        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            # Ellipse with focus at origin
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            # Rotate and translate to board
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        # Re-sample at constant comet_speed arc-length intervals
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        # Extract contiguous on-board segment
        board_start = None
        board_end = None
        for i, (x, y) in enumerate(path):
            if 0 <= x <= BOARD_SIZE and 0 <= y <= BOARD_SIZE:
                if board_start is None:
                    board_start = i
                board_end = i

        if board_start is None:
            continue
        visible = path[board_start : board_end + 1]
        if not (5 <= len(visible) <= 40):
            continue

        # Build 4 rotationally symmetric paths
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        # Ensure comets won't collide with any planets
        static_planets = []
        orbiting_planets = []
        for planet in initial_planets:
            if planet[0] in comet_planet_ids:
                continue
            pr = distance((planet[2], planet[3]), (CENTER, CENTER))
            if pr + planet[4] < ROTATION_RADIUS_LIMIT:
                orbiting_planets.append(planet)
            else:
                static_planets.append(planet)

        valid = True
        buf = COMET_RADIUS + 0.5
        for k, (cx, cy) in enumerate(visible):
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            game_step = spawn_step - 1 + k
            for planet in orbiting_planets:
                dx = planet[2] - CENTER
                dy = planet[3] - CENTER
                orb_r = math.sqrt(dx**2 + dy**2)
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + angular_velocity * game_step
                px = CENTER + orb_r * math.cos(cur_angle)
                py = CENTER + orb_r * math.sin(cur_angle)
                for sp in sym_pts:
                    if distance(sp, (px, py)) < planet[4] + COMET_RADIUS:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

        if valid:
            return paths
    return None

def interpreter(state, env):
    """
    The main game loop step logic. Returns the new environment state.
    This resolves a single tick: launches, production, movement, collision, and combat.
    """
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    # [INITIALIZATION BLOCK]
    # Runs only at step 0 to set up planets, speeds, and home bases.
    if not hasattr(obs0, "planets") or not obs0.planets:
        if not hasattr(env, "info") or env.info is None:
            env.info = {}
        seed = env.info.get("seed")
        if seed is None:
            seed = get(configuration, "seed", None)
        if seed is None:
            seed = random.randrange(2**31)
            
        # Scrub seed to hide comet schedules from agents
        try:
            configuration.seed = None
        except (AttributeError, TypeError):
            configuration["seed"] = None
        env.info["seed"] = seed
        init_rng = random.Random(seed)

        angular_velocity = init_rng.uniform(0.025, 0.05)
        obs0.angular_velocity = angular_velocity
        obs0.planets = generate_planets(init_rng)
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []
        obs0.next_fleet_id = 0
        obs0.comets = []
        obs0.comet_planet_ids = []

        # Assign home planets — randomly choose one 4-group, assign 10 starting ships
        num_groups = len(obs0.planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4

            if num_agents == 2:
                obs0.planets[base][1] = 0  # Player 0 home
                obs0.planets[base][5] = 10
                obs0.planets[base + 3][1] = 1  # Player 1 home
                obs0.planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    obs0.planets[base + j][1] = j
                    obs0.planets[base + j][5] = 10

        # Duplicate initial state to all agents
        for i in range(num_agents):
            state[i].observation.player = i
            if i > 0:
                state[i].observation.angular_velocity = obs0.angular_velocity
                state[i].observation.planets = obs0.planets
                state[i].observation.initial_planets = obs0.initial_planets
                state[i].observation.fleets = obs0.fleets
                state[i].observation.next_fleet_id = obs0.next_fleet_id
                state[i].observation.comets = obs0.comets
                state[i].observation.comet_planet_ids = obs0.comet_planet_ids

        return state

    if env.done:
        return state

    # [CLEANUP] Remove expired comets before fleets act on them
    expired_comet_pids = []
    for group in obs0.comets:
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            if idx >= len(group["paths"][i]):
                expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    # [COMET SPAWN]
    step = get(obs0, "step", 0)
    comet_speed = configuration.cometSpeed
    if (step + 1) in COMET_SPAWN_STEPS:
        env_info = getattr(env, "info", None) or {}
        episode_seed = env_info.get("seed", 0) or 0
        comet_rng = random.Random(f"orbit_wars-comet-{episode_seed}-{step + 1}")
        comet_paths = generate_comet_paths(
            obs0.initial_planets,
            obs0.angular_velocity,
            step + 1,
            obs0.comet_planet_ids,
            comet_speed,
            rng=comet_rng,
        )
        if comet_paths:
            next_id = max(p[0] for p in obs0.planets) + 1
            comet_ships = min(
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
                comet_rng.randint(1, 99),
            )
            group = {"planet_ids": [], "paths": comet_paths, "path_index": -1}
            for i, p_path in enumerate(comet_paths):
                pid = next_id + i
                group["planet_ids"].append(pid)
                obs0.comet_planet_ids.append(pid)
                # Starts off-board (-99, -99), will snap to path index 0 on first update
                planet = [
                    pid, -1, -99, -99, COMET_RADIUS, comet_ships, COMET_PRODUCTION,
                ]
                obs0.planets.append(planet)
                obs0.initial_planets.append(planet[:])
            obs0.comets.append(group)

    # -------------------------------------------------------------
    # 0. FLEET LAUNCH
    # -------------------------------------------------------------
    def process_moves(player_id, action):
        """Processes and validates player actions to spawn fleets."""
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)

            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)

            # Validations: Valid planet, owned by player, enough ships, > 0 ships
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships  # Subtract from garrison immediately
                    
                    # Spawn fleet just outside the planet's radius to avoid self-collision
                    start_x = from_planet[2] + math.cos(angle) * (from_planet[4] + 0.1)
                    start_y = from_planet[3] + math.sin(angle) * (from_planet[4] + 0.1)
                    obs0.fleets.append(
                        [
                            obs0.next_fleet_id,
                            player_id,
                            start_x,
                            start_y,
                            angle,
                            from_id,
                            ships,
                        ]
                    )
                    obs0.next_fleet_id += 1

    for i in range(num_agents):
        process_moves(i, state[i].action)

    # -------------------------------------------------------------
    # 1. PRODUCTION
    # -------------------------------------------------------------
    for planet in obs0.planets:
        if planet[1] != -1:  # Non-neutral planets produce ships
            planet[5] += planet[6]

    # -------------------------------------------------------------
    # 2. PRE-COMPUTE PLANET MOVEMENT
    # -------------------------------------------------------------
    # Pre-calculate where planets will be at the end of the tick.
    # Needed to construct the (start, end) segment for swept-pair hit tests.
    angular_velocity = obs0.angular_velocity
    step = get(obs0, "step", 1)
    comet_pid_set = set(obs0.comet_planet_ids)
    initial_by_id = {p[0]: p for p in obs0.initial_planets}

    planet_paths = {}
    expired_comet_pids = []

    for planet in obs0.planets:
        if planet[0] in comet_pid_set:
            continue
        old_pos = (planet[2], planet[3])
        new_pos = old_pos
        initial_p = initial_by_id.get(planet[0])
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx ** 2 + dy ** 2)
            # Only revolve if orbital distance + radius < ROTATION_RADIUS_LIMIT (50.0)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy, dx)
                current_angle = initial_angle + angular_velocity * step
                new_pos = (
                    CENTER + r * math.cos(current_angle),
                    CENTER + r * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # Pre-compute comet movements along their paths
    for group in obs0.comets:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            planet = next((p for p in obs0.planets if p[0] == pid), None)
            if planet is None:
                continue
            p_path = group["paths"][i]
            old_pos = (planet[2], planet[3])
            if idx >= len(p_path):
                expired_comet_pids.append(pid)
                planet_paths[pid] = (old_pos, old_pos, True)
            else:
                new_pos = (p_path[idx][0], p_path[idx][1])
                # If old_pos is off-board (initial placement), don't check collisions this tick
                check = old_pos[0] >= 0
                planet_paths[pid] = (old_pos, new_pos, check)

    # -------------------------------------------------------------
    # 3. FLEET MOVEMENT & COLLISIONS
    # -------------------------------------------------------------
    max_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets}

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        # Speed scaling formula: 1 + (max - 1) * log_1000(ships)^1.5
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # Swept collision with all planets
        hit_planet = False
        for planet in obs0.planets:
            path = planet_paths.get(planet[0])
            if path is None or not path[2]:
                continue
            p_old, p_new, _ = path
            if swept_pair_hit(old_pos, new_pos, p_old, p_new, planet[4]):
                combat_lists[planet[0]].append(fleet)
                fleets_to_remove.append(fleet)
                hit_planet = True
                break
        if hit_planet:
            continue

        # Out of bounds check
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue

        # Sun collision check
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    # -------------------------------------------------------------
    # 4. APPLY PLANET MOVEMENT
    # -------------------------------------------------------------
    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1]

    # Remove any expired comets after they complete their path
    if expired_comet_pids:
        expired_set = set(expired_comet_pids)
        obs0.planets = [p for p in obs0.planets if p[0] not in expired_set]
        obs0.initial_planets = [
            p for p in obs0.initial_planets if p[0] not in expired_set
        ]
        obs0.comet_planet_ids = [
            pid for pid in obs0.comet_planet_ids if pid not in expired_set
        ]
        for group in obs0.comets:
            group["planet_ids"] = [
                pid for pid in group["planet_ids"] if pid not in expired_set
            ]
        obs0.comets = [g for g in obs0.comets if g["planet_ids"]]

    obs0.fleets = [f for f in obs0.fleets if f not in fleets_to_remove]

    # -------------------------------------------------------------
    # 5. COMBAT RESOLUTION
    # -------------------------------------------------------------
    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue

        # Accumulate arriving ships by player ID
        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

        if not player_ships:
            continue

        # Sort players by the number of ships they sent
        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        # Determine survivors
        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships

            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0 # Tie means mutual destruction

            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        # Apply survivors to the planet's garrison
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                # Reinforcements
                planet[5] += survivor_ships
            else:
                # Attackers
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    # Capture! The planet flips ownership
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    # Sync state for all agents
    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids

    # -------------------------------------------------------------
    # 6. TERMINATION
    # -------------------------------------------------------------
    terminated = False
    step = get(obs0, "step", 0)
    if step >= configuration.episodeSteps - 2:
        terminated = True

    # Check if only one player is alive
    alive_players = set()
    for p in obs0.planets:
        if p[1] != -1:
            alive_players.add(p[1])
    for f in obs0.fleets:
        alive_players.add(f[1])

    if len(alive_players) <= 1:
        terminated = True

    if terminated:
        for s in state:
            s.status = "DONE"

        # Tally final scores: Ships on planets + ships in transit
        scores = [0] * num_agents
        for p in obs0.planets:
            if p[1] != -1:
                scores[p[1]] += p[5]
        for f in obs0.fleets:
            scores[f[1]] += f[6]

        max_score = max(scores)
        for i in range(num_agents):
            if scores[i] == max_score and max_score > 0:
                state[i].reward = 1
            else:
                state[i].reward = -1

    return state

def get(d, key, default):
    """Helper to get from dict or SimpleNamespace."""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)

# (Renderers and starter agents omitted for brevity as they don't affect logic)