import json
import math
from collections import namedtuple
from os import path
import random

# Named tuples for agent convenience.
# Planets and fleets share a common [id, owner, x, y, ...] prefix.
Planet = namedtuple(
    "Planet", ["id", "owner", "x", "y", "radius", "ships", "production"]
)
Fleet = namedtuple(
    "Fleet", ["id", "owner", "x", "y", "angle", "from_planet_id", "ships"]
)

# ---------------------------------------------------------
# CONSTANTS & BOARD SETUP
# ---------------------------------------------------------
BOARD_SIZE = 100.0            # Board is a continuous 100x100 2D space.
CENTER = BOARD_SIZE / 2.0     # The center of the board is (50, 50).
SUN_RADIUS = 10.0             # Fleets that pass within this distance to the center are destroyed.
ROTATION_RADIUS_LIMIT = 50.0  # Planets inside this limit revolve around the sun. Planets outside are static.
COMET_RADIUS = 1.0            # Radius of extra-solar temporary planets (comets).
COMET_PRODUCTION = 1          # Comets produce 1 ship per turn.
PLANET_CLEARANCE = 7          # Planets must be at least this far apart from each other.
MIN_PLANET_GROUPS = 5         # Minimum number of 4-planet groups (20 planets).
MAX_PLANET_GROUPS = 10        # Maximum number of 4-planet groups (40 planets).
MIN_STATIC_GROUPS = 3         # Minimum number of static (non-revolving) 4-planet groups (12 planets).
COMET_SPAWN_STEPS = [50, 150, 250, 350, 450] # Specific turns where comets spawn.

# ---------------------------------------------------------
# UTILITY MATH FUNCTIONS
# ---------------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def point_to_segment_distance(p, v, w):
    """
    Minimum distance from point `p` to a line segment defined by points `v` and `w`.
    Used to check if a fleet's path intersects the Sun's radius in a single tick.
    """
    l2 = (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2
    if l2 == 0.0:
        return distance(p, v) # v and w are the same point
    
    # t is the scalar projection of p onto the line segment v-w, clamped to [0, 1]
    t = max(
        0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2)
    )
    projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
    return distance(p, projection)

def swept_pair_hit(A, B, P0, P1, r):
    """
    Continuous swept-pair collision detection.
    This checks if a moving fleet (from A to B) collides with a moving planet (from P0 to P1)
    within collision radius `r` at ANY continuous time `t` between [0, 1] during the tick.
    
    It solves the quadratic inequality for distance <= radius:
    ||(A + (B-A)t) - (P0 + (P1-P0)t)||^2 <= r^2
    """
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    
    # Coefficients for a*t^2 + b*t + c <= 0
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    
    if a < 1e-12: # No relative movement between the two objects
        return c <= 0.0
        
    disc = b * b - 4.0 * a * c
    if disc < 0.0: # No real roots, objects never intersect
        return False
        
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a) # Entering the collision radius
    t2 = (-b + sq) / (2.0 * a) # Exiting the collision radius
    
    # Check if the collision interval [t1, t2] overlaps with the tick [0, 1]
    return t2 >= 0.0 and t1 <= 1.0


# ---------------------------------------------------------
# GENERATION FUNCTIONS
# ---------------------------------------------------------
def generate_planets(rng=None):
    """
    Generates planets with 4-fold rotational symmetry.
    This guarantees the game is perfectly fair for 2 or 4 players,
    as every quadrant is identical to the others rotated by 90 degrees.
    """
    if rng is None:
        rng = random
    planets = []
    num_q1 = rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
    id_counter = 0

    # Phase 1: Generate guaranteed STATIC planet groups (orbital distance + radius >= 50).
    static_groups = 0
    for _ in range(5000):
        if static_groups >= MIN_STATIC_GROUPS:
            break
        prod = rng.randint(1, 5)
        r = 1 + math.log(prod)
        angle = rng.uniform(0, math.pi / 2)  # Limit initial spawn to Quadrant 1 (Q1)
        
        min_orbital = ROTATION_RADIUS_LIMIT - r
        max_orbital = (BOARD_SIZE - CENTER - r) / max(math.cos(angle), math.sin(angle))
        if min_orbital > max_orbital:
            continue
            
        orbital_r = rng.uniform(min_orbital, max_orbital)
        x = CENTER + orbital_r * math.cos(angle)
        y = CENTER + orbital_r * math.sin(angle)

        # Validate board boundaries for all 4 symmetric reflections
        if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
            continue
        if (BOARD_SIZE - x) - r < 0 or (BOARD_SIZE - y) - r < 0:
            continue
            
        # Ensure they don't spawn overlapping the axes to maintain perfect symmetry
        if (x - CENTER) < r + 5 or (y - CENTER) < r + 5:
            continue

        ships = min(rng.randint(5, 99), rng.randint(5, 99))
        
        # 4 symmetric planets: Q1, Q2, Q3, Q4
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

    # Phase 2: Generate remaining planet groups (can be orbiting or static).
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

        # Reject if spawned inside or touching the Sun
        if orbital_radius < SUN_RADIUS + r + 10:
            continue

        # If static, make sure it's fully inside the board
        if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
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

        # Sweeping overlap check to ensure an orbiting planet won't collide with a static one 
        # at any point in its orbit.
        for tp in temp_planets:
            tp_orbital = distance((tp[2], tp[3]), (CENTER, CENTER))
            tp_is_rotating = tp_orbital + tp[4] < ROTATION_RADIUS_LIMIT

            for p in planets:
                p_orbital = distance((p[2], p[3]), (CENTER, CENTER))
                p_is_rotating = p_orbital + p[4] < ROTATION_RADIUS_LIMIT

                if distance((p[2], p[3]), (tp[2], tp[3])) < p[4] + tp[4] + PLANET_CLEARANCE:
                    valid = False
                    break

                # If one is rotating and one is static, check their minimum distance
                # over the entire rotation path using absolute difference in orbital radii
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
    """
    Generates an elliptical path for extra-solar objects (comets).
    Spawns them symmetrically so 4 identical comets fly across the board at once.
    """
    if rng is None:
        rng = random
    if comet_planet_ids is None:
        comet_planet_ids = set()
    else:
        comet_planet_ids = set(comet_planet_ids)
        
    for _ in range(300):
        # Generates a highly eccentric ellipse
        e = rng.uniform(0.75, 0.93)
        a = rng.uniform(60, 150)
        perihelion = a * (1 - e)
        if perihelion < SUN_RADIUS + COMET_RADIUS:
            continue

        b = a * math.sqrt(1 - e**2)
        c_val = a * e
        phi = rng.uniform(math.pi / 6, math.pi / 3)

        # Discretize the continuous ellipse into a dense array of points
        dense = []
        num = 5000
        for i in range(num):
            t = 0.3 * math.pi + 1.4 * math.pi * i / (num - 1)
            ex = c_val + a * math.cos(t)
            ey = b * math.sin(t)
            
            x = CENTER + ex * math.cos(phi) - ey * math.sin(phi)
            y = CENTER + ex * math.sin(phi) + ey * math.cos(phi)
            dense.append((x, y))

        # Re-sample the points so the comet moves exactly `comet_speed` units per tick
        path = [dense[0]]
        cum = 0.0
        target = comet_speed
        for i in range(1, len(dense)):
            cum += distance(dense[i], dense[i - 1])
            if cum >= target:
                path.append(dense[i])
                target += comet_speed

        # Clip the path to only points inside the board
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
        
        # Valid comet pass must last between 5 and 40 turns
        if not (5 <= len(visible) <= 40):
            continue

        # Map to 4 symmetric quadrants
        paths = [
            [[y, x] for x, y in visible],
            [[BOARD_SIZE - x, y] for x, y in visible],
            [[x, BOARD_SIZE - y] for x, y in visible],
            [[BOARD_SIZE - y, BOARD_SIZE - x] for x, y in visible],
        ]

        # Verify comet path doesn't crash into existing planets during its lifetime
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
            # No Sun crash
            if distance((cx, cy), (CENTER, CENTER)) < SUN_RADIUS + COMET_RADIUS:
                valid = False
                break

            sym_pts = [
                (cy, cx),
                (BOARD_SIZE - cx, cy),
                (cx, BOARD_SIZE - cy),
                (BOARD_SIZE - cy, BOARD_SIZE - cx),
            ]
            
            # No Static planet crash
            for planet in static_planets:
                for sp in sym_pts:
                    if distance(sp, (planet[2], planet[3])) < planet[4] + buf:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break

            # Calculate where orbiting planets will be on the specific tick the comet is here
            # to verify it doesn't crash into them
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

# ---------------------------------------------------------
# MAIN STEP LOOP (TURN RESOLUTION)
# ---------------------------------------------------------
def interpreter(state, env):
    """
    The engine resolving loop. Resolves 1 entire turn / tick.
    """
    configuration = env.configuration
    num_agents = len(state)
    obs0 = state[0].observation

    # [1. GAME INITIALIZATION (Step 0)]
    # Setup initial game state: planets, home bases, speeds
    if not hasattr(obs0, "planets") or not obs0.planets:
        # Resolve and stash episode seed hidden from agents so they can't predict comet spawns
        if not hasattr(env, "info") or env.info is None:
            env.info = {}
        seed = env.info.get("seed")
        if seed is None:
            seed = get(configuration, "seed", None)
        if seed is None:
            seed = random.randrange(2**31)
            
        try:
            configuration.seed = None
        except (AttributeError, TypeError):
            configuration["seed"] = None
        env.info["seed"] = seed
        init_rng = random.Random(seed)

        # Angular velocity randomly selected between [0.025, 0.05] rads/turn
        angular_velocity = init_rng.uniform(0.025, 0.05)
        obs0.angular_velocity = angular_velocity
        obs0.planets = generate_planets(init_rng)
        obs0.initial_planets = [p.copy() for p in obs0.planets]
        obs0.fleets = []
        obs0.next_fleet_id = 0
        obs0.comets = []
        obs0.comet_planet_ids = []

        # Assign home planets randomly. In a symmetric setup, picking one 4-group assigns a base to each player
        num_groups = len(obs0.planets) // 4
        if num_groups > 0:
            home_group = init_rng.randint(0, num_groups - 1)
            base = home_group * 4

            if num_agents == 2:
                obs0.planets[base][1] = 0  # Player 0 gets Q1 copy
                obs0.planets[base][5] = 10 # Start with 10 ships
                obs0.planets[base + 3][1] = 1  # Player 1 gets Q4 copy
                obs0.planets[base + 3][5] = 10
            elif num_agents == 4:
                for j in range(4):
                    obs0.planets[base + j][1] = j
                    obs0.planets[base + j][5] = 10

        # Duplicate initial internal state to all agents so they see the board
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

    # [2. COMET CLEANUP]
    # Remove expired comets before fleets act to prevent interacting with dead targets
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

    # [3. COMET SPAWN]
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
            # Random initial ship garrison for comets
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
                # Starts at (-99, -99), will immediately be moved to path index 0 on update
                planet = [
                    pid,
                    -1, # Neutral owner
                    -99,
                    -99,
                    COMET_RADIUS,
                    comet_ships,
                    COMET_PRODUCTION,
                ]
                obs0.planets.append(planet)
                obs0.initial_planets.append(planet[:])
            obs0.comets.append(group)

    # [4. FLEET LAUNCH (Agent Action Execution)]
    def process_moves(player_id, action):
        """Processes agent output and converts to new fleets."""
        if not action or not isinstance(action, list):
            return
        for move in action:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)

            from_planet = next((p for p in obs0.planets if p[0] == from_id), None)

            # Validations: Agent must own planet and have requested ships > 0
            if from_planet and from_planet[1] == player_id:
                if from_planet[5] >= ships and ships > 0:
                    from_planet[5] -= ships # Detract ships from garrison immediately
                    
                    # Spawn the fleet JUST OUTSIDE the collision radius of the home planet (+0.1)
                    # so it doesn't instantly collide with the planet it spawned from.
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

    # [5. PLANETARY PRODUCTION]
    for planet in obs0.planets:
        if planet[1] != -1: # Only owned planets (and comets temporarily owned) produce
            planet[5] += planet[6]

    # [6. PLANET MOVEMENT PRE-COMPUTATION]
    # We must calculate where the planets WILL be at the end of the tick
    # so we can use continuous swept-pair line segments to detect collisions.
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
        
        # If the planet is an inner planet, calculate its rotated position
        if initial_p is not None:
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx ** 2 + dy ** 2)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                initial_angle = math.atan2(dy, dx)
                current_angle = initial_angle + angular_velocity * step
                new_pos = (
                    CENTER + r * math.cos(current_angle),
                    CENTER + r * math.sin(current_angle),
                )
        planet_paths[planet[0]] = (old_pos, new_pos, True)

    # Update comet path progression index
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
                check = old_pos[0] >= 0 # Do not check collision if it just spawned at (-99, -99)
                planet_paths[pid] = (old_pos, new_pos, check)

    # [7. FLEET MOVEMENT & CONTINUOUS COLLISIONS]
    max_speed = configuration.shipSpeed
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in obs0.planets} # Track which fleets arrived at which planet

    for fleet in obs0.fleets:
        angle = fleet[4]
        ships = fleet[6]
        
        # Speed Formula: speed = 1.0 + (max_speed - 1) * [log(ships) / log(1000)] ^ 1.5
        speed = 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
        speed = min(speed, max_speed)
        
        old_pos = (fleet[2], fleet[3])
        fleet[2] += math.cos(angle) * speed
        fleet[3] += math.sin(angle) * speed
        new_pos = (fleet[2], fleet[3])

        # A) Check Planet Intersections
        # Run swept pair collision with all planets. Prioritize planets over out-of-bounds/sun 
        # so fast fleets still capture targets.
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

        # B) Check Out of Bounds
        if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
            fleets_to_remove.append(fleet)
            continue

        # C) Check Sun Intersection
        if point_to_segment_distance((CENTER, CENTER), old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(fleet)
            continue

    # [8. COMMIT PLANET MOVEMENT]
    for planet in obs0.planets:
        path = planet_paths.get(planet[0])
        if path is not None:
            planet[2], planet[3] = path[1] # Set position to `new_pos` calculated in step 6

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

    # [9. COMBAT RESOLUTION]
    for pid, planet_fleets in combat_lists.items():
        planet = next((p for p in obs0.planets if p[0] == pid), None)
        if not planet or not planet_fleets:
            continue

        # Sum incoming fleets by player
        player_ships = {}
        for fleet in planet_fleets:
            owner = fleet[1]
            player_ships[owner] = player_ships.get(owner, 0) + fleet[6]

        if not player_ships:
            continue

        # Sort the attacking forces by count (Top vs Second Top determines outcome)
        sorted_players = sorted(
            player_ships.items(), key=lambda item: item[1], reverse=True
        )
        top_player, top_ships = sorted_players[0]

        if len(sorted_players) > 1:
            second_ships = sorted_players[1][1]
            survivor_ships = top_ships - second_ships # Winner retains difference

            if sorted_players[0][1] == sorted_players[1][1]:
                survivor_ships = 0 # If tie, mutual destruction, no one survives

            survivor_owner = top_player if survivor_ships > 0 else -1
        else:
            survivor_owner = top_player
            survivor_ships = top_ships

        # Apply surviving fleet to the planetary garrison
        if survivor_ships > 0:
            if planet[1] == survivor_owner:
                # Reinforcement
                planet[5] += survivor_ships
            else:
                # Attack
                planet[5] -= survivor_ships
                if planet[5] < 0:
                    # Capture! Target's defense broken, planet flips ownership
                    planet[1] = survivor_owner
                    planet[5] = abs(planet[5])

    # Synchronize observation states across all agents
    for i in range(1, num_agents):
        state[i].observation.planets = obs0.planets
        state[i].observation.initial_planets = obs0.initial_planets
        state[i].observation.fleets = obs0.fleets
        state[i].observation.next_fleet_id = obs0.next_fleet_id
        state[i].observation.comets = obs0.comets
        state[i].observation.comet_planet_ids = obs0.comet_planet_ids

    # [10. TERMINATION & SCORING]
    terminated = False
    step = get(obs0, "step", 0)
    if step >= configuration.episodeSteps - 2:
        terminated = True # Max steps reached

    # Check if only one player is left alive
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

        # Final Score = Garrison ships + Fleet ships
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

# Helper method
def get(d, key, default):
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)

# Renderer / agents omitted
