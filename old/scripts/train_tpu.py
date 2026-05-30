import os
import time
import subprocess
import threading
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import functools
import glob

# Show XLA Compilation logs
os.environ["JAX_LOG_COMPILES"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from nnx_pgame import Transition
from nnx_pgame import NNXPgameEmitter, NNXPgameConfig
from qdax.utils.metrics import default_qd_metrics
import orbax.checkpoint as ocp

from networks import Actor, logits_to_action
import orbit_wars_jax

# =====================================================================
# 1. CLOUDFLARE R2 SYNC THREAD
# =====================================================================
class R2SyncThread(threading.Thread):
    def __init__(self, local_dir: str, bucket_name: str, sync_interval: int = 300):
        super().__init__(daemon=True)
        self.local_dir = local_dir
        self.bucket_name = bucket_name
        self.sync_interval = sync_interval
        os.makedirs(self.local_dir, exist_ok=True)
        
        from dotenv import load_dotenv
        load_dotenv()
        
        self.endpoint = os.environ.get("R2_ENDPOINT_URL", "")
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID", "")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        os.environ["AWS_DEFAULT_REGION"] = "auto"
        
    def run(self):
        print(f"Started R2 Sync Thread. Syncing {self.local_dir} to s3://{self.bucket_name} every {self.sync_interval}s.")
        while True:
            time.sleep(self.sync_interval)
            try:
                cmd = ["aws", "s3", "sync", self.local_dir, f"s3://{self.bucket_name}/v4/", "--endpoint-url", self.endpoint]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    pass # silent success
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] R2 Sync Failed: {result.stderr}")
            except Exception as e:
                print(f"Failed to sync to R2: {e}")


# =====================================================================
# 2. DUAL-FORMAT SELF-PLAY SCORING FUNCTION
# =====================================================================
def get_scoring_fn(actor_graph):
    from nnx_pgame import Transition
    
    # Helper to construct environment mask arrays
    def build_env_arrays(batch_size, state, params, pid, num_players):
        planets = jnp.zeros((batch_size, 60, 7))
        planets = planets.at[..., 0].set(jnp.arange(60))
        
        rel_p_owner = jnp.where(state.planet_owners == pid, 1.0, 
                                jnp.where(state.planet_owners == -1, 0.0, -1.0))
        planets = planets.at[..., 1].set(rel_p_owner)
        
        theta = -pid * (2 * jnp.pi / num_players)
        cos_t = jnp.cos(theta)
        sin_t = jnp.sin(theta)
        
        dx = state.planet_coords[..., 0] - 500.0
        dy = state.planet_coords[..., 1] - 500.0
        rot_x = dx * cos_t - dy * sin_t + 500.0
        rot_y = dx * sin_t + dy * cos_t + 500.0
        
        planets = planets.at[..., 2].set(rot_x)
        planets = planets.at[..., 3].set(rot_y)
        planets = planets.at[..., 4].set(params.planet_radii)
        planets = planets.at[..., 5].set(state.planet_ships)
        planets = planets.at[..., 6].set(params.planet_prod)
        
        fleets = jnp.zeros((batch_size, 7200, 6))
        fleets = fleets.at[..., 0].set(jnp.arange(7200))
        
        rel_f_owner = jnp.where(state.fleet_owners == pid, 1.0, 
                                jnp.where(state.fleet_owners == -1, 0.0, -1.0))
        fleets = fleets.at[..., 1].set(rel_f_owner)
        
        fleets = fleets.at[..., 2].set(state.fleet_angles + theta)
        
        fdx = state.fleet_coords[..., 0] - 500.0
        fdy = state.fleet_coords[..., 1] - 500.0
        frot_x = fdx * cos_t - fdy * sin_t + 500.0
        frot_y = fdx * sin_t + fdy * cos_t + 500.0
        
        fleets = fleets.at[..., 3].set(frot_x)
        fleets = fleets.at[..., 4].set(frot_y)
        fleets = fleets.at[..., 5].set(state.fleet_ship_count)
        
        p_mask = state.planet_owners != -1
        f_mask = state.fleet_owners != -1
        return planets, fleets, p_mask, f_mask
        
    def forward_pass(single_params, p, f, pm, fm):
        actor = nnx.merge(actor_graph, single_params)
        return actor(p, f, planet_mask=pm, fleet_mask=fm)
        
    vmap_forward = jax.vmap(forward_pass, in_axes=(0, 0, 0, 0, 0))

    def calculate_intercept_angle(state, params, ships):
        B = ships.shape[0]
        target_ids = jnp.broadcast_to(jnp.arange(60)[None, None, :], (B, 60, 60))
        # 1. Fleet speeds [B, 60, 60]
        safe_ships = jnp.maximum(ships, 1)
        raw_speed = 1.0 + (params.ship_speed[..., None, None] - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
        v_fleet = jnp.minimum(raw_speed, params.ship_speed[..., None, None])
        
        # 2. Simulate future coordinates for t=1..50
        ts = jnp.arange(1, 151, dtype=jnp.float32)
        future_steps = state.step[..., None] + ts # [B, 50]
        
        pr = params.planet_orbital_radii
        initial_angles = params.planet_initial_angles
        current_angles = initial_angles[..., None] + (params.angular_velocity[..., None, None] * future_steps[:, None, :])
        orbit_x = 50.0 + pr[..., None] * jnp.cos(current_angles)
        orbit_y = 50.0 + pr[..., None] * jnp.sin(current_angles)
        orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1) # [B, 60, 50, 2]
        
        ages = future_steps[:, None, :] - params.comet_spawn_steps[..., None]
        comet_ages = ages[:, -20:, :]
        safe_comet_ages = jnp.clip(comet_ages, 0, 150 - 1).astype(jnp.int32)
        
        B = safe_comet_ages.shape[0]
        b_idx = jnp.arange(B)[:, None, None]
        c_idx = jnp.arange(20)[None, :, None]
        idxed_comet_locations = params.comet_paths[b_idx, c_idx, safe_comet_ages, :]
        
        padded_comet_coords = jnp.zeros((B, 60, 150, 2))
        padded_comet_coords = padded_comet_coords.at[:, -20:, :, :].set(idxed_comet_locations)
        
        static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B, 60, 150, 2))
        
        future_coords = jnp.where(
            params.is_orbiting_planet[..., None, None], orbit_coords,
            jnp.where(
                params.is_comet[..., None, None], padded_comet_coords,
                static_coords
            )
        )
        
        # 3. Extract future coords for chosen targets [B, 60, 4, 50, 2]
        tfc = future_coords[b_idx, target_ids]
        
        # 4. Calculate Distance and Intercept Time
        src = state.planet_coords[:, :, None, None, :]
        dists = jnp.sqrt((tfc[..., 0] - src[..., 0])**2 + (tfc[..., 1] - src[..., 1])**2)
        
        # Account for fleet spawn offset (R_src + 0.1) and target collision radius (R_tgt)
        R_src = params.planet_radii[:, :, None, None]
        R_tgt = params.planet_radii[:, None, :, None]
        
        travel_dist = R_src + 0.1 + v_fleet[..., None] * ts[None, None, None, :]
        req_dist = dists - R_tgt
        
        can_reach = travel_dist >= req_dist
        intercept_t_idx = jnp.where(jnp.any(can_reach, axis=-1), jnp.argmax(can_reach, axis=-1), 149)
        
        # 5. Extract exact physical coordinates at intercept time
        idx = intercept_t_idx[..., None]
        ic_x = jnp.take_along_axis(tfc[..., 0], idx, axis=-1)[..., 0]
        ic_y = jnp.take_along_axis(tfc[..., 1], idx, axis=-1)[..., 0]
        
        src_x = state.planet_coords[..., 0][:, :, None]
        src_y = state.planet_coords[..., 1][:, :, None]
        
        standard_angles = jnp.arctan2(ic_y - src_y, ic_x - src_x)
        
        # 6. Deep Space 32-angle sweep (Raycast)
        N = 32
        test_angles = jnp.linspace(0, 2*jnp.pi, N, endpoint=False) # [32]
        ray_dx = jnp.cos(test_angles) # [32]
        ray_dy = jnp.sin(test_angles) # [32]
        
        P = state.planet_coords
        W = P[:, None, :, :] - P[:, :, None, :] # [B, src, tgt, 2]
        
        W_exp = W[..., None, :]
        V_exp = jnp.stack([ray_dx, ray_dy], axis=-1)[None, None, None, :, :]
        
        t = jnp.sum(W_exp * V_exp, axis=-1) # [B, 60, 60, 32]
        
        W_sq = jnp.sum(W**2, axis=-1)[..., None]
        d_sq = W_sq - t**2 # [B, 60, 60, 32]
        
        R = params.planet_radii[:, None, :, None]
        
        # Static Planet hit
        static_hit = (t > 0) & (d_sq < R**2)
        static_hit_dist = t - jnp.sqrt(jnp.maximum(0.0, R**2 - d_sq))
        
        # Blurred Annulus (Orbiting Planet) hit
        # Annulus is centered at (50, 50)
        W_center = jnp.array([50.0, 50.0])[None, None, None, :] - P[:, :, None, :] # [B, 60, 1, 2]
        t_center = jnp.sum(W_center * V_exp, axis=-1) # [B, 60, 1, 32] -> broadcast to [B, 60, 60, 32]
        W_center_sq = jnp.sum(W_center**2, axis=-1)[..., None]
        d_center_sq = W_center_sq - t_center**2
        
        pr = params.planet_orbital_radii[:, None, :, None]
        outer_R = pr + R
        
        # Does the ray intersect the outer radius of the orbit?
        orbit_hit = (t_center > 0) & (d_center_sq < outer_R**2)
        orbit_hit_dist = t_center - jnp.sqrt(jnp.maximum(0.0, outer_R**2 - d_center_sq))
        
        is_orbiting = params.is_orbiting_planet[:, None, :, None]
        
        hit = jnp.where(is_orbiting, orbit_hit, static_hit)
        # Note: We replace the single hit distance line below

        
        x = P[..., 0, None] # [B, 60, 1]
        y = P[..., 1, None] # [B, 60, 1]
        tx = jnp.where(ray_dx > 0, (100.0 - x) / (ray_dx + 1e-8), -x / (ray_dx - 1e-8))
        ty = jnp.where(ray_dy > 0, (100.0 - y) / (ray_dy + 1e-8), -y / (ray_dy - 1e-8))
        t_bounds = jnp.minimum(tx, ty) # [B, 60, 32]
        
        hit_dist = jnp.where(hit, jnp.where(is_orbiting, orbit_hit_dist, static_hit_dist), t_bounds[:, :, None, :])
        
        is_self_ray = jnp.arange(60)[None, :, None, None] == jnp.arange(60)[None, None, :, None]
        hit_dist = jnp.where(is_self_ray, t_bounds[:, :, None, :], hit_dist)
        
        min_hit_dist = jnp.min(hit_dist, axis=2) # [B, 60, 32]
        
        best_angle_idx = jnp.argmax(min_hit_dist, axis=-1) # [B, 60]
        deep_space_angles = test_angles[best_angle_idx][..., None] # [B, 60, 1]
        
        return jnp.concatenate([standard_angles, deep_space_angles], axis=-1)

    @jax.jit
    def run_1v1(policy_p, opp_p, random_key):
        b_size = jax.tree_util.tree_leaves(policy_p)[0].shape[0]
        rngs = jax.random.split(random_key, b_size)
        vmap_setup = jax.vmap(orbit_wars_jax.setup, in_axes=(0, None))
        init_state, params = vmap_setup(rngs, 2)
        
        def scan_step(carry, _):
            state, bd_state, prev_score = carry
            
            p0_planet_ships = jnp.sum(jnp.where(state.planet_owners == 0, state.planet_ships, 0), axis=-1)
            p0_fleet_ships = jnp.sum(jnp.where(state.fleet_owners == 0, state.fleet_ship_count, 0), axis=-1)
            p0_total_ships_prev = p0_planet_ships + p0_fleet_ships
            p0_produced = jnp.sum(jnp.where(state.planet_owners == 0, params.planet_prod, 0), axis=-1)
            p0_comet_prod = jnp.sum(jnp.where((state.planet_owners == 0) & params.is_comet, params.planet_prod, 0), axis=-1)
            
            p0_p, p0_f, p_mask, f_mask = build_env_arrays(b_size, state, params, pid=0, num_players=2)
            p1_p, p1_f, _, _ = build_env_arrays(b_size, state, params, pid=1, num_players=2)
            
            logits_p0 = vmap_forward(policy_p, p0_p, p0_f, p_mask, f_mask)
            logits_p1 = vmap_forward(opp_p, p1_p, p1_f, p_mask, f_mask)
            
            ships_p0, ds_angle_p0 = logits_to_action(logits_p0, state.planet_ships)
            ships_p1, ds_angle_p1 = logits_to_action(logits_p1, state.planet_ships)
            
            # Use Intercept Calculator to derive absolute angles
            intercept_p0 = calculate_intercept_angle(state, params, ships_p0[..., :60])
            angles_p0 = jnp.concatenate([intercept_p0, ds_angle_p0], axis=-1)
            intercept_p1 = calculate_intercept_angle(state, params, ships_p1[..., :60])
            angles_p1 = jnp.concatenate([intercept_p1, ds_angle_p1], axis=-1)
            
            # Mask out self-targeting and micro-fleets
            is_self = jnp.arange(60)[None, :, None] == jnp.arange(61)[None, None, :]
            ships_p0 = jnp.where(is_self | (ships_p0 < 1.0), 0.0, ships_p0)
            ships_p1 = jnp.where(is_self | (ships_p1 < 1.0), 0.0, ships_p1)
            
            is_p0 = (state.planet_owners == 0)[..., None]
            is_p1 = (state.planet_owners == 1)[..., None]
            
            final_ships = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, 0))
            final_ships = final_ships.astype(jnp.int32)
            final_angles = jnp.where(is_p0, angles_p0, jnp.where(is_p1, angles_p1, 0.0))
            
            env_action = orbit_wars_jax.EnvAction(ships=final_ships, angle=final_angles)
            vmap_step = jax.vmap(orbit_wars_jax.step, in_axes=(0, 0, 0, None))
            next_state, scores, done = vmap_step(state, params, env_action, 2)
            
            p0_planet_ships_next = jnp.sum(jnp.where(next_state.planet_owners == 0, next_state.planet_ships, 0), axis=-1)
            p0_fleet_ships_next = jnp.sum(jnp.where(next_state.fleet_owners == 0, next_state.fleet_ship_count, 0), axis=-1)
            p0_total_ships_next = p0_planet_ships_next + p0_fleet_ships_next
            ships_lost = jnp.maximum(0.0, p0_total_ships_prev + p0_produced - p0_total_ships_next)
            
            new_bd_state = (
                bd_state[0] + p0_fleet_ships_next,
                bd_state[1] + p0_planet_ships_next,
                bd_state[2] + p0_produced,
                bd_state[3] + ships_lost,
                bd_state[4] + p0_comet_prod
            )
            
            reward = scores[..., 0] - prev_score
            reward = jnp.where(done, reward * 100.0, reward)
            
            obs = (p0_p[::8], p0_f[::8])
            transition = Transition(
                obs=obs,
                actions=logits_p0[::8],
                rewards=reward[::8],
                dones=done[::8],
                next_obs=obs # Approximated for speed
            )
            
            return (next_state, new_bd_state, scores[..., 0]), (scores[..., 0], new_bd_state, transition)

        bd_init = (jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size))
        _, (scores_history, bd_history, transitions) = jax.lax.scan(scan_step, (init_state, bd_init, jnp.zeros(b_size, dtype=jnp.int32)), None, length=500)
        
        final_scores = scores_history[-1]
        final_bds = jnp.stack([x[-1] for x in bd_history], axis=-1)
        final_bds = final_bds / 500.0
        final_bds = final_bds[..., :4]
        
        max_vals = jnp.array([4000.0, 1000.0, 50.0, 50.0])
        final_bds = jnp.clip(final_bds / max_vals, 0.0, 1.0)
        
        return final_scores, final_bds, transitions
        
    @jax.jit
    def run_ffa(policy_p, opp1_p, opp2_p, opp3_p, random_key):
        b_size = jax.tree_util.tree_leaves(policy_p)[0].shape[0]
        rngs = jax.random.split(random_key, b_size)
        vmap_setup = jax.vmap(orbit_wars_jax.setup, in_axes=(0, None))
        init_state, params = vmap_setup(rngs, 4)
        
        def scan_step(carry, _):
            state, bd_state, prev_score = carry
            
            p0_planet_ships = jnp.sum(jnp.where(state.planet_owners == 0, state.planet_ships, 0), axis=-1)
            p0_fleet_ships = jnp.sum(jnp.where(state.fleet_owners == 0, state.fleet_ship_count, 0), axis=-1)
            p0_total_ships_prev = p0_planet_ships + p0_fleet_ships
            p0_produced = jnp.sum(jnp.where(state.planet_owners == 0, params.planet_prod, 0), axis=-1)
            p0_comet_prod = jnp.sum(jnp.where((state.planet_owners == 0) & params.is_comet, params.planet_prod, 0), axis=-1)
            
            p0_p, p0_f, p_mask, f_mask = build_env_arrays(b_size, state, params, pid=0, num_players=4)
            p1_p, p1_f, _, _ = build_env_arrays(b_size, state, params, pid=1, num_players=4)
            p2_p, p2_f, _, _ = build_env_arrays(b_size, state, params, pid=2, num_players=4)
            p3_p, p3_f, _, _ = build_env_arrays(b_size, state, params, pid=3, num_players=4)
            
            logits_p0 = vmap_forward(policy_p, p0_p, p0_f, p_mask, f_mask)
            logits_p1 = vmap_forward(opp1_p, p1_p, p1_f, p_mask, f_mask)
            logits_p2 = vmap_forward(opp2_p, p2_p, p2_f, p_mask, f_mask)
            logits_p3 = vmap_forward(opp3_p, p3_p, p3_f, p_mask, f_mask)
            
            ships_p0, ds_angle_p0 = logits_to_action(logits_p0, state.planet_ships)
            ships_p1, ds_angle_p1 = logits_to_action(logits_p1, state.planet_ships)
            ships_p2, ds_angle_p2 = logits_to_action(logits_p2, state.planet_ships)
            ships_p3, ds_angle_p3 = logits_to_action(logits_p3, state.planet_ships)
            
            intercept_p0 = calculate_intercept_angle(state, params, ships_p0[..., :60])
            angles_p0 = jnp.concatenate([intercept_p0, ds_angle_p0], axis=-1)
            intercept_p1 = calculate_intercept_angle(state, params, ships_p1[..., :60])
            angles_p1 = jnp.concatenate([intercept_p1, ds_angle_p1], axis=-1)
            intercept_p2 = calculate_intercept_angle(state, params, ships_p2[..., :60])
            angles_p2 = jnp.concatenate([intercept_p2, ds_angle_p2], axis=-1)
            intercept_p3 = calculate_intercept_angle(state, params, ships_p3[..., :60])
            angles_p3 = jnp.concatenate([intercept_p3, ds_angle_p3], axis=-1)
            
            # Mask out self-targeting and micro-fleets
            is_self = jnp.arange(60)[None, :, None] == jnp.arange(61)[None, None, :]
            mask = is_self | (ships_p0 < 1.0)
            ships_p0 = jnp.where(mask, 0.0, ships_p0)
            ships_p1 = jnp.where(is_self | (ships_p1 < 1.0), 0.0, ships_p1)
            ships_p2 = jnp.where(is_self | (ships_p2 < 1.0), 0.0, ships_p2)
            ships_p3 = jnp.where(is_self | (ships_p3 < 1.0), 0.0, ships_p3)
            
            is_p0 = (state.planet_owners == 0)[..., None]
            is_p1 = (state.planet_owners == 1)[..., None]
            is_p2 = (state.planet_owners == 2)[..., None]
            is_p3 = (state.planet_owners == 3)[..., None]
            
            final_ships = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, jnp.where(is_p2, ships_p2, jnp.where(is_p3, ships_p3, 0))))
            final_ships = final_ships.astype(jnp.int32)
            final_angles = jnp.where(is_p0, angles_p0, jnp.where(is_p1, angles_p1, jnp.where(is_p2, angles_p2, jnp.where(is_p3, angles_p3, 0.0))))
            
            env_action = orbit_wars_jax.EnvAction(ships=final_ships, angle=final_angles)
            vmap_step = jax.vmap(orbit_wars_jax.step, in_axes=(0, 0, 0, None))
            next_state, scores, done = vmap_step(state, params, env_action, 4)
            
            p0_planet_ships_next = jnp.sum(jnp.where(next_state.planet_owners == 0, next_state.planet_ships, 0), axis=-1)
            p0_fleet_ships_next = jnp.sum(jnp.where(next_state.fleet_owners == 0, next_state.fleet_ship_count, 0), axis=-1)
            p0_total_ships_next = p0_planet_ships_next + p0_fleet_ships_next
            ships_lost = jnp.maximum(0.0, p0_total_ships_prev + p0_produced - p0_total_ships_next)
            
            new_bd_state = (
                bd_state[0] + p0_fleet_ships_next,
                bd_state[1] + p0_planet_ships_next,
                bd_state[2] + p0_produced,
                bd_state[3] + ships_lost,
                bd_state[4] + p0_comet_prod
            )
            
            reward = scores[..., 0] - prev_score
            reward = jnp.where(done, reward * 100.0, reward)
            
            obs = (p0_p[::8], p0_f[::8])
            transition = Transition(
                obs=obs,
                actions=logits_p0[::8],
                rewards=reward[::8],
                dones=done[::8],
                next_obs=obs # Approximated for speed
            )
            
            return (next_state, new_bd_state, scores[..., 0]), (scores[..., 0], new_bd_state, transition)

        bd_init = (jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size))
        _, (scores_history, bd_history, transitions) = jax.lax.scan(scan_step, (init_state, bd_init, jnp.zeros(b_size, dtype=jnp.int32)), None, length=500)
        
        final_scores = scores_history[-1]
        final_bds = jnp.stack([x[-1] for x in bd_history], axis=-1)
        final_bds = final_bds / 500.0
        final_bds = final_bds[..., :4]
        
        max_vals = jnp.array([4000.0, 1000.0, 50.0, 50.0])
        final_bds = jnp.clip(final_bds / max_vals, 0.0, 1.0)
        
        return final_scores, final_bds, transitions

    @jax.jit
    def self_play_scoring_fn(policy_params, rep_genotypes, rep_fitnesses, random_key):
        # 1. Split Key
        key1, key2, key3 = jax.random.split(random_key, 3)
        batch_size = jax.tree_util.tree_leaves(policy_params)[0].shape[0]
        half_b = batch_size // 2
        
        # 2. Sample 3 opponents per individual in batch from repertoire
        valid_mask = (rep_fitnesses != -jnp.inf).squeeze()
        probs = jnp.where(valid_mask, 1.0, 0.0)
        probs = jnp.where(jnp.sum(probs) == 0, jnp.ones_like(probs), probs) # Fallback to uniform if empty
        
        sampled_indices = jax.random.choice(key1, jnp.arange(10000), shape=(batch_size, 3), p=probs)
        
        # Extract opponents
        opp_params = jax.tree_util.tree_map(lambda x: x[sampled_indices], rep_genotypes)
        
        # 3. Partition into 1v1 and FFA
        pol_1v1 = jax.tree_util.tree_map(lambda x: x[:half_b], policy_params)
        pol_ffa = jax.tree_util.tree_map(lambda x: x[half_b:], policy_params)
        
        opp1_1v1 = jax.tree_util.tree_map(lambda x: x[:half_b, 0], opp_params)
        
        opp1_ffa = jax.tree_util.tree_map(lambda x: x[half_b:, 0], opp_params)
        opp2_ffa = jax.tree_util.tree_map(lambda x: x[half_b:, 1], opp_params)
        opp3_ffa = jax.tree_util.tree_map(lambda x: x[half_b:, 2], opp_params)
        
        # 4. Evaluate Dual-Format
        fit_1v1, bd_1v1, trans_1v1 = run_1v1(pol_1v1, opp1_1v1, key2)
        fit_ffa, bd_ffa, trans_ffa = run_ffa(pol_ffa, opp1_ffa, opp2_ffa, opp3_ffa, key3)
        
        # 5. Re-assemble
        fitnesses = jnp.concatenate([fit_1v1, fit_ffa], axis=0)
        descriptors = jnp.concatenate([bd_1v1, bd_ffa], axis=0)
        
        transitions = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y], axis=1), trans_1v1, trans_ffa)
        
        return fitnesses, descriptors, {"transitions": transitions}, random_key
        
    return self_play_scoring_fn

# =====================================================================
# 3. MAIN TRAINING LOOP
# =====================================================================
def main():
    print("Initializing RL Pipeline...")
    from dotenv import load_dotenv; load_dotenv(); import os; R2SyncThread("/tmp/checkpoints_v4", os.environ.get("R2_BUCKET_NAME", "orbit-wars-checkpoints"), 300).start()
    
    print("Initializing NNX Architecture...")
    dummy_key = jax.random.PRNGKey(0)
    actor_nnx = Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(0))
    actor_graph, _ = nnx.split(actor_nnx)
    
    from networks import DoubleCritic
    critic_nnx = DoubleCritic(hidden_dim=32, num_sa_layers=4, rngs=nnx.Rngs(0))
    critic_graph, critic_params = nnx.split(critic_nnx)
    
    scoring_fn = get_scoring_fn(actor_graph)
    
    print("Computing CVT Centroids...")
    centroids = compute_cvt_centroids(
        num_descriptors=4,
        num_init_cvt_samples=100000,
        num_centroids=10000,
        minval=0.0,
        maxval=1.0,
        key=dummy_key
    )
    
    print("Configuring NNXPgameEmitter...")
    config = NNXPgameConfig(
        env_batch_size=128,
        pg_batch_size=32,
        buffer_size=20000,
        learning_starts=5000,
        gamma=0.99,
        mutation_iso_sigma=0.01,
        mutation_line_sigma=0.1
    )
    
    # Dummy Transition required to pre-allocate XLA Replay Buffer shapes
    dummy_planets = jnp.zeros((60, 7), dtype=jnp.float32)
    dummy_fleets = jnp.zeros((7200, 6), dtype=jnp.float32)
    dummy_obs = (dummy_planets, dummy_fleets)
    dummy_actions = jnp.zeros((60, 62), dtype=jnp.float32)
    dummy_reward = jnp.zeros((), dtype=jnp.float32)
    dummy_done = jnp.zeros((), dtype=jnp.bool_)
    
    dummy_transition = Transition(
        obs=dummy_obs,
        next_obs=dummy_obs,
        rewards=dummy_reward,
        dones=dummy_done,
        actions=dummy_actions
    )

    emitter = NNXPgameEmitter(
        config=config,
        actor_graph=actor_graph,
        critic_graph=critic_graph,
        dummy_transition=dummy_transition,
        init_critic_params=critic_params
    )
    
    print("Initializing Repertoire...")
    
    init_keys = jax.random.split(dummy_key, config.env_batch_size)
    
    @jax.vmap
    def init_policy(key):
        return nnx.split(Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(key)))[1]
    
    init_policy_params = init_policy(init_keys)

    # We need a dummy scoring function for map_elites.init() because it evaluates the initial batch!
    dummy_rep_genotypes = jax.tree_util.tree_map(lambda x: jnp.zeros((10000,) + x.shape[1:], dtype=x.dtype), init_policy_params)
    dummy_rep_fitnesses = jnp.full((10000,), -jnp.inf)
    
    def init_scoring_fn(genotypes, key):
        fit, desc, extra, _ = scoring_fn(genotypes, dummy_rep_genotypes, dummy_rep_fitnesses, key)
        return fit, desc, extra
        
    map_elites = MAPElites(
        scoring_function=init_scoring_fn,
        emitter=emitter,
        metrics_function=functools.partial(default_qd_metrics, qd_offset=0.0)
    )
    
    repertoire, emitter_state, metrics = map_elites.init(
        init_policy_params,
        centroids,
        dummy_key
    )
    
    checkpointer = ocp.PyTreeCheckpointer()
    
    # --- AUTO-RESUME LOGIC ---
    from dotenv import load_dotenv
    load_dotenv()
    r2_endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    r2_bucket = os.environ.get("R2_BUCKET_NAME", "")
    
    start_gen = 0
    os.makedirs("/tmp/checkpoints_v4/", exist_ok=True)
    if r2_endpoint and r2_bucket:
        print("Pulling latest checkpoints from R2 for Auto-Resume...")
        try:
            pull_cmd = ["aws", "s3", "sync", f"s3://{r2_bucket}/v4/", "/tmp/checkpoints_v4/", "--endpoint-url", r2_endpoint]
            subprocess.run(pull_cmd, capture_output=True)
        except Exception as e:
            print(f"Warning: Failed to pull from R2: {e}")
            
    # Find latest checkpoint
    checkpoint_dirs = glob.glob("/tmp/checkpoints_v4/qdax_rep_*")
    epochs = [int(d.split("_")[-1]) for d in checkpoint_dirs if d.split("_")[-1].isdigit()]
    if epochs:
        latest_gen = max(epochs)
        start_gen = latest_gen + 1
        latest_ckpt = f"/tmp/checkpoints_v4/qdax_rep_{latest_gen}"
        print(f"Resuming from checkpoint {latest_ckpt} (Generation {latest_gen})!")
        
        # Build Restore Args for Cross-Topology Loading (GPU -> TPU)
        from jax.sharding import SingleDeviceSharding
        default_device = jax.local_devices()[0]
        sharding = SingleDeviceSharding(default_device)
        sharding_tree = jax.tree_util.tree_map(lambda x: sharding, repertoire)
        restore_args = ocp.checkpoint_utils.construct_restore_args(repertoire, sharding_tree)
        
        repertoire = checkpointer.restore(latest_ckpt, item=repertoire, restore_args=restore_args)
    else:
        # Fallback to latest just in case
        latest_ckpt = "/tmp/checkpoints_v4/qdax_rep_latest"
        if os.path.exists(latest_ckpt):
            print(f"Resuming from checkpoint {latest_ckpt}!")
            
            from jax.sharding import SingleDeviceSharding
            default_device = jax.local_devices()[0]
            sharding = SingleDeviceSharding(default_device)
            sharding_tree = jax.tree_util.tree_map(lambda x: sharding, repertoire)
            restore_args = ocp.checkpoint_utils.construct_restore_args(repertoire, sharding_tree)
            
            repertoire = checkpointer.restore(latest_ckpt, item=repertoire, restore_args=restore_args)
        else:
            print("Starting MAP-Elites Evolutionary Loop from scratch!")
    # --------------------------
    
    random_key = dummy_key
    num_iterations = 10000
    for i in range(start_gen, num_iterations):
        start_time = time.time()
        
        # 1. EMIT
        random_key, emit_key, score_key, opp_key = jax.random.split(random_key, 4)
        genotypes, extra_info = map_elites._emitter.emit(
            repertoire, emitter_state, emit_key
        )
        
        # 2. EVALUATE DUAL-FORMAT WITH SELF-PLAY
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            genotypes, repertoire.genotypes, repertoire.fitnesses, score_key
        )
        
        # 3. UPDATE
        # QDax metrics requires passing metrics, but tell doesn't return metrics naturally unless we use update.
        # Actually, map_elites.tell returns (repertoire, emitter_state, metrics)
        extra_scores = {**extra_scores, **extra_info}
        repertoire, emitter_state, metrics = map_elites.tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            repertoire=repertoire,
            emitter_state=emitter_state
        )
        
        elapsed = time.time() - start_time
        
        # Get Max Fitness
        max_fitness = jnp.max(repertoire.fitnesses)
        coverage = jnp.sum(repertoire.fitnesses != -jnp.inf) / 10000.0 * 100.0
        
        print(f"[{i:04d}] Time: {elapsed:.2f}s | Max Fitness: {max_fitness:.2f} | Repertoire Coverage: {coverage:.1f}%")
        
        # VRAM Monitoring
        try:
            vram_info = subprocess.getoutput('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader').strip()
            if vram_info and 'not found' not in vram_info.lower() and 'failed' not in vram_info.lower():
                used, total = vram_info.split(', ')
                print(f"  -> GPU VRAM Usage: {used} MB / {total} MB ({int(used)/int(total)*100:.1f}%)")
        except Exception:
            pass
        
        # Historical Checkpoint Save
        if i % 10 == 0:
            save_path = f"/tmp/checkpoints_v4/qdax_rep_{i}"
            checkpointer.save(save_path, repertoire, force=True)
            print(f"  -> Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
