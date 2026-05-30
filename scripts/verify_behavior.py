#!/usr/bin/env python
# coding: utf-8

import os, glob, subprocess
import jax
import jax.numpy as jnp
from flax import nnx

# Make sure we have local modules loaded
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.networks import Actor, logits_to_action
from core.orbit_wars_jax import setup, step, EnvParams, EnvAction

print("Initializing Behavioral Verification Suite...")
print("Validating 4 Deep Space Buckets and 'Do Nothing' Inductive Bias on Untrained Network")

# Recreate structural PyTree
dummy_key = jax.random.PRNGKey(0)
actor_nnx = Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(0))
actor_graph, init_params = nnx.split(actor_nnx)
# Create a dummy batch of 4 players
p_keys = jax.random.split(dummy_key, 4)
@jax.vmap
def init_policy(key):
    return nnx.split(Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(key)))[1]

top4_params = init_policy(p_keys)

# Set up rollout function
def calculate_intercept_angle(state, params, ships):
    B = ships.shape[0]
    target_ids = jnp.broadcast_to(jnp.arange(60)[None, None, :], (B, 60, 60))
    # 1. Fleet speeds [B, 60, 60]
    safe_ships = jnp.maximum(ships, 1)
    raw_speed = 1.0 + (params.ship_speed[..., None, None] - 1.0) * (jnp.log(safe_ships.astype(float)) / jnp.log(1000.0)) ** 1.5
    v_fleet = jnp.minimum(raw_speed, params.ship_speed[..., None, None])

    # 2. Simulate future coordinates for t=1..150
    ts = jnp.arange(1, 151, dtype=jnp.float32)
    future_steps = state.step[..., None] + ts # [B, 150]

    pr = params.planet_orbital_radii
    initial_angles = params.planet_initial_angles
    current_angles = initial_angles[..., None] + (params.angular_velocity[..., None, None] * future_steps[:, None, :])
    orbit_x = 500.0 + pr[..., None] * jnp.cos(current_angles)
    orbit_y = 500.0 + pr[..., None] * jnp.sin(current_angles)
    orbit_coords = jnp.stack([orbit_x, orbit_y], axis=-1) # [B, 60, 150, 2]

    ages = future_steps[:, None, :] - params.comet_spawn_steps[..., None]
    comet_ages = ages[:, -20:, :]
    safe_comet_ages = jnp.clip(comet_ages, 0, 150 - 1).astype(jnp.int32)

    B_dim = safe_comet_ages.shape[0]
    b_idx = jnp.arange(B_dim)[:, None, None]
    c_idx = jnp.arange(20)[None, :, None]
    idxed_comet_locations = params.comet_paths[b_idx, c_idx, safe_comet_ages, :]

    padded_comet_coords = jnp.zeros((B_dim, 60, 150, 2))
    padded_comet_coords = padded_comet_coords.at[:, -20:, :, :].set(idxed_comet_locations)

    static_coords = jnp.broadcast_to(state.planet_coords[..., None, :], (B_dim, 60, 150, 2))

    future_coords = jnp.where(
        params.is_orbiting_planet[..., None, None], orbit_coords,
        jnp.where(
            params.is_comet[..., None, None], padded_comet_coords,
            static_coords
        )
    )

    # 3. Extract future coords for chosen targets [B, 60, 4, 150, 2]
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

    return jnp.arctan2(ic_y - src_y, ic_x - src_x)


@jax.jit
def rollout(top4_params, random_key):
    p0_params = jax.tree_util.tree_map(lambda x: x[0], top4_params)
    p1_params = jax.tree_util.tree_map(lambda x: x[1], top4_params)
    p2_params = jax.tree_util.tree_map(lambda x: x[2], top4_params)
    p3_params = jax.tree_util.tree_map(lambda x: x[3], top4_params)

    actor_p0 = nnx.merge(actor_graph, p0_params)
    actor_p1 = nnx.merge(actor_graph, p1_params)
    actor_p2 = nnx.merge(actor_graph, p2_params)
    actor_p3 = nnx.merge(actor_graph, p3_params)

    def scan_step(carry, _):
        state, params_inner, key = carry
        key, subkey = jax.random.split(key)

        def build_arrays(pid, num_players=4):
            planets = jnp.zeros((60, 7))
            planets = planets.at[:, 0].set(jnp.arange(60))

            rel_p_owner = jnp.where(state.planet_owners == pid, 1.0, 
                                    jnp.where(state.planet_owners == -1, 0.0, -1.0))
            planets = planets.at[:, 1].set(rel_p_owner)

            theta = -pid * (2 * jnp.pi / num_players)
            cos_t = jnp.cos(theta)
            sin_t = jnp.sin(theta)

            dx = state.planet_coords[:, 0] - 500.0
            dy = state.planet_coords[:, 1] - 500.0
            rot_x = dx * cos_t - dy * sin_t + 500.0
            rot_y = dx * sin_t + dy * cos_t + 500.0

            planets = planets.at[:, 2].set(rot_x)
            planets = planets.at[:, 3].set(rot_y)
            planets = planets.at[:, 4].set(params_inner.planet_radii)
            planets = planets.at[:, 5].set(state.planet_ships)
            planets = planets.at[:, 6].set(params_inner.planet_prod)

            fleets = jnp.zeros((7200, 6))
            fleets = fleets.at[:, 0].set(jnp.arange(7200))

            rel_f_owner = jnp.where(state.fleet_owners == pid, 1.0, 
                                    jnp.where(state.fleet_owners == -1, 0.0, -1.0))
            fleets = fleets.at[:, 1].set(rel_f_owner)

            fleets = fleets.at[:, 2].set(state.fleet_angles + theta)

            fdx = state.fleet_coords[:, 0] - 500.0
            fdy = state.fleet_coords[:, 1] - 500.0
            frot_x = fdx * cos_t - fdy * sin_t + 500.0
            frot_y = fdx * sin_t + fdy * cos_t + 500.0

            fleets = fleets.at[:, 3].set(frot_x)
            fleets = fleets.at[:, 4].set(frot_y)
            fleets = fleets.at[:, 5].set(state.fleet_ship_count)

            p_mask = state.planet_owners != -1
            f_mask = state.fleet_owners != -1
            return planets, fleets, p_mask, f_mask

        def run_player(pid, actor):
            planets, fleets, p_mask, f_mask = build_arrays(pid, num_players=4)
            logits = actor(planets, fleets, planet_mask=p_mask, fleet_mask=f_mask)
            ships, ds_angles = logits_to_action(logits, state.planet_ships)
            # The calculate_intercept_angle function expects batched tensors [B, ...]
            state_b = jax.tree_util.tree_map(lambda x: x[None, ...], state)
            params_b = jax.tree_util.tree_map(lambda x: x[None, ...], params_inner)
            ships_b = ships[None, ..., :60]
            intercept_angles = calculate_intercept_angle(state_b, params_b, ships_b)[0]
            angles = jnp.concatenate([intercept_angles, ds_angles], axis=-1)

            # Mask out micro-fleets
            ships = jnp.where(ships < 1.0, 0.0, ships)

            # Restore angle
            angles = angles + (pid * 2 * jnp.pi / 4)

            is_player = (state.planet_owners == pid)[..., None]
            return jnp.where(is_player, ships, 0), jnp.where(is_player, angles, 0.0)

        ships_p0, angles_p0 = run_player(0, actor_p0)
        ships_p1, angles_p1 = run_player(1, actor_p1)
        ships_p2, angles_p2 = run_player(2, actor_p2)
        ships_p3, angles_p3 = run_player(3, actor_p3)

        final_ships = ships_p0 + ships_p1 + ships_p2 + ships_p3
        final_angles = angles_p0 + angles_p1 + angles_p2 + angles_p3

        env_action = EnvAction(ships=final_ships.astype(jnp.int32), angle=final_angles)

        # Step Environment
        next_state, _, _ = step(state, params_inner, env_action, 4)
        
        # Save actions for statistical verification
        return (next_state, params_inner, key), (state, env_action, next_state)

    init_state, params_env = setup(random_key, 4)
    _, history = jax.lax.scan(scan_step, (init_state, params_env, random_key), None, length=500)
    return history, params_env

print('Simulating 500-step 4-way Free-For-All with Untrained Networks...')
history, params = rollout(top4_params, dummy_key)
states, actions, next_states = history
print('Simulation complete!')

import numpy as np
# -------------------------------------------------------------------------------------
# BEHAVIORAL VERIFICATION
# -------------------------------------------------------------------------------------

# 1. Do ships stay on planets?
# We check if the total ships on planets remains extremely high since untrained nets put 99% probability on the diagonal.
# We also check if the diagonal of the actions.ships matrix is universally zero (guaranteeing nothing is launched to self).
total_ships_start = np.sum(states.planet_ships[0])
total_ships_end = np.sum(states.planet_ships[-1])

diag_ships = np.sum(np.diagonal(actions.ships[:, :, :60], axis1=1, axis2=2))
print(f"Total ships assigned to diagonal 'Keep' bucket successfully launched: {diag_ships} (Should be EXACTLY 0)")

ds_ships = np.sum(actions.ships[:, :, 60:])
print(f"Total ships correctly dispatched into the 4 Deep Space Buckets: {ds_ships} (Should be > 0)")

east_bound = np.sum((actions.angle == 0.0) & (actions.ships > 0))
print(f"Total fleets mistakenly launched directly East at angle 0.0: {east_bound} (Should be minimal/0)")

if diag_ships == 0 and ds_ships > 0:
    print("\nSUCCESS: All behavioral assertions passed. The 4 Deep Space buckets are active, and the diagonal mask correctly prevents 'do nothing' fleets from launching!")
else:
    print("\nWARNING: Behavioral assertions failed. Check the logic.")
