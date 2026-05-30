import jax
import jax.numpy as jnp
from qdax.environments.base_wrappers import QDEnv
from brax.envs import State as BraxState

import orbit_wars_jax
from orbit_wars_jax import EnvState, EnvParams, EnvAction

class OrbitWarsQDWrapper(QDEnv):
    def __init__(self, num_players=2):
        self.num_players = num_players
        
        # Calculate flat observation size to satisfy QDax
        # planet_coords: 60x2, planet_ships: 60, planet_owners: 60
        # fleet_coords: 7200x2, fleet_ships: 7200, fleet_angles: 7200, fleet_owners: 7200
        # planet_radii: 60, planet_prod: 60
        self._obs_size = (60 * 2) + 60 + 60 + (7200 * 2) + 7200 + 7200 + 7200 + 60 + 60
        
    @property
    def observation_size(self):
        return self._obs_size
        
    @property
    def action_size(self):
        return 60 * 10
        
    @property
    def behavior_descriptor_length(self):
        return 4
        
    def _flatten_obs(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        # Concatenate all necessary state and params into a 1D vector
        flat = jnp.concatenate([
            state.planet_coords.flatten(),
            state.planet_ships.flatten(),
            state.planet_owners.flatten(),
            state.fleet_coords.flatten(),
            state.fleet_ship_count.flatten(),
            state.fleet_angles.flatten(),
            state.fleet_owners.flatten(),
            params.planet_radii.flatten(),
            params.planet_prod.flatten(),
        ])
        return flat
        
    def _unflatten_obs(self, flat: jnp.ndarray) -> tuple:
        pass # Not strictly needed if Actor takes flat obs, but we will pass state directly in custom networks

    def reset(self, rng: jnp.ndarray) -> BraxState:
        state, params = orbit_wars_jax.setup(rng, num_players=self.num_players)
        obs = self._flatten_obs(state, params)
        
        # Track BD stats in the pipeline_state to avoid passing them manually
        # stats: [ships_lost, ships_built, fleet_ships_total, planet_ships_total, comets_targeted]
        stats = jnp.zeros(5)
        
        return BraxState(
            pipeline_state=(state, params, stats),
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(0.0),
            metrics={
                "space_affinity": jnp.array(0.0),
                "aggression": jnp.array(0.0),
                "concentration": jnp.array(0.0),
                "comet_affinity": jnp.array(0.0),
                "fitness": jnp.array(0.0) # Map Elites strictly looks at this metric
            },
            info={}
        )
        
    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        env_state, params, stats = state.pipeline_state
        
        # Action is flat logits [600] from our Actor for Player 0
        logits_p0 = action.reshape(60, 10)
        
        # Decode Player 0 Action
        from networks import logits_to_action
        ships_p0, angles_p0 = logits_to_action(logits_p0, env_state.planet_ships)
        
        # Player 1 (Opponent): Idle bot. Outputs 0 ships.
        ships_p1 = jnp.zeros_like(ships_p0)
        angles_p1 = jnp.zeros_like(angles_p0)
        
        # Combine actions into EnvAction (shape: [MAX_BODIES, MAX_FLEETS_PER_PLANET_PER_STEP])
        # Only home planets can launch. Player 0 owns planet base, Player 1 owns base+2.
        # But `logits_to_action` gives us outputs for ALL planets.
        # We must mask it so Player 0 only commands Player 0 planets.
        is_p0 = (env_state.planet_owners == 0)[:, None]
        is_p1 = (env_state.planet_owners == 1)[:, None]
        
        final_ships = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, 0))
        final_angles = jnp.where(is_p0, angles_p0, jnp.where(is_p1, angles_p1, 0.0))
        
        env_action = EnvAction(ships=final_ships, angle=final_angles)
        
        # Step the environment
        next_env_state, scores, done = orbit_wars_jax.step(env_state, params, env_action, num_players=self.num_players)
        
        # QDax Critic Reward (Dense): Difference in ships
        reward = scores[0] - scores[1]
        
        # Calculate BDs at the end of the game
        win = jnp.where(scores[0] > scores[1], 1.0, 0.0)
        metrics = {
            "space_affinity": jnp.array(0.0),
            "aggression": jnp.array(0.0),
            "concentration": jnp.array(0.0),
            "comet_affinity": jnp.array(0.0),
            "fitness": win
        }
        
        # Return new state
        obs = self._flatten_obs(next_env_state, params)
        return state.replace(
            pipeline_state=(next_env_state, params, stats),
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics
        )
