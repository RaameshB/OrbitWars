import json

with open("train.py", "r") as f:
    original_code = f.read()

# We will rewrite the scoring function and the main loop.
new_train_py = """import os
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
                cmd = ["aws", "s3", "sync", self.local_dir, f"s3://{self.bucket_name}/", "--endpoint-url", self.endpoint]
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
    def build_env_arrays(batch_size, state, params):
        planets = jnp.zeros((batch_size, 60, 7))
        planets = planets.at[..., 0].set(jnp.arange(60))
        planets = planets.at[..., 1].set(state.planet_owners)
        planets = planets.at[..., 2:4].set(state.planet_coords)
        planets = planets.at[..., 4].set(params.planet_radii)
        planets = planets.at[..., 5].set(state.planet_ships)
        planets = planets.at[..., 6].set(params.planet_prod)
        
        fleets = jnp.zeros((batch_size, 7200, 5))
        fleets = fleets.at[..., 0].set(jnp.arange(7200))
        fleets = fleets.at[..., 1].set(state.fleet_angles)
        fleets = fleets.at[..., 2:4].set(state.fleet_coords)
        fleets = fleets.at[..., 4].set(state.fleet_ship_count)
        
        p_mask = state.planet_owners != -1
        f_mask = state.fleet_owners != -1
        return planets, fleets, p_mask, f_mask
        
    def forward_pass(single_params, p, f, pm, fm):
        actor = nnx.merge(actor_graph, single_params)
        return actor(p, f, planet_mask=pm, fleet_mask=fm)
        
    vmap_forward = jax.vmap(forward_pass, in_axes=(0, 0, 0, 0, 0))

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
            
            planets, fleets, p_mask, f_mask = build_env_arrays(b_size, state, params)
            
            logits_p0 = vmap_forward(policy_p, planets, fleets, p_mask, f_mask)
            logits_p1 = vmap_forward(opp_p, planets, fleets, p_mask, f_mask)
            
            ships_p0, angles_p0 = logits_to_action(logits_p0, state.planet_ships)
            ships_p1, angles_p1 = logits_to_action(logits_p1, state.planet_ships)
            
            is_p0 = (state.planet_owners == 0)[..., None]
            is_p1 = (state.planet_owners == 1)[..., None]
            
            final_ships = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, 0))
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
            
            obs = (planets, fleets)
            transition = Transition(
                obs=obs,
                actions=logits_p0,
                rewards=reward,
                dones=done,
                next_obs=obs # Approximated for speed
            )
            
            return (next_state, new_bd_state, scores[..., 0]), (scores[..., 0], new_bd_state, transition)

        bd_init = (jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size))
        _, (scores_history, bd_history, transitions) = jax.lax.scan(scan_step, (init_state, bd_init, jnp.zeros(b_size)), None, length=500)
        
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
            
            planets, fleets, p_mask, f_mask = build_env_arrays(b_size, state, params)
            
            logits_p0 = vmap_forward(policy_p, planets, fleets, p_mask, f_mask)
            logits_p1 = vmap_forward(opp1_p, planets, fleets, p_mask, f_mask)
            logits_p2 = vmap_forward(opp2_p, planets, fleets, p_mask, f_mask)
            logits_p3 = vmap_forward(opp3_p, planets, fleets, p_mask, f_mask)
            
            ships_p0, angles_p0 = logits_to_action(logits_p0, state.planet_ships)
            ships_p1, angles_p1 = logits_to_action(logits_p1, state.planet_ships)
            ships_p2, angles_p2 = logits_to_action(logits_p2, state.planet_ships)
            ships_p3, angles_p3 = logits_to_action(logits_p3, state.planet_ships)
            
            is_p0 = (state.planet_owners == 0)[..., None]
            is_p1 = (state.planet_owners == 1)[..., None]
            is_p2 = (state.planet_owners == 2)[..., None]
            is_p3 = (state.planet_owners == 3)[..., None]
            
            final_ships = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, jnp.where(is_p2, ships_p2, jnp.where(is_p3, ships_p3, 0))))
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
            
            obs = (planets, fleets)
            transition = Transition(
                obs=obs,
                actions=logits_p0,
                rewards=reward,
                dones=done,
                next_obs=obs # Approximated for speed
            )
            
            return (next_state, new_bd_state, scores[..., 0]), (scores[..., 0], new_bd_state, transition)

        bd_init = (jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size), jnp.zeros(b_size))
        _, (scores_history, bd_history, transitions) = jax.lax.scan(scan_step, (init_state, bd_init, jnp.zeros(b_size)), None, length=500)
        
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
        
        return fitnesses, descriptors, {\"transitions\": transitions}, random_key
        
    return self_play_scoring_fn

# =====================================================================
# 3. MAIN TRAINING LOOP
# =====================================================================
def main():
    print(\"Initializing RL Pipeline...\")
    R2SyncThread(\"/tmp/checkpoints\", \"orbit-wars-checkpoints\", 300).start()
    
    print(\"Initializing NNX Architecture...\")
    dummy_key = jax.random.PRNGKey(0)
    actor_nnx = Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(0))
    actor_graph, _ = nnx.split(actor_nnx)
    
    from networks import DoubleCritic
    critic_nnx = DoubleCritic(hidden_dim=32, num_sa_layers=4, rngs=nnx.Rngs(0))
    critic_graph, _ = nnx.split(critic_nnx)
    
    scoring_fn = get_scoring_fn(actor_graph)
    
    print(\"Computing CVT Centroids...\")
    centroids = compute_cvt_centroids(
        num_descriptors=4,
        num_init_cvt_samples=100000,
        num_centroids=10000,
        minval=0.0,
        maxval=1.0,
        key=dummy_key
    )
    
    print(\"Configuring NNXPgameEmitter...\")
    config = NNXPgameConfig(
        env_batch_size=2048,
        proportion_mutation_ga=0.5,
        pg_batch_size=256,
        buffer_size=60000,
        learning_starts=5000,
        gamma=0.99,
        mutation_iso_sigma=0.01,
        mutation_line_sigma=0.1
    )
    
    emitter = NNXPgameEmitter(
        config=config,
        actor_graph=actor_graph,
        critic_graph=critic_graph
    )
    
    print(\"Initializing Repertoire...\")
    map_elites = MAPElites(
        scoring_function=None, # Bypassing the core function loop
        emitter=emitter,
        metrics_function=default_qd_metrics
    )
    
    init_keys = jax.random.split(dummy_key, config.env_batch_size)
    
    @jax.vmap
    def init_policy(key):
        return nnx.split(Actor(hidden_dim=32, num_sa_layers=6, rngs=nnx.Rngs(key)))[1]
    
    init_policy_params = init_policy(init_keys)
    
    repertoire, emitter_state, metrics = map_elites.init(
        init_policy_params,
        centroids,
        dummy_key
    )
    
    checkpointer = ocp.PyTreeCheckpointer()
    
    # --- AUTO-RESUME LOGIC ---
    from dotenv import load_dotenv
    load_dotenv()
    r2_endpoint = os.environ.get(\"R2_ENDPOINT_URL\", \"\")
    r2_bucket = os.environ.get(\"R2_BUCKET_NAME\", \"\")
    
    start_gen = 0
    if r2_endpoint and r2_bucket:
        print(\"Pulling latest checkpoints from R2 for Auto-Resume...\")
        try:
            pull_cmd = [\"aws\", \"s3\", \"sync\", f\"s3://{r2_bucket}/\", \"/tmp/checkpoints/\", \"--endpoint-url\", r2_endpoint]
            subprocess.run(pull_cmd, capture_output=True)
        except Exception as e:
            print(f\"Warning: Failed to pull from R2: {e}\")
            
    # Find latest checkpoint
    checkpoint_dirs = glob.glob(\"/tmp/checkpoints/qdax_rep_*\")
    epochs = [int(d.split(\"_\")[-1]) for d in checkpoint_dirs if d.split(\"_\")[-1].isdigit()]
    if epochs:
        latest_gen = max(epochs)
        start_gen = latest_gen + 1
        latest_ckpt = f\"/tmp/checkpoints/qdax_rep_{latest_gen}\"
        print(f\"Resuming from checkpoint {latest_ckpt} (Generation {latest_gen})!\")
        repertoire = checkpointer.restore(latest_ckpt, item=repertoire)
    else:
        # Fallback to latest just in case
        latest_ckpt = \"/tmp/checkpoints/qdax_rep_latest\"
        if os.path.exists(latest_ckpt):
            print(f\"Resuming from checkpoint {latest_ckpt}!\")
            repertoire = checkpointer.restore(latest_ckpt, item=repertoire)
        else:
            print(\"Starting MAP-Elites Evolutionary Loop from scratch!\")
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
        
        print(f\"[{i:04d}] Time: {elapsed:.2f}s | Max Fitness: {max_fitness:.2f} | Repertoire Coverage: {coverage:.1f}%\")
        
        # VRAM Monitoring
        try:
            vram_info = subprocess.getoutput('nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits,noheader').strip()
            if vram_info and 'not found' not in vram_info.lower() and 'failed' not in vram_info.lower():
                used, total = vram_info.split(', ')
                print(f\"  -> GPU VRAM Usage: {used} MB / {total} MB ({int(used)/int(total)*100:.1f}%)\")
        except Exception:
            pass
        
        # Historical Checkpoint Save
        if i % 10 == 0:
            save_path = f\"/tmp/checkpoints/qdax_rep_{i}\"
            checkpointer.save(save_path, repertoire, force=True)
            print(f\"  -> Checkpoint saved to {save_path}\")

if __name__ == \"__main__\":
    main()
"""

with open("train_new.py", "w") as f:
    f.write(new_train_py)

print("Generated train_new.py")
