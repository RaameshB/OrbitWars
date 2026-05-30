import jax
import jax.numpy as jnp
import optax
from qdax.core.emitters.emitter import Emitter
from qdax.core.emitters.mutation_operators import isoline_variation
from flax import struct
from typing import Tuple, Any

class Transition(struct.PyTreeNode):
    obs: Any
    next_obs: Any
    rewards: jnp.ndarray
    dones: jnp.ndarray
    actions: jnp.ndarray
    
class ReplayBuffer(struct.PyTreeNode):
    data: Transition
    current_position: jnp.ndarray
    current_size: jnp.ndarray
    buffer_size: int = struct.field(pytree_node=False)

    @classmethod
    def init(cls, buffer_size: int, dummy_transition: Transition):
        # Create buffer arrays by adding the buffer_size dimension
        data = jax.tree_util.tree_map(
            lambda x: jnp.zeros((buffer_size,) + x.shape, dtype=x.dtype),
            dummy_transition
        )
        return cls(
            data=data,
            current_position=jnp.array(0, dtype=jnp.int32),
            current_size=jnp.array(0, dtype=jnp.int32),
            buffer_size=buffer_size
        )

    def insert(self, transitions: Transition):
        batch_size = transitions.rewards.shape[0]
        
        # We assume batch_size < buffer_size for simplicity.
        # Actually, if batch_size > buffer_size, we should wrap or just slice.
        
        # For simplicity in this env, we just do a dynamic update slice
        # But dynamic_update_slice doesn't wrap around automatically.
        # Let's do a simple indices approach
        indices = (self.current_position + jnp.arange(batch_size)) % self.buffer_size
        
        new_data = jax.tree_util.tree_map(
            lambda buf, trans: buf.at[indices].set(trans),
            self.data, transitions
        )
        
        return self.replace(
            data=new_data,
            current_position=(self.current_position + batch_size) % self.buffer_size,
            current_size=jnp.minimum(self.buffer_size, self.current_size + batch_size)
        )

    def sample(self, random_key, sample_size: int):
        indices = jax.random.randint(random_key, shape=(sample_size,), minval=0, maxval=self.current_size)
        return jax.tree_util.tree_map(
            lambda x: x[indices],
            self.data
        )

from flax import struct, nnx
from functools import partial
from typing import Tuple, Any

class NNXPgameConfig(struct.PyTreeNode):
    buffer_size: int = 1000000
    env_batch_size: int = 2048
    pg_batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    learning_starts: int = 1000
    mutation_iso_sigma: float = 0.01
    mutation_line_sigma: float = 0.1

class NNXPgameEmitterState(struct.PyTreeNode):
    replay_buffer: ReplayBuffer
    critic_params: Any
    target_critic_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    pg_actor_params: Any # The actor actively trained by Policy Gradients
    target_actor_params: Any
    random_key: jax.Array
    steps: jnp.ndarray

class NNXPgameEmitter(Emitter):
    def __init__(
        self,
        config: NNXPgameConfig,
        actor_graph: Any,
        critic_graph: Any,
        dummy_transition: Transition,
        init_critic_params: Any = None
    ):
        self._config = config
        self._actor_graph = actor_graph
        self._critic_graph = critic_graph
        self._actor_optimizer = optax.adam(learning_rate=config.actor_lr)
        self._critic_optimizer = optax.adam(learning_rate=config.critic_lr)
        self._init_critic_params = init_critic_params
        self._dummy_transition = dummy_transition

    @property
    def batch_size(self) -> int:
        return self._config.env_batch_size

    @property
    def use_all_data(self) -> bool:
        return False
        
    def init(self, key, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        # We assume genotypes is a PyTree batch. Pick the first one for our PG actor.
        pg_actor_params = jax.tree_util.tree_map(lambda x: x[0], genotypes)
        target_actor_params = jax.tree_util.tree_map(lambda x: x[0], genotypes)
        
        target_critic_params = self._init_critic_params
        
        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.buffer_size,
            dummy_transition=self._dummy_transition
        )
        
        actor_opt_state = self._actor_optimizer.init(pg_actor_params)
        critic_opt_state = self._critic_optimizer.init(self._init_critic_params)
        
        return NNXPgameEmitterState(
            replay_buffer=replay_buffer,
            critic_params=self._init_critic_params,
            target_critic_params=target_critic_params,
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
            pg_actor_params=pg_actor_params,
            target_actor_params=target_actor_params,
            random_key=key,
            steps=jnp.array(0)
        )

    @partial(jax.jit, static_argnames=("self",))
    def emit(self, repertoire, emitter_state, random_key):
        # 1. Sample parents for mutation
        num_mutants = self.batch_size - 1
        random_key, sample_key_1, sample_key_2 = jax.random.split(random_key, 3)
        parents1 = repertoire.select(sample_key_1, num_mutants).genotypes
        parents2 = repertoire.select(sample_key_2, num_mutants).genotypes
        
        # 2. Apply IsoLine Variation
        mutants = isoline_variation(parents1, parents2, random_key, 
                                                self._config.mutation_iso_sigma, 
                                                self._config.mutation_line_sigma)
        
        # 3. Inject the PG-Actor
        # Combine mutants [B-1] with PG-Actor [1] to form batch [B]
        pg_actor_expanded = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), emitter_state.pg_actor_params)
        combined_genotypes = jax.tree_util.tree_map(
            lambda m, p: jnp.concatenate([m, p], axis=0), 
            mutants, pg_actor_expanded
        )
        
        return combined_genotypes, {"random_key": random_key}

    @partial(jax.jit, static_argnames=("self",))
    def state_update(self, emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        # Insert transitions into replay buffer
        transitions = extra_scores["transitions"]
        # Flatten time and batch dimensions: [500, 2048, ...] -> [1024000, ...]
        flat_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), transitions
        )
        
        replay_buffer = emitter_state.replay_buffer.insert(flat_transitions)
        
        # Determine if we should train
        should_train = replay_buffer.current_size >= self._config.learning_starts
        
        # Function to perform TD3 training step
        def train_step(state):
            rb, c_params, tc_params, a_opt, c_opt, a_params, ta_params, rng, step = state
            
            # Sample batch
            rng, sample_key = jax.random.split(rng)
            batch = rb.sample(sample_key, sample_size=self._config.pg_batch_size)
            
            # Reconstruct modules
            actor = nnx.merge(self._actor_graph, a_params)
            target_actor = nnx.merge(self._actor_graph, ta_params)
            critic = nnx.merge(self._critic_graph, c_params)
            target_critic = nnx.merge(self._critic_graph, tc_params)
            
            # Critic Update
            def critic_loss_fn(c_p):
                c = nnx.merge(self._critic_graph, c_p)
                
                # Get next actions from target actor
                # Wait, our actor takes planets and fleets!
                # Transition batch.obs is EnvState.
                # In orbit_wars, EnvState is not directly feedable to actor, we pass planets and fleets
                # Actually, if we store planets and fleets directly in obs, it's easier.
                next_logits = target_actor(batch.next_obs[0], batch.next_obs[1])
                
                # Add target policy noise (clipped)
                # Next logits are logits! Should we add noise to logits?
                # Usually TD3 adds noise to actions. Since we feed logits to Critic, we can add noise to logits.
                noise = jnp.clip(
                    jax.random.normal(sample_key, next_logits.shape) * 0.2,
                    -0.5, 0.5
                )
                noisy_next_logits = next_logits + noise
                
                target_q1, target_q2 = target_critic(batch.next_obs[0], batch.next_obs[1], noisy_next_logits)
                target_q = jnp.minimum(target_q1, target_q2)
                
                # y = r + gamma * target_q * (1 - done)
                y = batch.rewards + self._config.gamma * target_q * (1.0 - batch.dones)
                
                q1, q2 = c(batch.obs[0], batch.obs[1], batch.actions)
                loss = jnp.mean((q1 - y)**2 + (q2 - y)**2)
                return loss
            
            c_loss, c_grads = jax.value_and_grad(critic_loss_fn)(c_params)
            c_updates, c_opt = self._critic_optimizer.update(c_grads, c_opt, c_params)
            c_params = optax.apply_updates(c_params, c_updates)
            
            # Actor Update (Delayed)
            def actor_update_fn(a_p, a_o, rng, c_p):
                def actor_loss_fn(p):
                    a = nnx.merge(self._actor_graph, p)
                    c = nnx.merge(self._critic_graph, c_p)
                    # Output of actor is fed into critic
                    logits = a(batch.obs[0], batch.obs[1])
                    q1, q2 = c(batch.obs[0], batch.obs[1], logits)
                    return -jnp.mean(q1)
                    
                a_loss, a_grads = jax.value_and_grad(actor_loss_fn)(a_p)
                a_updates, a_o = self._actor_optimizer.update(a_grads, a_o, a_p)
                new_a_p = optax.apply_updates(a_p, a_updates)
                return new_a_p, a_o
                
            do_actor_update = (step % self._config.policy_delay == 0)
            a_params, a_opt = jax.lax.cond(
                do_actor_update,
                lambda args: actor_update_fn(*args),
                lambda args: (args[0], args[1]),
                (a_params, a_opt, rng, c_params)
            )
            
            # Polyak Averaging
            def polyak_update(target, current):
                return jax.tree_util.tree_map(
                    lambda t, c: (1.0 - self._config.tau) * t + self._config.tau * c,
                    target, current
                )
                
            ta_params, tc_params = jax.lax.cond(
                do_actor_update,
                lambda _: (polyak_update(ta_params, a_params), polyak_update(tc_params, c_params)),
                lambda _: (ta_params, tc_params),
                None
            )
            
            return rb, c_params, tc_params, a_opt, c_opt, a_params, ta_params, rng, step + 1

        # Train loop for this generation:
        # MAP-Elites usually does ONE epoch per generation?
        # Actually, if we collect 500 * 2048 = 1M transitions per generation, doing ONE TD3 step is completely useless.
        # We should do e.g. 256 gradient steps per generation!
        
        def run_training_steps(state):
            return jax.lax.fori_loop(0, 256, lambda i, s: train_step(s), state)
            
        new_random_key = extra_scores.get("random_key", emitter_state.random_key)
        
        current_train_state = (
            replay_buffer, emitter_state.critic_params, emitter_state.target_critic_params,
            emitter_state.actor_opt_state, emitter_state.critic_opt_state, 
            emitter_state.pg_actor_params, emitter_state.target_actor_params, 
            new_random_key, emitter_state.steps
        )
        
        new_train_state = jax.lax.cond(
            should_train,
            run_training_steps,
            lambda s: s,
            current_train_state
        )
        
        rb, c_params, tc_params, a_opt, c_opt, a_params, ta_params, rng, step = new_train_state
        
        return NNXPgameEmitterState(
            replay_buffer=rb,
            critic_params=c_params,
            target_critic_params=tc_params,
            actor_opt_state=a_opt,
            critic_opt_state=c_opt,
            pg_actor_params=a_params,
            target_actor_params=ta_params,
            random_key=rng,
            steps=step
        )
