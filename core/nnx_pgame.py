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
        indices = (self.current_position + jnp.arange(batch_size)) % self.buffer_size
        # Sequential unique indices — deterministic, no scatter-add non-determinism risk
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
        return jax.tree_util.tree_map(lambda x: x[indices], self.data)

from flax import struct, nnx
from functools import partial
from typing import Tuple, Any

class NNXPgameConfig(struct.PyTreeNode):
    buffer_size: int = 200000
    env_batch_size: int = 2048
    pg_batch_size: int = 128
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
    pg_actor_params: Any
    target_actor_params: Any
    random_key: jax.Array
    steps: jnp.ndarray

class _TrainState(struct.PyTreeNode):
    """Compact training state passed through JIT — never includes the replay buffer.

    Keeping the replay buffer out of the JIT I/O boundary is the key OOM fix:
    jax.lax.cond (and even plain JIT output pytrees) force XLA to pre-allocate a
    separate output buffer for every leaf.  The actions field alone is
    buffer_size × 60 × 72 × 4 bytes ≈ 540 MB, which tips a tight TPU HBM budget
    over the edge.  By passing only the tiny model/opt state through JIT, and
    sampling batches eagerly before each call, that allocation is eliminated.
    """
    critic_params: Any
    target_critic_params: Any
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    pg_actor_params: Any
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
        num_mutants = self.batch_size - 1
        random_key, sample_key_1, sample_key_2 = jax.random.split(random_key, 3)
        parents1 = repertoire.select(sample_key_1, num_mutants).genotypes
        parents2 = repertoire.select(sample_key_2, num_mutants).genotypes
        mutants = isoline_variation(parents1, parents2, random_key,
                                    self._config.mutation_iso_sigma,
                                    self._config.mutation_line_sigma)
        pg_actor_expanded = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), emitter_state.pg_actor_params)
        combined_genotypes = jax.tree_util.tree_map(
            lambda m, p: jnp.concatenate([m, p], axis=0),
            mutants, pg_actor_expanded
        )
        return combined_genotypes, {"random_key": random_key}

    def state_update(self, emitter_state, repertoire, genotypes, fitnesses, descriptors, extra_scores):
        # Buffer insert runs eagerly (outside JIT) — no XLA binary cost
        transitions = extra_scores["transitions"]
        flat_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), transitions
        )
        replay_buffer = emitter_state.replay_buffer.insert(flat_transitions)
        emitter_state = NNXPgameEmitterState(
            replay_buffer=replay_buffer,
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            actor_opt_state=emitter_state.actor_opt_state,
            critic_opt_state=emitter_state.critic_opt_state,
            pg_actor_params=emitter_state.pg_actor_params,
            target_actor_params=emitter_state.target_actor_params,
            random_key=extra_scores.get("random_key", emitter_state.random_key),
            steps=emitter_state.steps,
        )
        # Python-level warm-up guard: skip gradient steps until the buffer has
        # enough data.  This avoids calling _jit_train_step at all during the
        # first few generations, which is when lax.cond would have OOM'd by
        # trying to pre-allocate output buffers for the full replay buffer.
        if int(replay_buffer.current_size) < self._config.learning_starts:
            return emitter_state
        # 32 gradient steps with the replay buffer sampled eagerly outside JIT.
        return self._run_n_train_steps(emitter_state, 32)

    def train_only(self, emitter_state):
        """32 more gradient steps without buffer insertion. Called 7× from train.py for 256 total."""
        if int(emitter_state.replay_buffer.current_size) < self._config.learning_starts:
            return emitter_state
        return self._run_n_train_steps(emitter_state, 32)

    def _run_n_train_steps(self, emitter_state: NNXPgameEmitterState, n: int) -> NNXPgameEmitterState:
        """Run *n* TD3 gradient steps.

        Batches are sampled EAGERLY before each JIT call so that the replay
        buffer never appears in the JIT I/O pytree.  This eliminates the
        ~540 MB XLA output-buffer allocation that caused the OOM when the
        buffer was passed through jax.lax.cond (or any JIT boundary).
        """
        rb = emitter_state.replay_buffer
        train_state = _TrainState(
            critic_params=emitter_state.critic_params,
            target_critic_params=emitter_state.target_critic_params,
            actor_opt_state=emitter_state.actor_opt_state,
            critic_opt_state=emitter_state.critic_opt_state,
            pg_actor_params=emitter_state.pg_actor_params,
            target_actor_params=emitter_state.target_actor_params,
            random_key=emitter_state.random_key,
            steps=emitter_state.steps,
        )
        for _ in range(n):
            # Split key outside JIT so the replay buffer sample is eager
            rng, sample_key = jax.random.split(train_state.random_key)
            batch = rb.sample(sample_key, self._config.pg_batch_size)
            # Pass remaining rng into JIT (used only for target-policy noise)
            train_state = self._jit_train_step(train_state.replace(random_key=rng), batch)
        return NNXPgameEmitterState(
            replay_buffer=rb,
            critic_params=train_state.critic_params,
            target_critic_params=train_state.target_critic_params,
            actor_opt_state=train_state.actor_opt_state,
            critic_opt_state=train_state.critic_opt_state,
            pg_actor_params=train_state.pg_actor_params,
            target_actor_params=train_state.target_actor_params,
            random_key=train_state.random_key,
            steps=train_state.steps,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _jit_train_step(self, train_state: _TrainState, batch: Transition) -> _TrainState:
        """Single TD3 gradient step on a pre-sampled *batch*.

        The replay buffer is NOT part of this function's input or output — it
        is sampled eagerly by the caller (_run_n_train_steps) and passed in as
        *batch*.  This means XLA only needs to allocate output buffers for the
        small model/opt state (~few MB), not for the large replay buffer.
        """
        c_params, tc_params = train_state.critic_params, train_state.target_critic_params
        a_opt, c_opt = train_state.actor_opt_state, train_state.critic_opt_state
        a_params, ta_params = train_state.pg_actor_params, train_state.target_actor_params
        rng, step = train_state.random_key, train_state.steps

        # Reconstruct attention masks from stored obs (radius > 0 = active planet; ships > 0 = active fleet)
        planet_mask = batch.obs[0][..., 4] > 0.0
        fleet_mask  = batch.obs[1][..., 5] > 0.0
        next_planet_mask = batch.next_obs[0][..., 4] > 0.0
        next_fleet_mask  = batch.next_obs[1][..., 5] > 0.0

        # Critic update — compute Bellman targets with frozen target networks
        target_actor = nnx.merge(self._actor_graph, ta_params)
        target_critic = nnx.merge(self._critic_graph, tc_params)
        next_logits = target_actor(batch.next_obs[0], batch.next_obs[1],
                                   planet_mask=next_planet_mask, fleet_mask=next_fleet_mask)
        # Split noise: larger for softmax logit dims, smaller for bounded sin/cos dims
        rng, k1, k2 = jax.random.split(rng, 3)
        logit_noise  = jnp.clip(jax.random.normal(k1, next_logits[..., :64].shape) * 0.2, -0.5,  0.5)
        sincos_noise = jnp.clip(jax.random.normal(k2, next_logits[..., 64:].shape) * 0.05, -0.1, 0.1)
        noise = jnp.concatenate([logit_noise, sincos_noise], axis=-1)
        target_q1, target_q2 = target_critic(batch.next_obs[0], batch.next_obs[1], next_logits + noise,
                                              planet_mask=next_planet_mask, fleet_mask=next_fleet_mask)
        y = batch.rewards + self._config.gamma * jnp.minimum(target_q1, target_q2) * (1.0 - batch.dones)

        def critic_loss_fn(c_p):
            c = nnx.merge(self._critic_graph, c_p)
            q1, q2 = c(batch.obs[0], batch.obs[1], batch.actions,
                        planet_mask=planet_mask, fleet_mask=fleet_mask)
            return jnp.mean((q1 - y) ** 2 + (q2 - y) ** 2)

        c_grads = jax.grad(critic_loss_fn)(c_params)
        c_updates, c_opt = self._critic_optimizer.update(c_grads, c_opt, c_params)
        c_params = optax.apply_updates(c_params, c_updates)

        # Actor update every step (no policy delay — eliminates lax.cond branch from XLA graph)
        def actor_loss_fn(a_p):
            a = nnx.merge(self._actor_graph, a_p)
            c = nnx.merge(self._critic_graph, c_params)
            logits = a(batch.obs[0], batch.obs[1], planet_mask=planet_mask, fleet_mask=fleet_mask)
            q1, _ = c(batch.obs[0], batch.obs[1], logits, planet_mask=planet_mask, fleet_mask=fleet_mask)
            return -jnp.mean(q1)

        a_grads = jax.grad(actor_loss_fn)(a_params)
        a_updates, a_opt = self._actor_optimizer.update(a_grads, a_opt, a_params)
        a_params = optax.apply_updates(a_params, a_updates)

        # Polyak averaging
        tau = self._config.tau
        ta_params = jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, ta_params, a_params)
        tc_params = jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, tc_params, c_params)

        return _TrainState(
            critic_params=c_params, target_critic_params=tc_params,
            actor_opt_state=a_opt, critic_opt_state=c_opt,
            pg_actor_params=a_params, target_actor_params=ta_params,
            random_key=rng, steps=step + 1,
        )
