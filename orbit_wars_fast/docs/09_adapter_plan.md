# RL Adapter (Implemented)

This capability is now implemented in:

- `orbit_wars_fast/rl_adapter.py`

## Included features

1. Observation flattening
   - `flatten_observations_jax(obs_jax)`
2. Action decoding
   - `decode_actions(observations, source_logits, angle, send_fraction)`
3. Episode rollout helper
   - `rollout_episode(simulator, policy_apply, params, rng_key, max_steps)`
4. Multi-seed evaluator
   - `evaluate_policy_seeds(make_simulator, policy_apply, params, seeds, base_key, max_steps)`

## Policy output contract

`policy_apply(params, obs_features, key)` should return a dictionary with keys:

- `source_logits`: `[num_agents, max_planets]`
- `angle`: `[num_agents]`
- `send_fraction`: `[num_agents]`

## Recommended backend

For parity-first training/evaluation:

- `JAXParityOrbitWarsSimulator`

For approximation-first speed pretraining:

- `jax_core.py`
