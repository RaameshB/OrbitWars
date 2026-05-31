# OrbitWars Documentation

| Document | Contents |
|----------|----------|
| [00_overview.md](00_overview.md) | System architecture, key concepts (MAP-Elites, PGA-ME, HoF), data flow |
| [01_game_engine.md](01_game_engine.md) | `core/orbit_wars_jax.py` — board, planet generation, comet system, step() phases |
| [02_networks.md](02_networks.md) | `core/networks.py` — Actor, DoubleCritic, ReZero, Fourier features, action encoding |
| [03_td3_emitter.md](03_td3_emitter.md) | `core/nnx_pgame.py` — TD3 training loop, replay buffer, JIT boundary OOM fix |
| [04_rollout_utils.md](04_rollout_utils.md) | `core/rollout_utils.py` — egocentric observation, intercept angle computation |
| [05_training_pipeline.md](05_training_pipeline.md) | `scripts/train.py` — scoring function, dual-format self-play, HoF, sharding, checkpointing |
| [06_scripts.md](06_scripts.md) | Supporting scripts: site generation, R2 compression, behavioral verification |
| [07_jax_patterns.md](07_jax_patterns.md) | JAX gotchas: static vs. traced, lax.while_loop, sharding protocol, bfloat16 |

Start with `00_overview.md` for the big picture, then read the module docs in order.
