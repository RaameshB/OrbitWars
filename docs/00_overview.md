# OrbitWars: System Overview

OrbitWars is a research project for training autonomous AI agents to play a real-time strategy game using Quality-Diversity (QD) evolutionary algorithms combined with deep reinforcement learning.

## The Game in One Sentence

Players simultaneously command fleets launched from planets they own. They try to capture enemy planets and eliminate other players, on a rotationally-symmetric map that may include orbiting planets and passing comets.

## The Research Goal

We want a **diverse archive** of well-performing strategies, not just one champion. Different strategies — aggressive rushers, comet-campers, balanced expanders — should all be represented. This is what MAP-Elites (a Quality-Diversity algorithm) gives us.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE (train.py)                         │
│                                                                             │
│  ┌───────────────────────────────┐  ┌────────────────────────────────────┐  │
│  │  MAP-Elites Archive            │  │  NNXPgameEmitter (nnx_pgame.py)   │  │
│  │                               │  │                                    │  │
│  │  10,000 cells in 4D BD space  │  │  ┌──────────────────────────────┐ │  │
│  │  Each cell = one agent pytree │  │  │  ReplayBuffer (32k entries)  │ │  │
│  │  Fitness = end-of-game score  │  │  └──────────────────────────────┘ │  │
│  │                               │  │  ┌──────────────────────────────┐ │  │
│  │  HoF Archive [0..33]  ←────── │──│  │  TD3 Actor + DoubleCritic   │ │  │
│  │  HoF Exploiter [34..49]       │  │  │  (networks.py)              │ │  │
│  └───────────────────────────────┘  │  └──────────────────────────────┘ │  │
│            ↑ tell()                 └────────────────────────────────────┘  │
│            │                                       ↑ emit() + state_update()│
│  ┌─────────┴─────────────────────────────────────────────────────────────┐  │
│  │          self_play_scoring_fn   (50% 1v1 | 50% FFA)                  │  │
│  │                                                                       │  │
│  │  policy_params [B, ...]  ──vmap──▶  Actor network  ──▶  logits       │  │
│  │                                         │                             │  │
│  │  calculate_intercept_angle() ◀──────────┘                            │  │
│  │                                         │                             │  │
│  │  orbit_wars_jax.step() ─────────────────┘                            │  │
│  │       (500 steps via lax.scan)                                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  R2SyncThread  →  Cloudflare R2  →  GitHub Actions  →  GitHub Pages │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### MAP-Elites: An Archive, Not a Winner

Imagine a 4D filing cabinet where each drawer corresponds to a distinct *style* of play (behavioral descriptor). When an agent plays, we measure what style it demonstrated, and if it was the *best agent ever observed in that style*, it goes in that drawer and evicts whoever was there before.

After thousands of generations, the archive contains the best possible agent for each style — a diverse set of specialists.

### Behavioral Descriptors (4D)

Each agent's style is measured by averaging 4 metrics over the game:
- **BD0**: Fleet aggression ratio — ships in fleets vs. ships parked on planets
- **BD1**: Territory ratio — fraction of active planets you own
- **BD2**: Production efficiency — your production rate vs. your planet count
- **BD3**: Comet exploitation — comet production captured vs. total comet production

### Policy Gradient MAP-Elites (PGA-ME)

Standard MAP-Elites only uses random mutation. PGA-ME adds a TD3 (actor-critic) policy gradient agent:
1. The best agent in the archive is also the **TD3 actor**
2. The TD3 actor trains using gameplay experiences stored in a replay buffer
3. Every generation, the TD3 actor's policy is injected into the batch of candidates

This gives gradient-directed improvement on top of the random exploration.

### Hall of Fame (HoF)

A circular buffer of 50 past agents used as opponents:
- **Archive pool [0..33]**: snapshots of the top agent taken every 10 generations
- **Exploiter pool [34..49]**: agents that were fine-tuned specifically to beat the current archive

Keeping both prevents the exploiters from crowding out the archive history.

---

## File Map

| File | Role |
|------|------|
| `core/orbit_wars_jax.py` | Game engine — physics, rules, state |
| `core/networks.py` | Neural network architectures (Actor, DoubleCritic) |
| `core/nnx_pgame.py` | TD3 emitter, replay buffer, training loop |
| `core/rollout_utils.py` | Observation building, intercept angle calculation |
| `scripts/train.py` | Main training loop, MAP-Elites orchestration, multi-device sharding |
| `scripts/generate_site.py` | Pick the best agent from the archive and run a game for the website |
| `scripts/jax_visualizer.py` | Convert JAX game trajectory → Kaggle HTML replay |
| `scripts/migrate_r2_compression.py` | One-shot tool to compress old R2 checkpoints |
| `scripts/verify_behavior.py` | Sanity-check script to validate the diagonal mask and deep-space buckets |

---

## Data Flow for One Generation

```
1. emit()       Isoline-mutate archive samples  +  inject TD3 actor
                → genotypes [B, ...]

2. scoring_fn   Run 500-step games (vmap over B)
                → fitnesses [B], descriptors [B,4], transitions

3. tell()       Insert best agents into archive cells
                → updated repertoire

4. state_update() Insert transitions into replay buffer,
                  run 32 (× 8 = 256 total) TD3 gradient steps
                  → updated TD3 actor
```

## Hardware Targets

| Flag | `env_batch` | Notes |
|------|-------------|-------|
| `--hardware cpu` | 2 | Local debug only |
| `--hardware gpu` | 128 × N | N = number of GPUs detected |
| `--hardware tpu` | 128 × N | N = 8 on TPUv5e-8 |
| `--hardware blackwell` | 2048 | Single-device Blackwell GPU |
