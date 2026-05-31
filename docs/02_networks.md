# Neural Network Architecture: `core/networks.py`

The agent's brain is a transformer-style architecture that reads the current game state and outputs launch orders for every planet. This document covers each component and the design decisions behind it.

---

## Observation Format

The network receives two arrays per timestep:

```
planets  [B, 60, 7]      — one row per planet/comet slot
fleets   [B, 1024, 6]    — one row per fleet slot
```

**Planet features** (7 per row):
```
[0] slot_id      float index 0..59
[1] rel_owner    +1 = mine, 0 = neutral, -1 = enemy
[2] x            rotated coordinate (I always see myself at angle 0)
[3] y            rotated coordinate
[4] radius       collision and capture radius
[5] ships        current garrison
[6] production   ships/turn this planet generates
```

**Fleet features** (6 per row):
```
[0] slot_id      float index 0..1023
[1] rel_owner    +1 = mine, 0 = (unused), -1 = enemy
[2] angle        heading angle (world frame, rotated to my perspective)
[3] x            position
[4] y            position
[5] ships        fleet size
```

**Coordinate rotation**: The observation builder rotates all coordinates by `-pid × 2π/num_players`. This means every player *always* sees themselves as if they were player 0 in the bottom-right quadrant. The network never needs to learn "I am player 2 who starts in the top-left" — it only needs to learn strategy from a consistent egocentric view.

---

## Feature Engineering

### Fourier Position Encoding

Raw coordinates (0–100) are passed through a multi-frequency sinusoidal encoding:

```python
norm_coords = (coords - 50) / 50      # normalize to [-1, 1]
freqs = exp(linspace(0, log(10), 4))  # 4 frequencies: [1.0, 2.15, 4.64, 10.0]
sin(norm_coords × freqs × π)          # 2 coords × 4 freqs = 8 sin features
cos(norm_coords × freqs × π)          # 8 cos features
```

Total: 2 (raw) + 8 + 8 = 18 position features per coordinate set.

**Why not just raw coords?** The network sees through linear layers whose weights are learned. A linear layer can barely express "is this planet close to mine?" from raw numbers. Fourier features let the network detect relative position at many scales simultaneously — from "is this roughly adjacent" to "is this on the exact opposite side of the board."

Think of it like giving the network a map with both a street grid and a neighborhood map overlaid — it can reason at multiple zoom levels.

### Relative Spatial Features

For attention biases, the engine computes pairwise features between any two sets of coordinates:

```python
delta = kv_coords - q_coords          # [Q, KV, 2] difference vectors
dist = sqrt(dx² + dy²) / 141.42       # normalized by board diagonal
theta = arctan2(dy, dx)
→ returns (norm_dist, sin_theta, cos_theta)  # [Q, KV, 3]
```

This is used to bias attention scores — a planet should attend more strongly to nearby fleets and less to distant ones, purely from geometry.

---

## ReZero

Every residual connection in the network is gated by a learned scalar `alpha`:

```python
output = x + alpha × f(x)   # alpha initialized to 0
```

At initialization, `alpha = 0` everywhere, so `output = x` — the new layers are pure identity functions. This means:
- The untrained network produces sensible (do-nothing) behavior
- Gradients don't vanish or explode in early training
- Each new layer "earns its place" as training progresses

Without ReZero, a 7-layer transformer starting from random weights often takes many epochs just to learn that "don't do anything" is a valid strategy, because the identity is not the network's natural initial state.

---

## PlanetCrossAttentionBlock

Planets attend to fleets to gather tactical information.

```
planets [B, 60, 7]   →   planet_mlp   →   p_emb [B, 60, D]
fleets  [B, 1024, 6] →   fleet_mlp    →   f_emb [B, 1024, D]

Relative spatial bias:
  get_relative_features(planet_coords, fleet_coords) → [B, 60, 1024, 3]
  rel_bias_mlp(relative_feats) × alpha              → [B, 2, 60, 1024]
  (2 heads; gated by its own ReZero alpha)

Cross-attention:
  Q = p_emb  (planets ask: "what fleets are relevant to me?")
  K = f_emb  (fleets offer their keys)
  V = f_emb  (fleets offer their values)
  mask = attn_bias + (-1e9 for inactive fleets)

p_emb = ReZero(p_emb, cross_attn(p_emb, f_emb, f_emb))
p_emb = ReZero(p_emb, ffn(p_emb))
```

Output shape: `[B, 60, D]` — each planet embedding is now enriched with fleet awareness.

---

## PlanetSelfAttentionBlock

Planets attend to each other to reason about territorial strategy.

```
p_emb [B, 60, D]      (from cross-attention above)
p_coords [B, 60, 2]

Relative spatial bias:
  get_relative_features(p_coords, p_coords) → [B, 60, 60, 3]
  rel_bias_mlp(relative_feats) × alpha      → [B, 2, 60, 60]

Self-attention:
  Q = K = V = p_emb
  mask = attn_bias + (-1e9 for inactive planet slots)

p_emb = ReZero(p_emb, self_attn(p_emb))
p_emb = ReZero(p_emb, ffn(p_emb))
```

The `Actor` stacks **6 of these blocks** after the cross-attention block. Six layers gives the network enough depth to reason about multi-hop strategic dependencies (e.g., "if I take that planet, my fleet from this one can reinforce the next one").

---

## Actor: Action Head

After 6 self-attention blocks, each planet has a rich embedding `p_emb [B, 60, D]`.

### Ship Allocation

Ships are allocated using a dot-product attention mechanism between planet embeddings:

```python
q = q_action(p_emb)   # [B, 60, D]
k = k_action(p_emb)   # [B, 60, D]
planetary_logits = einsum('bid,bjd->bij', q, k)   # [B, 60, 60]
```

This asks: "how attractive is planet j as a target for ships from planet i?" The score is purely learned from the embeddings — it's high-dimensional dot-product similarity.

Deep-space bins (free-angle launches not targeting any specific planet) have their own linear head:
```python
ds_prob = deep_space_prob(p_emb) + 1.0   # [B, 60, 4], bias +1 to compete
```

### Per-Planet Temperature

Each planet has its own temperature for softmax scaling:
```python
T = softplus(temperature_head(p_emb)) + 0.1   # [B, 60, 1], always > 0.1
```

**Why per-planet temperature?** Different situations call for different commitment levels. A planet under attack might benefit from decisive action (low T, sharp softmax). An unexplored backline planet might just want a small exploratory probe (high T, diffuse softmax). The network learns this from the game state rather than having a fixed global temperature.

### Sin/Cos Angle Output

For the 4 deep-space bins, the actor outputs sin and cos of the desired angle:

```python
ds_sincos = tanh(deep_space_sincos(p_emb))   # [B, 60, 8] → 4 sin + 4 cos
angle = arctan2(sin, cos)                     # decoded at action time
```

**Why sin/cos instead of a direct angle?** Direct angle output (e.g., `tanh(x) × π`) has a discontinuity at ±π — moving continuously from +π to -π requires the output to jump from +1 to -1. This discontinuity creates a gradient cliff that makes some launch directions nearly unreachable by gradient descent. Sin/cos is smooth everywhere; arctan2 is only applied at decode time and never backpropagated through.

### Unified Output Tensor

```
logits [B, 60, 72]:
  [:, :, 0:60]  — planet-to-planet ship allocation logits (temperature-divided)
  [:, :, 60:64] — deep-space bin logits (temperature-divided)
  [:, :, 64:72] — 4 sin + 4 cos for deep-space angles (tanh range, not temperature-divided)
```

---

## `logits_to_action()`: Logits to Ships

```python
probs = softmax(logits[:, :, :64])     # [B, 60, 64] — probability over 64 targets
ships = probs × current_ships          # fractional ship allocation per target
ships = ships × _SELF_TARGET_MASK      # zero out the diagonal (no self-sends)
```

**`_SELF_TARGET_MASK`**: A constant `[60, 64]` matrix with zeros on the diagonal of the 60×60 block and ones elsewhere. Multiplying by this mask makes "send ships to yourself" physically impossible — the ships simply disappear. This forces the network to actually do something; the bias toward self-targeting (structural +1.0 on diagonal logits) naturally encourages "conserve ships" rather than "waste ships".

---

## DoubleCritic

The critic estimates Q-values (expected future reward) given a state and an action.

**Why Double?** TD3 (Twin Delayed Deep Deterministic) uses two independent critics and takes the minimum of their predictions for target computation. This prevents overestimation bias — a single critic tends to over-predict Q-values, leading the actor to exploit those inflated estimates and get stuck in bad policies.

The critic architecture mirrors the actor:
1. `PlanetCrossAttentionBlock` (fleet awareness)
2. 4 × `PlanetSelfAttentionBlock` (territorial reasoning; 4 layers, not 6, for efficiency)
3. **Action injection**: the actor's logit output is projected into the planet embedding space and concatenated: `pa_emb = [p_emb | a_emb]`
4. Projected down to D: `pa_proj = silu(pa_proj(pa_emb))`
5. One global attention block to let planets "vote" on value
6. **Masked mean pooling**: average over active planets only → `global_emb [B, D]`
7. MLP → scalar Q-value

**Softmax in critic's action ingestion**:
```python
a_emb = action_logits_proj(softmax(actions[:64])) + action_sincos_proj(actions[64:])
```
The critic softmaxes the logit portion before projecting it. This normalizes away the actor's temperature drift — the critic cares about *which targets were chosen*, not *how confident* the actor was.

---

## Parameter Count (approximate)

With `hidden_dim=32`, `num_sa_layers=6`:
- Embedding MLPs: ~4k params each
- Attention heads (6 layers): ~24k each
- Relative bias MLPs: ~2k each
- Action/temperature heads: ~1k

**Total Actor**: ~200k parameters. Small enough for fast QD sweeps across 10,000 archive cells.
