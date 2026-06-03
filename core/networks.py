import jax
import jax.numpy as jnp
from flax import nnx

def get_fourier_features(coords, num_bands=4):
    # coords: [..., 2], board is 100×100 with CENTER=50
    norm_coords = (coords - 50.0) / 50.0

    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10.0), num_bands))
    args = norm_coords[..., None] * freqs[None, :] * jnp.pi

    sin_feats = jnp.sin(args).reshape(coords.shape[:-1] + (-1,))
    cos_feats = jnp.cos(args).reshape(coords.shape[:-1] + (-1,))

    return jnp.concatenate([norm_coords, sin_feats, cos_feats], axis=-1)

def get_relative_features(q_coords, kv_coords):
    # q_coords: [..., Q, 2], kv_coords: [..., KV, 2]
    # Returns [..., Q, KV, 3] -> (dist, sin_theta, cos_theta)

    # delta: [..., Q, KV, 2]
    delta = kv_coords[..., None, :, :] - q_coords[..., :, None, :]
    dx = delta[..., 0]
    dy = delta[..., 1]

    dist = jnp.sqrt(dx**2 + dy**2 + 1e-8)
    # Normalize dist by board diagonal (sqrt(2)*100 ≈ 141.4)
    norm_dist = dist / 141.42

    theta = jnp.arctan2(dy, dx)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    return jnp.stack([norm_dist, sin_theta, cos_theta], axis=-1)

# Destination mask [60, 64]: diagonal zeroed (no self-send), DS columns (60-63) always allowed.
_SELF_TARGET_MASK = jnp.concatenate([1.0 - jnp.eye(60), jnp.ones((60, 4))], axis=-1)

class PlanetCrossAttentionBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # 7 base features + 2*4*2 = 16 fourier + 2 norm_coords = 25 features
        self.planet_mlp = nnx.Sequential(
            nnx.Linear(7 + 18, hidden_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )

        # 6 base features + 18 fourier = 24 features
        self.fleet_mlp = nnx.Sequential(
            nnx.Linear(6 + 18, hidden_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )

        # Relative Bias MLP [3 -> 16 -> num_heads]
        self.rel_bias_mlp = nnx.Sequential(
            nnx.Linear(3, 16, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(16, 2, rngs=rngs)
        )

        # Cross Attention: Planets attend to Fleets
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        )

    def __call__(self, planets, fleets, fleet_mask=None):
        p_coords = planets[..., 2:4]   # [rot_x, rot_y]
        f_coords = fleets[..., 3:5]    # [rot_x, rot_y] — NOT index 2:4 which is [angle, rot_x]

        p_fourier = get_fourier_features(p_coords)
        f_fourier = get_fourier_features(f_coords)

        p_in = jnp.concatenate([planets, p_fourier], axis=-1)
        f_in = jnp.concatenate([fleets, f_fourier], axis=-1)

        p_emb = self.planet_mlp(p_in)  # [B, 60, D]
        f_emb = self.fleet_mlp(f_in)   # [B, N, D]

        rel_feats  = get_relative_features(p_coords, f_coords)   # [..., 60, N, 3]
        bias_logits = self.rel_bias_mlp(rel_feats)                # [..., 60, N, 2]
        attn_bias   = jnp.moveaxis(bias_logits, -1, -3)          # [..., 2, 60, N]

        if fleet_mask is not None:
            attn_bias = jnp.where(fleet_mask[..., None, None, :], attn_bias, -1e9)

        ca_out = self.cross_attn(inputs_q=self.norm1(p_emb), inputs_k=f_emb, inputs_v=f_emb, mask=attn_bias)
        p_emb = p_emb + ca_out
        return p_emb + self.ffn(self.norm2(p_emb))


class PlanetMidCrossAttentionBlock(nnx.Module):
    """Mid-network cross-attention: takes pre-embedded planet features, re-attends to fleets."""
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.fleet_mlp = nnx.Sequential(
            nnx.Linear(6 + 18, hidden_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )
        self.rel_bias_mlp = nnx.Sequential(
            nnx.Linear(3, 16, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(16, 2, rngs=rngs)
        )
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        )

    def __call__(self, p_emb, fleets, p_coords, fleet_mask=None):
        f_coords = fleets[..., 3:5]   # [rot_x, rot_y]
        f_fourier = get_fourier_features(f_coords)
        f_in  = jnp.concatenate([fleets, f_fourier], axis=-1)
        f_emb = self.fleet_mlp(f_in)

        rel_feats  = get_relative_features(p_coords, f_coords)
        bias_logits = self.rel_bias_mlp(rel_feats)
        attn_bias   = jnp.moveaxis(bias_logits, -1, -3)

        if fleet_mask is not None:
            attn_bias = jnp.where(fleet_mask[..., None, None, :], attn_bias, -1e9)

        ca_out = self.cross_attn(inputs_q=self.norm1(p_emb), inputs_k=f_emb, inputs_v=f_emb, mask=attn_bias)
        p_emb = p_emb + ca_out
        return p_emb + self.ffn(self.norm2(p_emb))


class PlanetSelfAttentionBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self._head_dim = hidden_dim // 2
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.norm1  = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2  = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.q_norm = nnx.LayerNorm(self._head_dim, rngs=rngs)
        self.k_norm = nnx.LayerNorm(self._head_dim, rngs=rngs)
        self.rel_bias_mlp = nnx.Sequential(
            nnx.Linear(3, 16, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(16, 2, rngs=rngs)
        )
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        )

    def __call__(self, p_emb, p_coords, planet_mask=None):
        rel_feats   = get_relative_features(p_coords, p_coords)   # [..., 60, 60, 3]
        bias_logits = self.rel_bias_mlp(rel_feats)                 # [..., 60, 60, 2]
        attn_bias   = jnp.moveaxis(bias_logits, -1, -3)           # [..., 2, 60, 60]

        if planet_mask is not None:
            attn_bias = jnp.where(planet_mask[..., None, None, :], attn_bias, -1e9)

        x = self.norm1(p_emb)
        q = self.q_norm(self.self_attn.query(x))   # [..., 60, 2, 24]
        k = self.k_norm(self.self_attn.key(x))     # [..., 60, 2, 24]
        v = self.self_attn.value(x)                # [..., 60, 2, 24]

        q = jnp.moveaxis(q, -2, -3)               # [..., 2, 60, 24]
        k = jnp.moveaxis(k, -2, -3)
        v = jnp.moveaxis(v, -2, -3)

        scale       = self._head_dim ** -0.5
        attn_logits = jnp.einsum('...hid,...hjd->...hij', q, k) * scale + attn_bias
        attn_out    = jnp.einsum('...hij,...hjd->...hid',
                                 jax.nn.softmax(attn_logits, axis=-1), v)  # [..., 2, 60, 24]
        sa_out = self.self_attn.out(jnp.moveaxis(attn_out, -3, -2))        # [..., 60, 48]

        p_emb = p_emb + sa_out
        return p_emb + self.ffn(self.norm2(p_emb))


class Actor(nnx.Module):
    """CA → SA×(n//2) → CA → SA×(n//2) interleaved architecture."""
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.ca_block_0 = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        self.ca_block_1 = PlanetMidCrossAttentionBlock(hidden_dim, rngs=rngs)

        self.final_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        # Decomposed action head:
        #   hold_head  → sigmoid → fraction of ships to hold (trained via MSE)
        #   dest_q/k   → dot-product logits over 60 planets (self masked) + 4 DS bins
        #   ds_sincos  → sin/cos angle per DS bin; atan2 applied at decode time
        self.hold_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.dest_q    = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.dest_k    = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.ds_prob   = nnx.Linear(hidden_dim, 4, rngs=rngs)
        self.ds_sincos = nnx.Linear(hidden_dim, 8, rngs=rngs)

    def __call__(self, planets, fleets, planet_mask=None, fleet_mask=None):
        p_emb    = self.ca_block_0(planets, fleets, fleet_mask=fleet_mask)
        p_coords = planets[..., 2:4]

        mid = self.num_sa_layers // 2
        for i in range(mid):
            p_emb = getattr(self, f'sa_block_{i}')(p_emb, p_coords, planet_mask=planet_mask)

        p_emb = self.ca_block_1(p_emb, fleets, p_coords, fleet_mask=fleet_mask)

        for i in range(mid, self.num_sa_layers):
            p_emb = getattr(self, f'sa_block_{i}')(p_emb, p_coords, planet_mask=planet_mask)

        p_emb = self.final_norm(p_emb)

        hold_logit = self.hold_head(p_emb)                                          # [B, 60, 1]
        planet_logits = jnp.einsum('...id,...jd->...ij', self.dest_q(p_emb), self.dest_k(p_emb))  # [B, 60, 60]
        # +1.0 exploration bias keeps DS competitive against 60 planet logits
        ds_prob   = self.ds_prob(p_emb) + 1.0                                       # [B, 60, 4]
        ds_sincos = jnp.tanh(self.ds_sincos(p_emb))                                 # [B, 60, 8]

        # Output layout per planet: [hold(1) | dest_logits(64) | ds_sincos(8)] = 73
        return jnp.concatenate([
            hold_logit,
            jnp.concatenate([planet_logits, ds_prob], axis=-1),
            ds_sincos,
        ], axis=-1)

class Critic(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)

        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))

        self.action_hold_proj   = nnx.Linear(1,  hidden_dim, rngs=rngs)
        self.action_logits_proj = nnx.Linear(64, hidden_dim, rngs=rngs)
        self.action_sincos_proj = nnx.Linear(8,  hidden_dim, rngs=rngs)

        self.pa_proj = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

        self.global_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.norm_global = nnx.LayerNorm(hidden_dim, rngs=rngs)

        self.value_head = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, 1, rngs=rngs)
        )

    def __call__(self, planets, fleets, actions, planet_mask=None, fleet_mask=None):
        p_emb = self.cross_attention_block(planets, fleets, fleet_mask=fleet_mask)

        p_coords = planets[..., 2:4]
        for i in range(self.num_sa_layers):
            sa = getattr(self, f'sa_block_{i}')
            p_emb = sa(p_emb, p_coords, planet_mask=planet_mask)

        # actions layout: [hold(1) | dest_logits(64) | ds_sincos(8)] = 73
        a_emb = (self.action_hold_proj(jax.nn.sigmoid(actions[..., :1]))
               + self.action_logits_proj(jax.nn.softmax(actions[..., 1:65], axis=-1))
               + self.action_sincos_proj(actions[..., 65:73]))
        pa_emb = jnp.concatenate([p_emb, a_emb], axis=-1)
        pa_proj = jax.nn.silu(self.pa_proj(pa_emb))

        sa_mask = None
        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]

        ga_out = self.global_attn(inputs_q=self.norm_global(pa_proj), mask=sa_mask)
        ga_out = pa_proj + ga_out

        if planet_mask is not None:
            ga_out = jnp.where(planet_mask[..., None], ga_out, 0.0)
            valid_counts = jnp.maximum(jnp.sum(planet_mask, axis=-1, keepdims=True), 1.0)
            global_emb = jnp.sum(ga_out, axis=-2) / valid_counts
        else:
            global_emb = jnp.mean(ga_out, axis=-2)

        return self.value_head(global_emb)

class DoubleCritic(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.q1 = Critic(hidden_dim, num_sa_layers, rngs=rngs)
        self.q2 = Critic(hidden_dim, num_sa_layers, rngs=rngs)

    def __call__(self, planets, fleets, actions, planet_mask=None, fleet_mask=None):
        v1 = self.q1(planets, fleets, actions, planet_mask=planet_mask, fleet_mask=fleet_mask)
        v2 = self.q2(planets, fleets, actions, planet_mask=planet_mask, fleet_mask=fleet_mask)
        return v1, v2

def logits_to_action(raw_actions, current_ships, mask_self=True):
    """
    raw_actions: [B, 60, 73]
      [0]      hold logit   → sigmoid → hold fraction
      [1:65]   dest logits  → softmax over 60 planets + 4 DS bins (self masked)
      [65:73]  ds sincos    → atan2 → launch angle for 4 DS bins

    returns: (ships, ds_angle)
      ships:    [B, 60, 64]  ships allocated per destination (60 planets + 4 DS)
      ds_angle: [B, 60, 4]   launch angle for each DS bin
    """
    hold_frac = jax.nn.sigmoid(raw_actions[..., 0])          # [B, 60]
    dest_logits = raw_actions[..., 1:65]                      # [B, 60, 64]
    if mask_self:
        dest_logits = dest_logits * _SELF_TARGET_MASK
    dest_probs = jax.nn.softmax(dest_logits, axis=-1)         # [B, 60, 64]

    launch_ships = (1.0 - hold_frac) * current_ships          # [B, 60]
    ships = dest_probs * launch_ships[..., None]              # [B, 60, 64]

    ds_sincos = raw_actions[..., 65:73]
    ds_angle  = jnp.arctan2(ds_sincos[..., :4], ds_sincos[..., 4:])

    return ships, ds_angle
