import jax
import jax.numpy as jnp
from flax import nnx

class ReZero(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jnp.zeros(()))

    def __call__(self, x, f_x):
        return x + self.alpha[...] * f_x

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

# Diagonal of [60,60] planetary block zeroed; deep-space columns (60-63) always allowed.
# Shape [60, 64] — defined once, never changes across steps or games.
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
        self.rezero = ReZero(rngs=rngs)
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

        ca_out = self.cross_attn(inputs_q=p_emb, inputs_k=f_emb, inputs_v=f_emb, mask=attn_bias)
        p_emb = self.rezero(p_emb, ca_out)
        return self.rezero(p_emb, self.ffn(p_emb))


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
        self.rezero = ReZero(rngs=rngs)
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

        ca_out = self.cross_attn(inputs_q=p_emb, inputs_k=f_emb, inputs_v=f_emb, mask=attn_bias)
        p_emb = self.rezero(p_emb, ca_out)
        return self.rezero(p_emb, self.ffn(p_emb))


class PlanetSelfAttentionBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.rezero = ReZero(rngs=rngs)
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

        sa_out = self.self_attn(inputs_q=p_emb, mask=attn_bias)
        p_emb = self.rezero(p_emb, sa_out)
        return self.rezero(p_emb, self.ffn(p_emb))


class Actor(nnx.Module):
    """CA → SA×(n//2) → CA → SA×(n//2) interleaved architecture."""
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.ca_block_0 = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        self.ca_block_1 = PlanetMidCrossAttentionBlock(hidden_dim, rngs=rngs)

        self.q_action = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.k_action = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.deep_space_prob = nnx.Linear(hidden_dim, 4, rngs=rngs)
        # sin/cos output avoids the ±π wrap discontinuity; atan2 applied at decode time
        self.deep_space_sincos = nnx.Linear(hidden_dim, 8, rngs=rngs)
        # Per-planet temperature: softplus + 0.1 ensures T > 0.1
        self.temperature_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(self, planets, fleets, planet_mask=None, fleet_mask=None):
        p_emb    = self.ca_block_0(planets, fleets, fleet_mask=fleet_mask)
        p_coords = planets[..., 2:4]

        mid = self.num_sa_layers // 2
        for i in range(mid):
            p_emb = getattr(self, f'sa_block_{i}')(p_emb, p_coords, planet_mask=planet_mask)

        p_emb = self.ca_block_1(p_emb, fleets, p_coords, fleet_mask=fleet_mask)

        for i in range(mid, self.num_sa_layers):
            p_emb = getattr(self, f'sa_block_{i}')(p_emb, p_coords, planet_mask=planet_mask)

        planetary_logits = jnp.einsum('...id,...jd->...ij', self.q_action(p_emb), self.k_action(p_emb))

        # +1.0 exploration bias keeps deep space competitive against 60 planet logits
        ds_prob   = self.deep_space_prob(p_emb) + 1.0
        ds_sincos = jnp.tanh(self.deep_space_sincos(p_emb))

        T = jax.nn.softplus(self.temperature_head(p_emb)) + 0.1
        logits = jnp.concatenate([
            jnp.concatenate([planetary_logits, ds_prob], axis=-1) / T,
            ds_sincos,
        ], axis=-1)

        return logits

class Critic(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)

        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))

        self.action_logits_proj = nnx.Linear(64, hidden_dim, rngs=rngs)
        self.action_sincos_proj = nnx.Linear(8, hidden_dim, rngs=rngs)

        self.pa_proj = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

        self.global_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.rezero_global = ReZero(rngs=rngs)

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

        a_emb = self.action_logits_proj(jax.nn.softmax(actions[..., :64], axis=-1)) + self.action_sincos_proj(actions[..., 64:])
        pa_emb = jnp.concatenate([p_emb, a_emb], axis=-1)
        pa_proj = jax.nn.silu(self.pa_proj(pa_emb))

        sa_mask = None
        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]

        ga_out = self.global_attn(inputs_q=pa_proj, mask=sa_mask)
        ga_out = self.rezero_global(pa_proj, ga_out)

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

def logits_to_action(raw_actions, current_ships):
    """
    raw_actions: [B, 60, 72]  — 64 temperature-scaled softmax logits + 8 ds sin/cos
    current_ships: [B, 60]

    returns: (ships, angle)
    ships: [B, 60, 64] - ship allocations per source planet to 60 planet targets + 4 deep space bins
    angle: [B, 60, 4] - launch angle for the 4 deep space bins, decoded from sin/cos
    """
    logits = raw_actions[..., :64]
    probs  = jax.nn.softmax(logits, axis=-1)

    ships = probs * current_ships[..., None]
    ships = ships * _SELF_TARGET_MASK

    ds_sincos = raw_actions[..., 64:72]
    ds_angle  = jnp.arctan2(ds_sincos[..., :4], ds_sincos[..., 4:])

    return ships, ds_angle
