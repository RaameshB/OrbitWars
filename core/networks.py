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
        
        # Relative Bias MLP [3 -> 16 -> num_heads], gated by ReZero scalar
        self.rel_bias_mlp = nnx.Sequential(
            nnx.Linear(3, 16, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(16, 2, rngs=rngs)
        )
        self.rel_bias_alpha = nnx.Param(jnp.zeros(()))

        # Cross Attention: Planets attend to Fleets
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.rezero_ca = ReZero(rngs=rngs)
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        )
        self.rezero_ffn = ReZero(rngs=rngs)

    def __call__(self, planets, fleets, fleet_mask=None):
        """
        planets: [B, 60, 7]
        fleets: [B, 1000, 6]
        """
        # Absolute Fourier Features
        p_coords = planets[..., 2:4] # Assuming 2:4 are y, x or x, y
        f_coords = fleets[..., 2:4]  # Assuming 2:4 are x, y
        
        p_fourier = get_fourier_features(p_coords)
        f_fourier = get_fourier_features(f_coords)
        
        p_in = jnp.concatenate([planets, p_fourier], axis=-1)
        f_in = jnp.concatenate([fleets, f_fourier], axis=-1)
        
        p_emb = self.planet_mlp(p_in) # [B, 60, D]
        f_emb = self.fleet_mlp(f_in)   # [B, 1000, D]
        
        ca_mask = None
        if fleet_mask is not None:
            ca_mask = fleet_mask[..., None, None, :]
            
        # Relative Bias
        rel_feats = get_relative_features(p_coords, f_coords) # [..., Q, KV, 3]
        bias_logits = self.rel_bias_alpha[...] * self.rel_bias_mlp(rel_feats) # [..., Q, KV, num_heads]
        bias_logits = jnp.moveaxis(bias_logits, -1, -3) # [..., num_heads, Q, KV]
        
        # We need to manually add the bias to the attention matrix if using Flax nnx
        # But wait! nnx.MultiHeadAttention doesn't natively accept a spatial bias tensor easily unless passed as a mask or subclassed.
        # Actually, mask can be a float tensor added to the logits!
        # If ca_mask is boolean, we convert it to logits bias (-1e9) and add our spatial bias.
        
        attn_bias = bias_logits
        if ca_mask is not None:
            # Add -1e9 where mask is False
            attn_bias = jnp.where(ca_mask, attn_bias, -1e9)
            
        # Passing float tensor to mask acts as an attention bias in modern Flax!
        ca_out = self.cross_attn(inputs_q=p_emb, inputs_k=f_emb, inputs_v=f_emb, mask=attn_bias)
        p_emb = self.rezero_ca(p_emb, ca_out)
        return self.rezero_ffn(p_emb, self.ffn(p_emb))

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
        self.rezero_sa = ReZero(rngs=rngs)
        self.rel_bias_mlp = nnx.Sequential(
            nnx.Linear(3, 16, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(16, 2, rngs=rngs)
        )
        self.rel_bias_alpha = nnx.Param(jnp.zeros(()))
        self.ffn = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim * 2, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)
        )
        self.rezero_ffn = ReZero(rngs=rngs)

    def __call__(self, p_emb, p_coords, planet_mask=None):

        rel_feats = get_relative_features(p_coords, p_coords) # [..., 60, 60, 3]
        bias_logits = self.rel_bias_alpha[...] * self.rel_bias_mlp(rel_feats) # [..., 60, 60, num_heads]
        attn_bias = jnp.moveaxis(bias_logits, -1, -3) # [..., num_heads, 60, 60]

        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]
            attn_bias = jnp.where(sa_mask, attn_bias, -1e9)

        sa_out = self.self_attn(inputs_q=p_emb, mask=attn_bias)
        p_emb = self.rezero_sa(p_emb, sa_out)
        return self.rezero_ffn(p_emb, self.ffn(p_emb))

class Actor(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        
        # Shared Action Head MLP applied per-planet
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        
        self.q_action = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.k_action = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.deep_space_prob = nnx.Linear(hidden_dim, 4, rngs=rngs)
        # sin/cos output avoids the ±π wrap discontinuity; atan2 is only applied at action decode time
        self.deep_space_sincos = nnx.Linear(hidden_dim, 8, rngs=rngs)
        # Per-planet temperature head: each planet's embedding decides how decisive its launches are.
        # softplus + 0.1 ensures T > 0.1, init near 0.79 (≈T=1). Different games → different planet
        # states → different temperatures. This is a network output, not a fixed per-slot parameter.
        self.temperature_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        
    def __call__(self, planets, fleets, planet_mask=None, fleet_mask=None):
        # [B, 60, D]
        p_emb = self.cross_attention_block(planets, fleets, fleet_mask=fleet_mask)
        
        p_coords = planets[..., 2:4]
        for i in range(self.num_sa_layers):
            sa = getattr(self, f'sa_block_{i}')
            p_emb = sa(p_emb, p_coords, planet_mask=planet_mask)
        
        planetary_logits = jnp.einsum('...id,...jd->...ij', self.q_action(p_emb), self.k_action(p_emb))  # [..., 60, 60]

        # Mild structural self-targeting bias; learned temperature handles global sharpness
        planetary_logits = planetary_logits + jnp.eye(60) * 1.0

        # +1.0 exploration bias keeps deep space competitive against 60 planet logits
        ds_prob = self.deep_space_prob(p_emb) + 1.0  # [B, 60, 4]

        # sin/cos angle output: smooth gradients everywhere, atan2 applied at decode time
        ds_sincos = jnp.tanh(self.deep_space_sincos(p_emb))  # [B, 60, 8]

        # Per-planet temperature: [B, 60, 1], broadcasts over the 64-dim logit axis
        T = jax.nn.softplus(self.temperature_head(p_emb)) + 0.1
        # Assemble unified hybrid tensor [B, 60, 72]
        logits = jnp.concatenate([
            jnp.concatenate([planetary_logits, ds_prob], axis=-1) / T,
            ds_sincos,
        ], axis=-1)

        return logits

class Critic(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        
        # Inject action
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        
        self.action_logits_proj = nnx.Linear(64, hidden_dim, rngs=rngs)
        self.action_sincos_proj = nnx.Linear(8, hidden_dim, rngs=rngs)

        # Project [p_emb || a_emb] from 2D down to D before global attention
        self.pa_proj = nnx.Linear(hidden_dim * 2, hidden_dim, rngs=rngs)

        # Global aggregation self-attention (now takes D-dim input)
        self.global_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.rezero_global = ReZero(rngs=rngs)
        
        # Final value MLP
        self.value_head = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            jax.nn.silu,
            nnx.Linear(hidden_dim, 1, rngs=rngs)
        )
        
    def __call__(self, planets, fleets, actions, planet_mask=None, fleet_mask=None):
        # [B, 60, D]
        p_emb = self.cross_attention_block(planets, fleets, fleet_mask=fleet_mask)
        
        p_coords = planets[..., 2:4]
        for i in range(self.num_sa_layers):
            sa = getattr(self, f'sa_block_{i}')
            p_emb = sa(p_emb, p_coords, planet_mask=planet_mask)
        
        # Softmax normalizes away actor temperature drift; sin/cos already in [-1,1]
        a_emb = self.action_logits_proj(jax.nn.softmax(actions[..., :64], axis=-1)) + self.action_sincos_proj(actions[..., 64:])
        pa_emb = jnp.concatenate([p_emb, a_emb], axis=-1)

        # Project 2D → D so global attention and ReZero can use a proper residual
        pa_proj = jax.nn.silu(self.pa_proj(pa_emb))  # [B, 60, D]

        # Global attention
        sa_mask = None
        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]

        ga_out = self.global_attn(inputs_q=pa_proj, mask=sa_mask)
        ga_out = self.rezero_global(pa_proj, ga_out)  # [B, 60, D]
        
        # Masked Global Mean Pooling over planets
        if planet_mask is not None:
            # Mask out invalid planets before sum
            ga_out = jnp.where(planet_mask[..., None], ga_out, 0.0)
            valid_counts = jnp.sum(planet_mask, axis=-1, keepdims=True)
            valid_counts = jnp.maximum(valid_counts, 1.0)
            global_emb = jnp.sum(ga_out, axis=-2) / valid_counts
        else:
            global_emb = jnp.mean(ga_out, axis=-2) # [..., D]
            
        # Final value
        q_val = self.value_head(global_emb) # [B, 1]
        return q_val

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
    # Temperature already applied by Actor; just softmax directly
    logits = raw_actions[..., :64]
    probs = jax.nn.softmax(logits, axis=-1)  # [B, 60, 64]

    ships = probs * current_ships[..., None]
    ships = ships * _SELF_TARGET_MASK  # zero self-sends (static constant, not recomputed)

    # Decode sin/cos → angle; gradient is smooth everywhere unlike direct tanh*π output
    ds_sincos = raw_actions[..., 64:72]  # [B, 60, 8]
    ds_sin = ds_sincos[..., :4]
    ds_cos = ds_sincos[..., 4:]
    ds_angle = jnp.arctan2(ds_sin, ds_cos)  # [B, 60, 4]

    return ships, ds_angle
