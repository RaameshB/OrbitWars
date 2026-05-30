import jax
import jax.numpy as jnp
from flax import nnx

class ReZero(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.alpha = nnx.Param(jnp.zeros(()))
        
    def __call__(self, x, f_x):
        return x + self.alpha[...] * f_x

def get_fourier_features(coords, num_bands=4):
    # coords: [..., 2] normalized to roughly [-1, 1]
    # We'll assume board size 1000, so normalize by 500
    norm_coords = (coords - 500.0) / 500.0
    
    freqs = jnp.exp(jnp.linspace(0.0, jnp.log(10.0), num_bands))
    # [..., 2, num_bands]
    args = norm_coords[..., None] * freqs[None, :] * jnp.pi
    
    # Flatten last two dims
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
    # Normalize dist by board size 1000
    norm_dist = dist / 1000.0
    
    theta = jnp.arctan2(dy, dx)
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    
    return jnp.stack([norm_dist, sin_theta, cos_theta], axis=-1)

class PlanetCrossAttentionBlock(nnx.Module):
    def __init__(self, hidden_dim: int, rngs: nnx.Rngs):
        # 7 base features + 2*4*2 = 16 fourier + 2 norm_coords = 25 features
        self.planet_mlp = nnx.Sequential(
            nnx.Linear(7 + 18, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )
        
        # 6 base features + 18 fourier = 24 features
        self.fleet_mlp = nnx.Sequential(
            nnx.Linear(6 + 18, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        )
        
        # Relative Bias Projection [3 -> num_heads]
        self.rel_bias_proj = nnx.Linear(3, 2, use_bias=False, rngs=rngs)
        
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
        
    def __call__(self, planets, fleets, fleet_mask=None):
        """
        planets: [B, 60, 7]
        fleets: [B, 7200, 5]
        """
        # Absolute Fourier Features
        p_coords = planets[..., 2:4] # Assuming 2:4 are y, x or x, y
        f_coords = fleets[..., 2:4]  # Assuming 2:4 are x, y
        
        p_fourier = get_fourier_features(p_coords)
        f_fourier = get_fourier_features(f_coords)
        
        p_in = jnp.concatenate([planets, p_fourier], axis=-1)
        f_in = jnp.concatenate([fleets, f_fourier], axis=-1)
        
        p_emb = self.planet_mlp(p_in) # [B, 60, D]
        f_emb = self.fleet_mlp(f_in)   # [B, 7200, D]
        
        ca_mask = None
        if fleet_mask is not None:
            ca_mask = fleet_mask[..., None, None, :]
            
        # Relative Bias
        rel_feats = get_relative_features(p_coords, f_coords) # [..., Q, KV, 3]
        bias_logits = self.rel_bias_proj(rel_feats) # [..., Q, KV, num_heads]
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
        return self.rezero_ca(p_emb, ca_out)

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
        self.rel_bias_proj = nnx.Linear(3, 2, use_bias=False, rngs=rngs)
        
    def __call__(self, p_emb, p_coords, planet_mask=None):
        
        rel_feats = get_relative_features(p_coords, p_coords) # [..., 60, 60, 3]
        bias_logits = self.rel_bias_proj(rel_feats) # [..., 60, 60, num_heads]
        attn_bias = jnp.moveaxis(bias_logits, -1, -3) # [..., num_heads, 60, 60]
        
        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]
            attn_bias = jnp.where(sa_mask, attn_bias, -1e9)
            
        sa_out = self.self_attn(inputs_q=p_emb, mask=attn_bias)
        return self.rezero_sa(p_emb, sa_out)

class Actor(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        
        # Shared Action Head MLP applied per-planet
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        
        self.action_head = nnx.Linear(hidden_dim, 60, rngs=rngs)
        self.deep_space_prob = nnx.Linear(hidden_dim, 4, rngs=rngs)
        self.deep_space_angle = nnx.Linear(hidden_dim, 4, rngs=rngs)
        
    def __call__(self, planets, fleets, planet_mask=None, fleet_mask=None):
        # [B, 60, D]
        p_emb = self.cross_attention_block(planets, fleets, fleet_mask=fleet_mask)
        
        p_coords = planets[..., 2:4]
        for i in range(self.num_sa_layers):
            sa = getattr(self, f'sa_block_{i}')
            p_emb = sa(p_emb, p_coords, planet_mask=planet_mask)
        
        # [B, 60, 60]
        planetary_logits = self.action_head(p_emb)
        
        # Add a strong structural bias to self-targeting (the diagonal)
        # This acts as a default 'do not launch' behavior for untrained networks
        planetary_logits = planetary_logits + jnp.eye(60) * 5.0
        
        # [B, 60, 4]
        # Add a +1.0 exploration bias so deep space isn't statistically drowned out by the 59 enemy planet logits
        ds_prob = self.deep_space_prob(p_emb) + 1.0
        ds_angle = jnp.tanh(self.deep_space_angle(p_emb)) * jnp.pi
        
        # Assemble unified hybrid tensor [B, 60, 68]
        logits = jnp.concatenate([planetary_logits, ds_prob, ds_angle], axis=-1)
        
        return logits

class Critic(nnx.Module):
    def __init__(self, hidden_dim: int, num_sa_layers: int = 6, rngs: nnx.Rngs = None):
        self.cross_attention_block = PlanetCrossAttentionBlock(hidden_dim, rngs=rngs)
        
        # Inject action
        self.num_sa_layers = num_sa_layers
        for i in range(num_sa_layers):
            setattr(self, f'sa_block_{i}', PlanetSelfAttentionBlock(hidden_dim, rngs=rngs))
        
        self.action_proj = nnx.Linear(68, hidden_dim, rngs=rngs)
        
        # Global aggregation self-attention
        self.global_attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=hidden_dim * 2,
            qkv_features=hidden_dim,
            out_features=hidden_dim,
            rngs=rngs,
            decode=False
        )
        self.rezero_global = ReZero(rngs=rngs)
        
        # Final value MLP
        self.value_head = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, 1, rngs=rngs)
        )
        
    def __call__(self, planets, fleets, actions, planet_mask=None, fleet_mask=None):
        # [B, 60, D]
        p_emb = self.cross_attention_block(planets, fleets, fleet_mask=fleet_mask)
        
        p_coords = planets[..., 2:4]
        for i in range(self.num_sa_layers):
            sa = getattr(self, f'sa_block_{i}')
            p_emb = sa(p_emb, p_coords, planet_mask=planet_mask)
        
        # Project actions and concatenate: [B, 60, D]
        a_emb = self.action_proj(actions)
        
        # [B, 60, 2D]
        pa_emb = jnp.concatenate([p_emb, a_emb], axis=-1)
        
        # Global attention
        sa_mask = None
        if planet_mask is not None:
            sa_mask = planet_mask[..., None, None, :]
            
        ga_out = self.global_attn(inputs_q=pa_emb, mask=sa_mask)
        
        # Since pa_emb is 2D and ga_out is D, we need a slight projection for rezero if we want a strict residual,
        # but actually we can just pass it as the new embedding.
        # Let's skip rezero here to change dimension, or project pa_emb down to D first.
        # Actually, simpler:
        ga_out = nnx.relu(ga_out) # Just use ga_out directly [B, 60, D]
        
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
    raw_actions: [B, 60, 62]
    current_ships: [B, 60]
    
    returns: (ships, angle)
    ships: [B, 60, 64] - Number of ships sent from each planet to each of the 60 targets + 4 deep space targets
    angle: [B, 60, 4] - Continuous launch angle for the deep space reserve
    """
    
    # Unpack hybrid tensor
    logits = raw_actions[..., :64]
    deep_space_angle = raw_actions[..., 64:68]
    
    # Scale logits by Temperature to produce harder deterministic actions
    # Temperature < 1.0 makes softmax sharper. 
    T = 0.1 
    probs = jax.nn.softmax(logits / T, axis=-1) # [B, 60, 61]
    
    # Multiply the probability distribution by the available ships on the planet
    ships = probs * current_ships[..., None]
    
    # Zero out the diagonal to correctly enforce the "Wait and do nothing" inductive bias
    # Ships targeting their own planet should NOT be launched as fleets.
    mask = jnp.ones((60, 64))
    mask = mask.at[:60, :60].set(1.0 - jnp.eye(60))
    ships = ships * mask
    
    return ships, deep_space_angle
