"""
OrbitWars RL Training — REINFORCE → PPO with S5 critic + ELO Hall of Fame
Script version: v11.24
# v10.1: ESN capacity d_max=12, w_max=256
# v10.2: CMA-ES val/train split (80/20); log r2_val + r2_train + gap
# v10.3: PCA-compressed readout (D_total 3120→432); W_enc[D_max,enc_dim,W_max] unified
# v10.4: Widened CMA-ES bounds — SR [0.1, 2.0], beta [~0, 20.1]
# v10.5: Entropy cap — coef 0.01→0.001; linear penalty above ent_target=8.0
# v10.6: Launch gate bias -1.5 applied to hold_head.bias.value at load time
# v10.7: Advantage scale = clip(esn_r2_val, floor=0.2, 1.0) — ESN-confidence-weighted updates
# v10.8: SR bounds [0.7,1.6]; r2_scale floor 0.2→0 (skip PPO when val R²<0); entropy floor 6.0
# v10.9: Fix SR=0.9 (remove from CMA-ES, N_CMA_DIMS 6→5); lookback buffer (last N iters)
# v11.0: Replace SuperESN+CMA-ES critic with single-layer GRU (MSE on GAE returns)
# v11.7: ELO updates every iter (k/hof_update_every); update_elo wired up so HoF entries drift; admission merit-only
# v11.8: HoF fills unconditionally; new entries start at current_elo so eviction is merit-based
# v11.9: Replace GRU critic with fixed-reservoir ESN + ridge regression readout; startup grid search
# v11.10: Fix ESN scan bug — was emitting pre-update state (off-by-one); now emits h_new
# v11.11: Replace single-layer ESNCritic with DeePrESNCritic (6-layer, PCA inter-layer, ridge)
# v11.12: Fix readout to use raw reservoir states (paper Eq.12); PCA only for inter-layer; λ grid 1e-5..0.1
# v11.13: Normalize features before ridge (fix R²=-583 explosion); λ grid 1..10000 for autocorrelated data
# v11.14: Sep-CMA-ES HPO (8×3 gens) over (SR, leak); GCV for optimal λ each iter; gram matrix on-device
# v11.15: Replace GCV (wrong for autocorrelated data) with val-game R² grid (200 log-λ, fully in JAX)
# v11.16: Replace DeePr-ESN critic with trainable S5 (diagonal SSM, HiPPO init, associative scan)
# v11.17: Correct S5 init: B~=V⁻¹B, C~=CV from HiPPO-N eigendecomp; D feedthrough; .real not 2×real
# v11.23: R²-gated PPO (skip when prev R²<0.2); per-device HoF opponent sampling (8 opps/iter)
# v11.24: Fix ent cap/floor to use per-planet entropy (normalize by n_owned); thresholds 4.0/1.5

Architecture:
  - Actor:   BC-pretrained transformer, AdamW, Gaussian logit noise for exploration
  - Critic:  S5 (diagonal SSM, HiPPO init, trained end-to-end with Adam)
  - Opponents: ELO Hall of Fame, 70% ELO-weighted + 30% uniform sampling
  - Exploiters: periodically spawned, trained against frozen current policy,
                enter HoF on merit (no special treatment once added)
  - Parallelism: pmap over devices (TPUv5e-8 / 2×T4), vmap over games per device

Usage (Colab / TPU VM):
    python scripts/train_rl.py --weights bc_v14bc/weights_bc_v14.pkl
"""
import os, sys, time, math, pickle, threading, subprocess, dataclasses, functools
from collections import deque
from typing import Any
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import numpy as np

# ── JAX backend (set before importing jax) ────────────────────────────────────
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.90')
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')

import jax, jax.numpy as jnp, jax.random as jr
from flax import nnx
import optax

from core.networks import Actor, logits_to_action
from core import orbit_wars_jax as env_lib
from core.rollout_utils import calculate_intercept_angle

# ── Config ────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Config:
    # Rollout
    games_per_device: int  = 64       # × n_devices = total parallel games; 64 for T4, 128 for TPUv5e
    max_steps:        int  = 500      # MAX_EPISODE_STEPS
    fleet_cap:        int  = 64       # store this many fleet slots per step

    # PPO
    ppo_epochs:       int   = 4
    minibatch_size:   int   = 2048    # transitions per minibatch
    lr:               float = 3e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_eps:         float = 0.1
    launch_gate_bias: float = 0.0     # 0 = use BC's natural gate; S5 critic teaches hold/launch from advantages
    entropy_coef:     float = 0.001   # small bonus; σ=0.3 noise handles exploration
    ent_max_coef:     float = 0.05    # penalty coef for cap and floor
    ent_target:       float = 4.0     # per-planet nats cap; max = ln(64)≈4.158 (BC near-uniform)
    ent_floor:        float = 1.5     # per-planet nats floor; ln(4.5)≈1.5 fires when very concentrated
    max_grad_norm:    float = 0.5
    logit_sigma:      float = 0.3     # Gaussian exploration noise on logits

    # DeePr-ESN critic
    esn_d_max:    int   = 6    # reservoir layers (K)
    esn_w_max:    int   = 256  # units per layer (N per reservoir)
    esn_enc_dim:  int   = 60   # PCA compression dim for inter-layer connections only

    # Critic warmup — hold policy fixed, train critic only on BC rollouts
    critic_warmup_iters: int = 25   # iters before PPO starts; critic converges on stationary dist
    r2_ppo_threshold:   float = 0.2  # skip PPO if prev iter R² below this; critic recovers undisturbed

    # Hall of Fame
    hof_size:         int   = 10
    hof_eval_games:   int   = 20      # games to eval new checkpoint vs HoF
    hof_update_every: int   = 50      # iters between HoF evals
    hof_elo_weight:   float = 0.70    # fraction of opponent sampling by ELO
    hof_init_elo:     float = 1000.0
    hof_k_factor:     float = 32.0    # ELO K-factor

    # Exploiters
    exploiter_spawn_every: int = 200  # iters between exploiter spawns
    exploiter_budget:      int = 100  # training iters per exploiter run
    exploiter_hof_min_elo: float = 1050.0  # min ELO to enter HoF

    # Architecture
    hidden_dim:       int   = 48
    num_sa_layers:    int   = 6

    # Reward shaping
    # Exponential ramp: weight at step T = e^alpha × weight at step 0.
    # Median reward step (half weight before / after):
    #   alpha=1 → step 320,  alpha=2 → 360,  alpha=3 → 395,  alpha=4 → 415 (of 500)
    # alpha=2 keeps early/mid-game signal meaningful; alpha=3 for more endgame emphasis.
    reward_alpha:     float = 2.0

    # Checkpointing
    save_every:       int   = 100
    r2_prefix:        str   = 'rl_v1'
    out_dir:          str   = '/content/rl_checkpoints'


CFG = Config()

# ── R2 sync thread ────────────────────────────────────────────────────────────
class R2SyncThread(threading.Thread):
    def __init__(self, local_dir, bucket, interval=300):
        super().__init__(daemon=True)
        self.local_dir, self.bucket, self.interval = local_dir, bucket, interval
        import boto3 as _b3
        self.s3 = _b3.client('s3',
            endpoint_url=os.environ['R2_ENDPOINT_URL'],
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name='auto')

    def run(self):
        while True:
            time.sleep(self.interval)
            try:
                for fn in os.listdir(self.local_dir):
                    lp = os.path.join(self.local_dir, fn)
                    self.s3.upload_file(lp, os.environ['R2_BUCKET_NAME'],
                                        f'{CFG.r2_prefix}/{fn}')
                print(f'[{time.strftime("%H:%M:%S")}] R2 sync OK')
            except Exception as e:
                print(f'[{time.strftime("%H:%M:%S")}] R2 sync failed: {e}')

# ── Batched obs builder (mirrors train_backup build_env_arrays) ───────────────
def build_env_arrays(state, params, pid, num_players):
    """Build observation arrays for all games in a batch.
    state/params: leading batch dim B (games per device).
    Returns: planets [B,60,7], fleets [B,FLEET_CAP,6], pmask [B,60], fmask [B,FLEET_CAP]
    """
    B = state.planet_owners.shape[0]
    theta  = -pid * (2 * jnp.pi / num_players)
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)

    rel_own = jnp.where(state.planet_owners == pid, 1.0,
                        jnp.where(state.planet_owners == -1, 0.0, -1.0))
    dx = state.planet_coords[..., 0] - 50.0
    dy = state.planet_coords[..., 1] - 50.0
    planets = jnp.stack([
        jnp.broadcast_to(jnp.arange(60), (B, 60)).astype(jnp.float32),
        rel_own,
        dx * cos_t - dy * sin_t + 50.0,
        dx * sin_t + dy * cos_t + 50.0,
        params.planet_radii,
        state.planet_ships.astype(jnp.float32),
        params.planet_prod.astype(jnp.float32),
    ], axis=-1)  # [B, 60, 7]

    rel_f = jnp.where(state.fleet_owners == pid, 1.0,
                      jnp.where(state.fleet_owners == -1, 0.0, -1.0))
    fdx = state.fleet_coords[..., 0] - 50.0
    fdy = state.fleet_coords[..., 1] - 50.0
    fleets_full = jnp.stack([
        jnp.broadcast_to(jnp.arange(env_lib.MAX_FLEETS), (B, env_lib.MAX_FLEETS)).astype(jnp.float32),
        rel_f,
        state.fleet_angles + theta,
        fdx * cos_t - fdy * sin_t + 50.0,
        fdx * sin_t + fdy * cos_t + 50.0,
        state.fleet_ship_count.astype(jnp.float32),
    ], axis=-1)  # [B, MAX_FLEETS, 6]

    fleets = fleets_full[:, :CFG.fleet_cap, :]
    pmask  = state.planet_owners != -1
    fmask  = state.fleet_owners[:, :CFG.fleet_cap] != -1
    return planets, fleets, pmask, fmask


# ── Actor helpers ─────────────────────────────────────────────────────────────
def make_actor():
    return Actor(hidden_dim=CFG.hidden_dim, num_sa_layers=CFG.num_sa_layers, rngs=nnx.Rngs(0))

def params_to_np(params):
    return jax.tree_util.tree_map(np.array, params)

def np_to_params(np_params):
    return jax.tree_util.tree_map(jnp.array, np_params)

def make_forward(actor_graph):
    """Return a vmappable forward function."""
    def fwd(params, planets, fleets, pmask, fmask):
        return nnx.merge(actor_graph, params)(planets, fleets,
                                              planet_mask=pmask, fleet_mask=fmask)
    return fwd

def make_encode(actor_graph):
    def enc(params, planets, fleets, fmask):
        pmask = planets[..., 5] >= 0
        return nnx.merge(actor_graph, params).encode(planets, fleets,
                                                      planet_mask=pmask, fleet_mask=fmask)
    return enc


# ── GRU critic ───────────────────────────────────────────────────────────────
#
# Single-layer GRU trained each iter via MSE on GAE return targets.
# Params stored as a plain JAX pytree; optimizer is optax Adam.
# values() returns constant 0.5 at iter 0 (cold start), then GRU predictions.
#
class DeePrESNCritic:
    """Multi-layer Deep-reservoir ESN with PCA inter-layer encoders + ridge readout.

    Readout = [emb, x_res^1,...,x_res^K] (raw reservoir states, paper Eq.12).
    HPO (first iter only): Sep-CMA-ES over (SR, leak).
    Lambda: 200-point log-grid search maximising val R² on held-out games each
    iter — correctly handles autocorrelated trajectories (GCV does not: it treats
    150K autocorrelated timesteps as independent, always selects near-zero lambda).
    All heavy computation runs in JAX (on TPU); no large host transfers.
    """
    WO_FRAC     = 0.05
    SPARSITY    = 0.1
    VAL_FRAC    = 0.20           # fraction of games held out for lambda + HPO scoring
    SR_BOUNDS   = (0.70, 0.99)
    LEAK_BOUNDS = (0.10, 0.70)
    POPSIZE     = 8
    N_GENS      = 3
    LAM_LOG_LO  = 0.0            # e^0  = 1
    LAM_LOG_HI  = 9.2            # e^9.2 ≈ 10000
    LAM_GRID_N  = 200

    def __init__(self):
        D_MAX   = CFG.esn_d_max
        W_MAX   = CFG.esn_w_max
        ENC_DIM = CFG.esn_enc_dim
        n_in    = CFG.hidden_dim
        D_total = n_in + D_MAX * W_MAX  # 48 + 6*256 = 1584 raw reservoir features

        self._D_MAX   = D_MAX
        self._W_MAX   = W_MAX
        self._ENC_DIM = ENC_DIM
        self._D_total = D_total
        self._n_in    = n_in

        rng = np.random.RandomState(42)

        self._W_in0 = jnp.array(
            rng.uniform(-1, 1, (W_MAX, n_in)).astype(np.float32))
        self._W_in_deep = jnp.array(
            rng.uniform(-1, 1, (D_MAX - 1, W_MAX, ENC_DIM)).astype(np.float32))

        # Store unit-SR matrices; scale by SR at runtime for HPO candidates
        W_res_unit = np.zeros((D_MAX, W_MAX, W_MAX), dtype=np.float32)
        for i in range(D_MAX):
            W = rng.uniform(-0.5, 0.5, (W_MAX, W_MAX))
            W *= rng.rand(W_MAX, W_MAX) < self.SPARSITY
            rho = np.max(np.abs(np.linalg.eigvals(W)))
            W_res_unit[i] = (W / rho if rho > 1e-6 else W).astype(np.float32)
        self._W_res_unit = W_res_unit

        W_enc = np.zeros((D_MAX, ENC_DIM, W_MAX), dtype=np.float32)
        for i in range(D_MAX):
            W_enc[i, :, :ENC_DIM] = np.eye(ENC_DIM)
        self._W_enc = jnp.array(W_enc)

        sr0 = float(np.mean(self.SR_BOUNDS))
        lk0 = float(np.mean(self.LEAK_BOUNDS))
        self._best_sr   = sr0
        self._best_leak = lk0
        self._best_lam  = 100.0
        self._W_res     = jnp.array(W_res_unit * sr0)

        self._W_out     = None
        self._feat_mean = None
        self._feat_std  = None
        self._fitted    = False

        (self._jit_forward,
         self._jit_refit_pca,
         self._jit_gram_eigh,
         self._jit_best_lambda) = self._make_jit_fns()

    def _make_jit_fns(self):
        W_in0     = self._W_in0
        W_in_deep = self._W_in_deep
        D_MAX     = self._D_MAX
        W_MAX     = self._W_MAX
        ENC_DIM   = self._ENC_DIM
        LAM_LOG_LO = self.LAM_LOG_LO
        LAM_LOG_HI = self.LAM_LOG_HI
        LAM_GRID_N = self.LAM_GRID_N

        @jax.jit
        def forward(emb, W_enc, W_res, leak):
            """emb [B,T,D] → (features [B,T,D_total], states [B,T,D_MAX,W_MAX]).
            W_res and leak are traced — JIT compiles once for all values."""
            W_r0 = W_res[0]
            def seq0(seq):
                def step(h, u):
                    h2 = (1 - leak) * h + leak * jnp.tanh(W_r0 @ h + W_in0 @ u)
                    return h2, h2
                _, s = jax.lax.scan(step, jnp.zeros(W_MAX), seq)
                return s
            s0   = jax.vmap(seq0)(emb)
            enc0 = s0 @ W_enc[0].T
            parts        = [emb, s0]
            layer_states = [s0]
            x            = enc0

            for i in range(1, D_MAX):
                W_ri = W_res[i]
                W_ii = W_in_deep[i - 1]
                def seq_i(seq, W_r=W_ri, W_i=W_ii):
                    def step(h, u):
                        h2 = (1 - leak) * h + leak * jnp.tanh(W_r @ h + W_i @ u)
                        return h2, h2
                    _, s = jax.lax.scan(step, jnp.zeros(W_MAX), seq)
                    return s
                si   = jax.vmap(seq_i)(x)
                parts.append(si)
                layer_states.append(si)
                if i < D_MAX - 1:
                    x = si @ W_enc[i].T

            features       = jnp.concatenate(parts, axis=-1)
            states_stacked = jnp.stack(layer_states, axis=2)
            return features, states_stacked

        @jax.jit
        def refit_pca(states_stacked):
            B, T = states_stacked.shape[:2]
            S = states_stacked.transpose(2, 0, 1, 3).reshape(D_MAX, B * T, W_MAX)
            def fit_one(s):
                C = s.T @ s
                _, vecs = jnp.linalg.eigh(C)
                return vecs[:, -ENC_DIM:].T
            return jax.vmap(fit_one)(S)

        @jax.jit
        def gram_eigh(feat_flat, mask_flat, y_flat):
            """feat [N,D], mask [N], y [N] → mu [D], std [D], d [D], V [D,D], q [D].
            Computes gram matrix and eigendecomp entirely on-device."""
            n   = mask_flat.sum() + 1e-8
            Xm  = feat_flat * mask_flat[:, None]
            mu  = Xm.sum(0) / n
            std = jnp.sqrt(((feat_flat - mu) ** 2 * mask_flat[:, None]).sum(0) / n + 1e-8)
            Xn  = (feat_flat - mu) / std
            Xnm = Xn * mask_flat[:, None]
            ynm = y_flat * mask_flat
            A   = Xnm.T @ Xnm                   # [D,D] gram matrix
            Xty = Xnm.T @ ynm                   # [D]
            d, V = jnp.linalg.eigh(A)           # ascending eigenvalues; d≥0 for PSD
            d   = jnp.maximum(d, 0.0)           # clip numerical negatives
            q   = V.T @ Xty                     # [D] projected targets
            return mu, std, d, V, q

        @jax.jit
        def best_lambda(X_val_norm, y_val, mask_val, q, d):
            """200-point log-grid search for lambda maximising val R².
            X_val_norm [N_val,D], q [D], d [D] — all on-device.
            Returns (best_log_lam scalar, best_r2 scalar)."""
            log_grid = jnp.linspace(LAM_LOG_LO, LAM_LOG_HI, LAM_GRID_N)
            lams     = jnp.exp(log_grid)                             # [L]
            # W(λ) = V diag(1/(d+λ)) q  →  pred = X_val @ V @ diag(1/(d+λ)) @ q
            # Equivalently: pred = X_val_norm @ W_lam.T where W_lam [L,D]
            W_lam    = q[None, :] / (d[None, :] + lams[:, None])    # [L,D]
            preds    = X_val_norm @ W_lam.T                          # [N_val,L]
            n_v      = mask_val.sum() + 1e-8
            mu_y     = (y_val * mask_val).sum() / n_v
            ss_tot   = ((y_val - mu_y) ** 2 * mask_val).sum() + 1e-8
            ss_res   = ((y_val[:, None] - preds) ** 2 * mask_val[:, None]).sum(0)  # [L]
            r2s      = 1.0 - ss_res / ss_tot
            best_i   = jnp.argmax(r2s)
            return log_grid[best_i], r2s[best_i]

        return forward, refit_pca, gram_eigh, best_lambda

    def _build_W_res(self, sr):
        return jnp.array(self._W_res_unit * sr)

    def _solve_from_eigh(self, V, d, q, lam):
        """Ridge solution W = V diag(1/(d+λ)) q. O(D²), avoids recomputing gram."""
        return V @ (q / (d + lam))

    def _r2(self, Xn, y, mask, W_out):
        pred   = Xn @ W_out
        n      = mask.sum() + 1e-8
        mu     = (y * mask).sum() / n
        ss_res = ((y - pred) ** 2 * mask).sum()
        ss_tot = ((y - mu)   ** 2 * mask).sum() + 1e-8
        return float(1.0 - ss_res / ss_tot)

    # ── HPO: Sep-CMA-ES over (SR, leak); val-R² lambda each candidate ────────

    def _score_candidate(self, feat_jax, mask_2d_jax, y_2d_jax, idx_trn, idx_val):
        """Score one (SR, leak) candidate entirely on-device. Returns (r2_val, lambda)."""
        D     = feat_jax.shape[-1]
        m_trn = mask_2d_jax[idx_trn].reshape(-1)
        if int(m_trn.sum()) < 10:
            return -1.0, self._best_lam

        # Gram + eigendecomp on train split (on-device)
        mu, std, d, V, q = self._jit_gram_eigh(
            feat_jax[idx_trn].reshape(-1, D), m_trn, y_2d_jax[idx_trn].reshape(-1))

        # Val: normalise with train stats, then find optimal lambda
        X_val_norm = (feat_jax[idx_val].reshape(-1, D) - mu) / std
        best_log_lam, r2_val = self._jit_best_lambda(
            X_val_norm, y_2d_jax[idx_val].reshape(-1),
            mask_2d_jax[idx_val].reshape(-1), q, d)

        return float(r2_val), float(math.exp(float(best_log_lam)))

    def _run_hpo(self, emb, mask_2d_jax, y_2d_jax):
        """Sep-CMA-ES over (SR, leak) in [0,1]² space. Returns (sr, leak, lam, r2_val)."""
        from evosax.algorithms import Sep_CMA_ES

        B       = emb.shape[0]
        n_val   = max(1, int(B * self.VAL_FRAC))
        idx_val = np.arange(n_val)
        idx_trn = np.arange(n_val, B)

        SR_lo, SR_hi = self.SR_BOUNDS
        LK_lo, LK_hi = self.LEAK_BOUNDS

        def decode(x_arr):
            sr   = SR_lo + float(np.clip(x_arr[0], 0., 1.)) * (SR_hi - SR_lo)
            leak = LK_lo + float(np.clip(x_arr[1], 0., 1.)) * (LK_hi - LK_lo)
            return sr, leak

        strategy = Sep_CMA_ES(population_size=self.POPSIZE, solution=jnp.zeros(2))
        params   = strategy.default_params.replace(std_init=0.3)
        rng_key  = jax.random.PRNGKey(42)
        state    = strategy.init(rng_key, jnp.array([0.5, 0.5]), params)

        best_r2  = -np.inf
        best_sr, best_lk, best_lam = self._best_sr, self._best_leak, self._best_lam

        print(f'  [esn-hpo] Sep-CMA-ES {self.POPSIZE}×{self.N_GENS} gens over (SR, leak); val-R² for λ', flush=True)
        for gen in range(self.N_GENS):
            rng_key, ask_key, tell_key = jax.random.split(rng_key, 3)
            x, state = strategy.ask(ask_key, state, params)
            fitnesses = np.zeros(self.POPSIZE, dtype=np.float32)
            for i in range(self.POPSIZE):
                sr, lk = decode(np.array(x[i]))
                W_r    = self._build_W_res(sr)
                feat, _ = self._jit_forward(emb, self._W_enc, W_r, jnp.float32(lk))
                r2v, lam = self._score_candidate(feat, mask_2d_jax, y_2d_jax, idx_trn, idx_val)
                fitnesses[i] = -r2v
                if r2v > best_r2:
                    best_r2, best_sr, best_lk, best_lam = r2v, sr, lk, lam
                print(f'    g{gen}i{i} sr={sr:.3f} lk={lk:.3f} → r2_val={r2v:.3f} λ={lam:.1f}', flush=True)
            state, _ = strategy.tell(tell_key, x, jnp.array(fitnesses), state, params)
            print(f'  [esn-hpo] gen {gen}: best r2_val={best_r2:.3f}  sr={best_sr:.3f} lk={best_lk:.3f}', flush=True)

        return best_sr, best_lk, best_lam, best_r2

    # ── Public API ────────────────────────────────────────────────────────────

    def values(self, emb):
        """emb [B,T,D] → [B,T] values."""
        if not self._fitted:
            return jnp.full(emb.shape[:2], 0.5)
        feat, _ = self._jit_forward(emb, self._W_enc, self._W_res, jnp.float32(self._best_leak))
        Xn = (feat.reshape(-1, self._D_total) - self._feat_mean) / self._feat_std
        return (Xn @ self._W_out).reshape(emb.shape[:2])

    def update(self, emb, active, return_targets):
        """HPO on first iter, then PCA refit + val-R² lambda + ridge each iter. Returns (r2_val, 0.0, elapsed_s)."""
        t0   = time.time()
        B, T = emb.shape[:2]
        wo   = int(self.WO_FRAC * T)
        mask = active.astype(jnp.float32)
        wash = jnp.ones((B, T), dtype=jnp.float32).at[:, :wo].set(0.0)
        mask_2d = mask * wash                                       # [B,T]

        n_val   = max(1, int(B * self.VAL_FRAC))
        idx_val = np.arange(n_val)
        idx_trn = np.arange(n_val, B)

        # First iter: HPO selects (SR, leak) before PCA refit
        if not self._fitted:
            sr, lk, lam, r2v = self._run_hpo(emb, mask_2d, return_targets)
            self._best_sr   = sr
            self._best_leak = lk
            self._best_lam  = lam
            self._W_res     = self._build_W_res(sr)
            print(f'  [esn-hpo] Selected: sr={sr:.3f} leak={lk:.3f} λ={lam:.1f} r2_val={r2v:.3f}', flush=True)

        leak_f32 = jnp.float32(self._best_leak)

        # PCA refit with selected reservoir, then forward with updated encoders
        _, states   = self._jit_forward(emb, self._W_enc, self._W_res, leak_f32)
        self._W_enc = self._jit_refit_pca(states)
        feat, _     = self._jit_forward(emb, self._W_enc, self._W_res, leak_f32)

        # Gram + eigendecomp on train split (on-device)
        mu, std, d, V, q = self._jit_gram_eigh(
            feat[idx_trn].reshape(-1, self._D_total),
            mask_2d[idx_trn].reshape(-1),
            return_targets[idx_trn].reshape(-1))
        self._feat_mean = mu
        self._feat_std  = std

        # Val-R² optimal lambda (on-device, no scipy)
        X_val_norm = (feat[idx_val].reshape(-1, self._D_total) - mu) / std
        best_log_lam, r2_val_jax = self._jit_best_lambda(
            X_val_norm,
            return_targets[idx_val].reshape(-1),
            mask_2d[idx_val].reshape(-1), q, d)
        self._best_lam = float(math.exp(float(best_log_lam)))

        # Ridge solve via eigendecomp (O(D²), no gram recomputation)
        self._W_out  = self._solve_from_eigh(V, d, q, self._best_lam)
        self._fitted = True

        # Train R² for logging
        X_trn_norm = (feat[idx_trn].reshape(-1, self._D_total) - mu) / std
        r2_train   = self._r2(X_trn_norm, return_targets[idx_trn].reshape(-1),
                               mask_2d[idx_trn].reshape(-1), self._W_out)
        r2_val     = float(r2_val_jax)
        print(f'  [esn] sr={self._best_sr:.3f} leak={self._best_leak:.3f} '
              f'λ={self._best_lam:.1f} R²_trn={r2_train:.3f} R²_val={r2_val:.3f}', flush=True)
        return r2_val, 0.0, time.time() - t0


# ── S5 critic ────────────────────────────────────────────────────────────────
# Simplified State Space Sequence model (Smith et al. 2023), Appendix A + B.
# L=4 layers, H=64, block-diagonal HiPPO init with blocks=8 (block_size=8).
# Conjugate symmetry (Section 3.2): N = blocks*(block_size//2) = 32 states.
# Λ fixed (shared across layers); B~,C~,D,log_Δ,LN learned per layer.
# Output per layer: 2·Re(C~·x) + D·u  (2× from conjugate symmetry).

def _hippo_dplr(block_size):
    """Exact match to lindermanlab/S5 make_DPLR_HiPPO for one block.
    Returns Lambda [block_size] complex64 (ALL eigenvalues), V [block_size,block_size].
    eigh sorts ascending: [:block_size//2] has Im<0, [block_size//2:] has Im>0."""
    P_vec = np.sqrt(1 + 2 * np.arange(block_size, dtype=np.float64))
    A = P_vec[:, None] * P_vec[None, :]
    A = np.tril(A) - np.diag(np.arange(block_size, dtype=np.float64))
    nhippo = -A
    p = np.sqrt(np.arange(block_size, dtype=np.float64) + 0.5)
    S = nhippo + np.outer(p, p)          # A^Normal — NOT symmetric
    Lambda_imag, V = np.linalg.eigh(S * -1j)
    Lambda_real = np.mean(np.diag(S)) * np.ones(block_size)   # all = -0.5
    Lambda = (Lambda_real + 1j * Lambda_imag).astype(np.complex64)
    return Lambda, V.astype(np.complex64)


def _s5_layer_norm(x, scale, bias, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean) ** 2).mean(-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def _s5_scan_op(elem_i, elem_j):
    A_i, Bu_i = elem_i; A_j, Bu_j = elem_j
    return A_j * A_i, A_j * Bu_i + Bu_j


def _s5_net(params, Lambda_re, Lambda_im, u):
    """Single-game forward. u [T, D_in] → values [T].
    Lambda_{re,im} [N] fixed (tiled HiPPO blocks, positive-Im only)."""
    x = u @ params['W_proj'].T + params['b_proj']          # [T, H]

    N      = Lambda_re.shape[0]
    Lambda = Lambda_re.astype(jnp.complex64) + 1j * Lambda_im.astype(jnp.complex64)
    T      = x.shape[0]

    for layer in params['layers']:
        h     = _s5_layer_norm(x, layer['ln1_scale'], layer['ln1_bias'])
        dt    = jnp.exp(layer['log_dt'])                   # [N]
        A_bar = jnp.exp(dt * Lambda)                       # [N]
        B     = layer['B_re'].astype(jnp.complex64) + 1j * layer['B_im'].astype(jnp.complex64)  # [N, H]
        B_bar = ((A_bar - 1.0) / Lambda)[:, None] * B     # [N, H]
        C     = layer['C_re'].astype(jnp.complex64) + 1j * layer['C_im'].astype(jnp.complex64)  # [H, N]
        Bu    = h.astype(jnp.complex64) @ B_bar.T          # [T, N]
        As    = jnp.broadcast_to(A_bar, (T, N))
        _, xs = jax.lax.associative_scan(_s5_scan_op, (As, Bu))    # [T, N]
        ys    = 2.0 * (xs @ C.T).real + layer['D'] * h    # [T, H]
        x     = _s5_layer_norm(x + jax.nn.gelu(ys), layer['ln2_scale'], layer['ln2_bias'])

    return (x @ params['W_head'].T + params['b_head'])[:, 0]        # [T]


def _s5_init_params(D_in, H, key, L=4, blocks=8, dt_min=0.001, dt_max=0.1):
    """L-layer S5, block-diagonal HiPPO init (blocks copies of HiPPO-block_size).
    Conjugate symmetry: N = blocks * (block_size//2) total complex states.
    Returns (params dict, Lambda_re [N], Lambda_im [N])."""
    block_size = H // blocks           # 64 // 8 = 8
    N_block    = block_size // 2       # conjugate symmetry per block: 4
    N          = blocks * N_block      # total states: 32

    # One block's eigendecomposition — tiled across all blocks
    Lambda_b, V_b = _hippo_dplr(block_size)   # [block_size], [block_size, block_size]
    Lambda_one  = Lambda_b[N_block:]           # [N_block] positive-Im
    V_N_one     = V_b[:, N_block:]             # [block_size, N_block]
    V_N_inv_one = V_N_one.conj().T             # [N_block, block_size]

    Lambda_all = np.tile(Lambda_one, blocks)   # [N] tiled eigenvalues

    # 2 shared keys + 4 per layer (B, C, D, dt)
    all_keys  = jax.random.split(key, 2 + L * 4)
    k_proj, k_head = all_keys[0], all_keys[1]
    lkeys = all_keys[2:]

    std = H ** -0.5

    def _block_B(B_real):
        # B_real [H, H]; split into blocks along rows, project each
        parts = [V_N_inv_one @ B_real[j*block_size:(j+1)*block_size, :]
                 for j in range(blocks)]
        return np.concatenate(parts, axis=0).astype(np.complex64)  # [N, H]

    def _block_C(C_real):
        # C_real [H, H]; split into blocks along cols, project each
        parts = [C_real[:, j*block_size:(j+1)*block_size] @ V_N_one
                 for j in range(blocks)]
        return np.concatenate(parts, axis=1).astype(np.complex64)  # [H, N]

    layers = []
    for i in range(L):
        k_B, k_C, k_D, k_dt = lkeys[4*i], lkeys[4*i+1], lkeys[4*i+2], lkeys[4*i+3]
        B_tilde = _block_B(np.array(jax.random.normal(k_B, (H, H)) * std))
        C_tilde = _block_C(np.array(jax.random.normal(k_C, (H, H)) * std))
        layers.append({
            'B_re': jnp.array(B_tilde.real),  'B_im': jnp.array(B_tilde.imag),
            'C_re': jnp.array(C_tilde.real),  'C_im': jnp.array(C_tilde.imag),
            'D':      jax.random.normal(k_D, (H,)),
            'log_dt': jax.random.uniform(k_dt, (N,),
                                         minval=math.log(dt_min), maxval=math.log(dt_max)),
            'ln1_scale': jnp.ones(H),  'ln1_bias':  jnp.zeros(H),
            'ln2_scale': jnp.ones(H),  'ln2_bias':  jnp.zeros(H),
        })

    params = {
        'W_proj': jax.random.normal(k_proj, (H, D_in)) * D_in ** -0.5,
        'b_proj': jnp.zeros(H),
        'layers': layers,
        'W_head': jax.random.normal(k_head, (1, H)) * H ** -0.5,
        'b_head': jnp.zeros(1),
    }
    Lambda_re = jnp.array(Lambda_all.real, dtype=jnp.float32)   # [N]
    Lambda_im = jnp.array(Lambda_all.imag, dtype=jnp.float32)   # [N]
    return params, Lambda_re, Lambda_im


class S5Critic:
    """S5 critic (Smith et al. 2023): L=4 layers, block-diagonal HiPPO init.

    H=64, blocks=8 (block_size=8), N=32 complex states (conjugate symmetry).
    Λ fixed (tiled HiPPO-8 eigenvalues). B̃,C̃,D,log_Δ,LN learned per layer.
    Returns (r2_val, 0.0, elapsed_s) from update() to match the DeePrESN API.
    """
    H        = 64     # hidden dim
    L        = 4      # S5 layers
    BLOCKS   = 8      # block-diagonal blocks (block_size = H // BLOCKS = 8)
    VAL_FRAC = 0.20
    LR       = 2e-3
    N_EPOCHS = 4
    MB_GAMES = 32

    def __init__(self):
        key = jax.random.PRNGKey(42)
        params, Lambda_re, Lambda_im = _s5_init_params(CFG.hidden_dim, self.H, key,
                                                        L=self.L, blocks=self.BLOCKS)
        self._params    = params
        self._fitted    = False
        opt             = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(self.LR))
        self._opt_state = opt.init(self._params)

        # Lambda is fixed — captured as a closure, not in the grad path
        Lre, Lim = Lambda_re, Lambda_im

        @jax.jit
        def _predict_jit(params, emb):
            return jax.vmap(lambda u: _s5_net(params, Lre, Lim, u))(emb)

        @jax.jit
        def _train_step_jit(params, opt_state, emb_mb, y_mb, mask_mb):
            def loss_fn(p):
                pred = jax.vmap(lambda u: _s5_net(p, Lre, Lim, u))(emb_mb)
                return ((pred - y_mb) ** 2 * mask_mb).sum() / (mask_mb.sum() + 1e-8)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = opt.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt_state, loss

        self._predict_jit    = _predict_jit
        self._train_step_jit = _train_step_jit

    def values(self, emb):
        """emb [B, T, D] → [B, T] values ∈ [0, 1]."""
        if not self._fitted:
            return jnp.full(emb.shape[:2], 0.5)
        return jnp.clip(self._predict_jit(self._params, emb), 0.0, 1.0)

    def update(self, emb, active, return_targets):
        """Train on current rollout. Returns (r2_val, 0.0, elapsed_s)."""
        t0      = time.time()
        B, T, D = emb.shape

        wo   = max(1, int(0.05 * T))
        wash = jnp.ones((B, T), dtype=jnp.float32).at[:, :wo].set(0.0)
        mask = active.astype(jnp.float32) * wash

        n_val   = max(1, int(B * self.VAL_FRAC))
        idx_val = np.arange(n_val)
        idx_trn = np.arange(n_val, B)

        emb_t  = emb[idx_trn]; y_t = return_targets[idx_trn]; m_t = mask[idx_trn]
        n_trn  = len(idx_trn)
        rng_np = np.random.RandomState(int(time.time() * 1000) & 0xFFFF)

        for _ in range(self.N_EPOCHS):
            perm = rng_np.permutation(n_trn)
            for s in range(0, n_trn - self.MB_GAMES + 1, self.MB_GAMES):
                mb = perm[s:s + self.MB_GAMES]
                self._params, self._opt_state, _ = self._train_step_jit(
                    self._params, self._opt_state, emb_t[mb], y_t[mb], m_t[mb])
        self._fitted = True

        def _r2(pred, y, m):
            n  = m.sum() + 1e-8
            mu = (y * m).sum() / n
            return float(1.0 - ((y - pred) ** 2 * m).sum() / (((y - mu) ** 2 * m).sum() + 1e-8))

        r2_trn = _r2(self._predict_jit(self._params, emb_t), y_t, m_t)
        r2_val = _r2(self._predict_jit(self._params, emb[idx_val]),
                     return_targets[idx_val], mask[idx_val])
        print(f'  [s5] R²_trn={r2_trn:.3f}  R²_val={r2_val:.3f}', flush=True)
        return r2_val, 0.0, time.time() - t0


class _OldSuperESN:
    """D_max-layer reservoir with PCA inter-layer encoders.

    CMA-ES parameter vector layout (raw[5]): SR is fixed, not tuned.
      0  leak_rate:    sigmoid(raw)             → (0, 1)
      1  log_beta:     raw clamped (-10, 3)     → exp → ridge regularisation
      2  washout_frac: sigmoid(raw)*0.15        → fraction of T to discard
      3  n_layers:     sigmoid(raw)*(D_max-1)+1 → rounded int in [1, D_max]
      4  width_frac:   sigmoid(raw)*W_max       → rounded int, min 8
    """
    N_CMA_DIMS = 5

    def __init__(self, n_in, d_max, w_max, enc_dim, sparsity=0.1, seed=0):
        self.n_in    = n_in
        self.d_max   = d_max
        self.w_max   = w_max
        self.enc_dim = enc_dim
        self.D_total = n_in + d_max * enc_dim   # readout uses PCA-compressed states

        rng = np.random.RandomState(seed)

        # Layer 0 input: n_in → w_max
        self.W_in0 = jnp.array(
            rng.uniform(-1, 1, (w_max, n_in)).astype(np.float32))

        # Deeper layer inputs: enc_dim → w_max (receives PCA-encoded prev state)
        self.W_in_deep = jnp.array(
            rng.uniform(-1, 1, (d_max - 1, w_max, enc_dim)).astype(np.float32))

        # Recurrent weights, spectral radius normalised to 1 at init
        W_res_all = np.zeros((d_max, w_max, w_max), dtype=np.float32)
        for i in range(d_max):
            W = rng.uniform(-0.5, 0.5, (w_max, w_max)) * (rng.rand(w_max, w_max) < sparsity)
            rho = np.max(np.abs(np.linalg.eigvals(W)))
            W_res_all[i] = (W / rho if rho > 0 else W).astype(np.float32)
        self.W_res_all = jnp.array(W_res_all)

        # CMA-ES best state
        self.best_raw   = jnp.zeros(self.N_CMA_DIMS)
        self.best_W_out = None

        # PCA encoders: one per layer (inter-layer for 0..D_max-2, readout for all D_max)
        W_enc_init = np.zeros((d_max, enc_dim, w_max), dtype=np.float32)
        for i in range(d_max):
            W_enc_init[i, :, :enc_dim] = np.eye(enc_dim)
        self.best_W_enc = jnp.array(W_enc_init)   # [D_max, enc_dim, W_max]

    def decode_hyperparams(self, raw):
        """Decode raw CMA-ES vector to human-readable hyperparams dict."""
        def sig(x): return 1.0 / (1.0 + np.exp(-float(x)))
        raw = np.array(raw)
        return dict(
            sr      = CFG.esn_fixed_sr,
            leak    = sig(raw[0]),
            beta    = np.exp(np.clip(raw[1], -10.0, 3.0)),
            wo_frac = sig(raw[2]) * 0.15,
            n_layers= round(sig(raw[3]) * (self.d_max - 1) + 1),
            width   = max(round(sig(raw[4]) * self.w_max), 8),
        )

    def make_jit_fns(self):
        W_in0     = self.W_in0
        W_in_deep = self.W_in_deep
        W_res_all = self.W_res_all
        d_max     = self.d_max
        w_max     = self.w_max
        enc_dim   = self.enc_dim
        D_total   = self.D_total

        fixed_sr = CFG.esn_fixed_sr
        def _decode(raw):
            sig      = jax.nn.sigmoid
            sr       = fixed_sr                                  # fixed, not tuned
            leak     = sig(raw[0])
            beta     = jnp.exp(jnp.clip(raw[1], -10.0, 3.0))   # range [~0, 20.1]
            wo_frac  = sig(raw[2]) * 0.15
            n_layers = jnp.round(sig(raw[3]) * (d_max - 1) + 1).astype(jnp.int32)
            width    = jnp.maximum(jnp.round(sig(raw[4]) * w_max).astype(jnp.int32), 8)
            return sr, leak, beta, wo_frac, n_layers, width

        def _forward_compact(emb, raw, W_enc):
            """emb [G,T,n_in], W_enc [D_max,enc_dim,W_max] → (M [G,T,D_total], washout).
            parts = [emb, enc0, enc1, ...] where enc_i = s_i @ W_enc[i].T (PCA-compressed).
            Same W_enc[i] feeds into layer i+1 (inter-layer) and into the readout."""
            sr, leak, beta, wo_frac, n_layers, width = _decode(raw)
            T          = emb.shape[1]
            washout    = jnp.round(wo_frac * T).astype(jnp.int32)
            unit_mask  = (jnp.arange(w_max) < width).astype(jnp.float32)
            layer_mask = (jnp.arange(d_max) < n_layers).astype(jnp.float32)

            W_r0 = W_res_all[0] * sr
            def _seq0(inp):
                def _step(h, u):
                    h2 = (1.-leak)*h + leak*jnp.tanh(W_r0@h + W_in0@u)
                    return h2*unit_mask, h2*unit_mask
                _, s = jax.lax.scan(_step, jnp.zeros(w_max), inp)
                return s
            s    = jax.vmap(_seq0)(emb) * layer_mask[0]  # [G,T,W_max]
            enc  = s @ W_enc[0].T * layer_mask[0]         # [G,T,enc_dim]
            parts = [emb, enc]
            x = enc                                        # inter-layer input to layer 1

            for i in range(1, d_max):
                W_r = W_res_all[i] * sr
                W_i = W_in_deep[i - 1]
                def _seq_i(inp, W_r=W_r, W_i=W_i):
                    def _step(h, u):
                        h2 = (1.-leak)*h + leak*jnp.tanh(W_r@h + W_i@u)
                        return h2*unit_mask, h2*unit_mask
                    _, s = jax.lax.scan(_step, jnp.zeros(w_max), inp)
                    return s
                s   = jax.vmap(_seq_i)(x) * layer_mask[i]
                enc = s @ W_enc[i].T * layer_mask[i]      # [G,T,enc_dim]
                parts.append(enc)
                x = enc                                    # inter-layer input to layer i+1

            return jnp.concatenate(parts, axis=-1), washout

        def _forward_full(emb, raw, W_enc):
            """Like _forward_compact but also returns per-layer states [G,T,D_max,W_max]."""
            sr, leak, beta, wo_frac, n_layers, width = _decode(raw)
            T          = emb.shape[1]
            washout    = jnp.round(wo_frac * T).astype(jnp.int32)
            unit_mask  = (jnp.arange(w_max) < width).astype(jnp.float32)
            layer_mask = (jnp.arange(d_max) < n_layers).astype(jnp.float32)

            W_r0 = W_res_all[0] * sr
            def _seq0(inp):
                def _step(h, u):
                    h2 = (1.-leak)*h + leak*jnp.tanh(W_r0@h + W_in0@u)
                    return h2*unit_mask, h2*unit_mask
                _, s = jax.lax.scan(_step, jnp.zeros(w_max), inp)
                return s
            s0   = jax.vmap(_seq0)(emb) * layer_mask[0]
            enc0 = s0 @ W_enc[0].T * layer_mask[0]
            parts        = [emb, enc0]
            layer_states = [s0]               # raw states for PCA refit
            x = enc0

            for i in range(1, d_max):
                W_r = W_res_all[i] * sr
                W_i = W_in_deep[i - 1]
                def _seq_i(inp, W_r=W_r, W_i=W_i):
                    def _step(h, u):
                        h2 = (1.-leak)*h + leak*jnp.tanh(W_r@h + W_i@u)
                        return h2*unit_mask, h2*unit_mask
                    _, s = jax.lax.scan(_step, jnp.zeros(w_max), inp)
                    return s
                si   = jax.vmap(_seq_i)(x) * layer_mask[i]
                enci = si @ W_enc[i].T * layer_mask[i]
                parts.append(enci)
                layer_states.append(si)       # raw states for PCA refit
                x = enci

            M              = jnp.concatenate(parts, axis=-1)
            states_stacked = jnp.stack(layer_states, axis=2)  # [G,T,D_max,W_max]
            return M, states_stacked, washout

        @jax.jit
        def eval_candidate(raw, W_enc, emb, lengths, outcomes):
            """Fit W_out on 80% of games, return (W_out, r2_val, r2_train).
            CMA-ES selects by r2_val so it cannot overfit to the training split."""
            G, T   = emb.shape[:2]
            G_tr   = G * 4 // 5          # 80% train, 20% val (static at JIT time)
            G_val  = G - G_tr

            M, washout = _forward_compact(emb, raw, W_enc)

            t_idx = jnp.arange(T)
            mask  = (t_idx[None,:] >= washout) & (t_idx[None,:] < lengths[:,None])
            M_f   = M.reshape(G * T, D_total)
            beta  = jnp.exp(jnp.clip(raw[2], -10.0, 1.0))

            # ── Fit on train split ────────────────────────────────────────────
            mask_tr = mask[:G_tr]                           # [G_tr, T]
            w_tr    = mask_tr.reshape(G_tr * T).astype(jnp.float32)
            M_f_tr  = M_f[:G_tr * T]
            y_tr    = jnp.repeat(outcomes[:G_tr], T)
            MW_tr   = M_f_tr * w_tr[:, None]
            A       = MW_tr.T @ M_f_tr + beta * jnp.eye(D_total)
            b       = MW_tr.T @ y_tr
            W_out   = jnp.linalg.solve(A, b)

            def _game_r2(M_split, mask_split, out_split, n_g):
                preds_t = M_split @ W_out
                preds_g = (preds_t.reshape(n_g, T) * mask_split).sum(1) / (
                    mask_split.sum(1).astype(jnp.float32) + 1e-8)
                ss_res = ((out_split - preds_g) ** 2).sum()
                ss_tot = ((out_split - out_split.mean()) ** 2).sum()
                return 1.0 - ss_res / (ss_tot + 1e-8)

            # ── Val R² (fitness for CMA-ES) ───────────────────────────────────
            r2_val   = _game_r2(M_f[G_tr * T:], mask[G_tr:], outcomes[G_tr:], G_val)
            # ── Train R² (for logging only) ───────────────────────────────────
            r2_train = _game_r2(M_f_tr, mask_tr, outcomes[:G_tr], G_tr)

            return W_out, r2_val, r2_train

        # Sequential scan over CMA-ES candidates instead of vmap: each candidate's
        # M matrix [G,T,D_total] is allocated and freed before the next, avoiding
        # the ~13GB peak that vmap causes by materialising all popsize at once.
        @jax.jit
        def eval_population(population, W_enc, emb, lengths, outcomes):
            def step(_, raw):
                return None, eval_candidate(raw, W_enc, emb, lengths, outcomes)
            _, (all_W_out, all_r2_val, all_r2_train) = jax.lax.scan(
                step, None, population)
            return all_W_out, all_r2_val, all_r2_train

        @jax.jit
        def refit_encoders(states_stacked):
            """PCA refit: states [G,T,D_max,W_max] → W_enc [D_max,enc_dim,W_max]."""
            G, T = states_stacked.shape[:2]
            # All D_max layers — each gets an encoder for readout + inter-layer
            S = (states_stacked.transpose(2, 0, 1, 3)
                               .reshape(d_max, G * T, w_max))
            def fit_one(s):
                C = s.T @ s                              # [W_max, W_max] gram
                _, eigvecs = jnp.linalg.eigh(C)          # ascending eigenvalues
                return eigvecs[:, -enc_dim:].T            # [enc_dim, W_max] top vecs
            return jax.vmap(fit_one)(S)                  # [D_max, enc_dim, W_max]

        @jax.jit
        def rls_update(P, t, M_f, w, y, beta):
            """Recursive least squares with forgetting factor λ=0.95."""
            lam   = 0.95
            MW    = M_f * w[:, None]
            P_new = lam * P + MW.T @ M_f
            t_new = lam * t + MW.T @ y
            W_out = jnp.linalg.solve(P_new + beta * jnp.eye(D_total), t_new)
            return P_new, t_new, W_out

        forward_for_best = jax.jit(_forward_full)

        @jax.jit
        def predict(W_out, raw, W_enc, emb, lengths):
            """Returns [G,T] values clipped to [0,1]."""
            G, T  = emb.shape[:2]
            M, wo = _forward_compact(emb, raw, W_enc)
            v     = (M @ W_out).clip(0.0, 1.0)
            v_ws  = v[:, wo]
            v     = jnp.where(jnp.arange(T)[None,:] < wo, v_ws[:,None], v)
            return v

        return eval_population, refit_encoders, rls_update, forward_for_best, predict


class _OldESNCritic:
    """SuperESN + evosax CMA-ES with PCA inter-layer encoders and RLS readout.

    Each iteration:
      1. CMA-ES population evaluated in parallel (vmapped GPU call).
      2. Best candidate's layer states used to refit PCA encoders (eigh).
      3. W_out updated via RLS (λ=0.95) so the readout accumulates history.
    """
    def __init__(self, cma_rng_seed=1):
        from evosax.algorithms import CMA_ES   # evosax ≥0.2.0
        self.esn = SuperESN(
            CFG.hidden_dim, CFG.esn_max_depth, CFG.esn_max_width, CFG.esn_enc_dim)
        (self._eval_pop, self._refit_enc, self._rls_update,
         self._forward_for_best, self._predict) = self.esn.make_jit_fns()

        _dummy          = jnp.zeros(_OldSuperESN.N_CMA_DIMS)
        strategy        = CMA_ES(population_size=CFG.esn_cma_popsize, solution=_dummy)
        es_params       = strategy.default_params
        es_state        = strategy.init(jr.PRNGKey(cma_rng_seed), _dummy, es_params)
        self._strategy  = strategy
        self._es_params = es_params
        self._es_state  = es_state
        self._cma_rng   = jr.PRNGKey(cma_rng_seed + 1)

        # RLS state: P = large-ridge prior → W_out=0 until first data arrives
        D = self.esn.D_total
        self._rls_P = jnp.eye(D) * 1e3
        self._rls_t = jnp.zeros(D)

        self._iter = 0

    def update(self, emb_padded, lengths, outcomes):
        t0      = time.time()
        run_cma = (self._iter % CFG.esn_cma_every == 0)

        if run_cma:
            # ── CMA-ES population eval (all candidates in one vmapped call) ──
            self._cma_rng, ask_key, tell_key = jr.split(self._cma_rng, 3)
            population, self._es_state = self._strategy.ask(
                ask_key, self._es_state, self._es_params)

            all_W_out, all_r2_val, all_r2_train = self._eval_pop(
                population, self.esn.best_W_enc, emb_padded, lengths, outcomes)

            self._es_state, _ = self._strategy.tell(
                tell_key, population, -all_r2_val, self._es_state, self._es_params)

            best_i    = jnp.argmax(all_r2_val)      # stays on device
            best_raw  = population[best_i]
            cma_r2_val   = float(all_r2_val[best_i])    # one intentional sync for logging
            cma_r2_train = float(all_r2_train[best_i])
            self.esn.best_raw = best_raw

            # ── Forward pass for best candidate to get layer states ──────────
            G, T = emb_padded.shape[:2]
            M, states, washout = self._forward_for_best(
                emb_padded, best_raw, self.esn.best_W_enc)

            t_idx = jnp.arange(T)
            mask  = (t_idx[None,:] >= washout) & (t_idx[None,:] < lengths[:,None])
            w     = mask.reshape(G * T).astype(jnp.float32)
            y     = jnp.repeat(outcomes, T)
            M_f   = M.reshape(G * T, self.esn.D_total)
            beta  = jnp.exp(jnp.clip(best_raw[2], -10.0, 1.0))

            # ── RLS readout update (accumulates across iterations) ───────────
            self._rls_P, self._rls_t, W_out_rls = self._rls_update(
                self._rls_P, self._rls_t, M_f, w, y, beta)
            self.esn.best_W_out = W_out_rls

            # ── PCA encoder refit from best candidate's layer states ─────────
            self.esn.best_W_enc = self._refit_enc(states)

            # ── Hyperparameter readout ────────────────────────────────────────
            hp = self.esn.decode_hyperparams(best_raw)
            overfit_gap = cma_r2_train - cma_r2_val
            print(
                f'  ESN  r2_val={cma_r2_val:.3f}  r2_train={cma_r2_train:.3f}'
                f'  gap={overfit_gap:+.3f}  '
                f'sr={hp["sr"]:.2f}  leak={hp["leak"]:.2f}  '
                f'beta={hp["beta"]:.1e}  '
                f'layers={hp["n_layers"]}  width={hp["width"]}  '
                f'wo={hp["wo_frac"]:.3f}',
                flush=True)

            esn_r2 = cma_r2_val
        else:
            esn_r2 = float('nan')

        self._iter += 1
        return time.time() - t0, esn_r2, run_cma

    def values(self, emb_padded, lengths):
        """Returns [G,T] jnp value array. Constant 0.5 until first CMA-ES eval."""
        if self.esn.best_W_out is None:
            return jnp.full(emb_padded.shape[:2], 0.5)
        return self._predict(
            self.esn.best_W_out, self.esn.best_raw, self.esn.best_W_enc,
            emb_padded, lengths)


# ── ELO Hall of Fame ──────────────────────────────────────────────────────────
class EloHoF:
    """Fixed-capacity Hall of Fame. Entries sorted by ELO descending."""

    def __init__(self, capacity=CFG.hof_size, init_elo=CFG.hof_init_elo):
        self.capacity  = capacity
        self.init_elo  = init_elo
        self.entries   = []  # list of {'params': np_dict, 'elo': float, 'tag': str}

    def add(self, np_params, tag='policy', elo=None):
        """Add entry; evict lowest-ELO member if at capacity."""
        entry = {'params': np_params, 'elo': elo if elo is not None else self.init_elo, 'tag': tag}
        self.entries.append(entry)
        self.entries.sort(key=lambda e: e['elo'], reverse=True)
        if len(self.entries) > self.capacity:
            self.entries = self.entries[:self.capacity]

    def sample_opponent(self, rng: np.random.Generator):
        """70% ELO-weighted + 30% uniform. Returns (idx, params)."""
        if not self.entries:
            return None, None
        if len(self.entries) == 1 or rng.random() > CFG.hof_elo_weight:
            idx = int(rng.integers(len(self.entries)))
        else:
            elos   = np.array([e['elo'] for e in self.entries], dtype=np.float64)
            elos  -= elos.min() - 1.0
            probs  = elos / elos.sum()
            idx    = int(rng.choice(len(self.entries), p=probs))
        return idx, self.entries[idx]['params']

    def update_elo(self, opp_idx, outcome, current_elo, k):
        """
        outcome: win_rate from current agent's POV (1=always win, 0=always lose).
        Updates opponent entry ELO in-place; caller manages current_elo separately.
        """
        if opp_idx >= len(self.entries):
            return
        opp_elo  = self.entries[opp_idx]['elo']
        expected = 1.0 / (1.0 + 10 ** ((opp_elo - current_elo) / 400.0))
        self.entries[opp_idx]['elo'] += k * (1.0 - outcome - expected)
        self.entries.sort(key=lambda e: e['elo'], reverse=True)

    def summary(self):
        lines = [f'  HoF ({len(self.entries)} entries):']
        for i, e in enumerate(self.entries):
            lines.append(f'    [{i}] {e["tag"]:20s}  ELO={e["elo"]:.0f}')
        return '\n'.join(lines)


def sample_opp_per_device(hof, rng, n_devices, fallback_params_np):
    """Sample n_devices (possibly different) HoF opponents, one per TPU device.
    Returns (indices: list[int], stacked_params: pytree with leading dim n_devices).
    """
    indices, np_list = [], []
    for _ in range(n_devices):
        idx, np_p = hof.sample_opponent(rng)
        if np_p is None:
            idx, np_p = 0, fallback_params_np
        indices.append(idx)
        np_list.append(np_p)
    stacked = jax.tree_util.tree_map(
        lambda *arrs: jnp.stack(arrs), *[np_to_params(p) for p in np_list])
    return indices, stacked


# ── Rollout collection (lax.scan, single device) ─────────────────────────────
def make_rollout_fn(actor_graph, num_players=2):
    """Return a JIT-compiled rollout fn for one device's game batch.

    num_players=2 → 1v1 duel; num_players=4 → 4-way FFA (3 HoF opponents).
    Reward: per-step exponentially time-weighted linear rank, normalised to [0,1].
    """
    vmap_fwd  = jax.vmap(make_forward(actor_graph), in_axes=(None, 0, 0, 0, 0))
    vmap_enc  = jax.vmap(make_encode(actor_graph),  in_axes=(None, 0, 0, 0))
    vmap_step = jax.vmap(env_lib.step, in_axes=(0, 0, 0, None))

    # Precompute per-step weights and normalisation constant (scalar)
    _ts          = jnp.arange(CFG.max_steps, dtype=jnp.float32)
    _weights     = jnp.exp(CFG.reward_alpha * _ts / CFG.max_steps)
    _reward_norm = 1.0 / _weights.sum()   # so max cumulative reward == 1.0

    @jax.jit
    def rollout(actor_params, opp_params, keys, noise_key):
        B          = keys.shape[0]
        noise_keys = jr.split(noise_key, CFG.max_steps)

        vmap_setup = jax.vmap(env_lib.setup, in_axes=(0, None))
        init_state, env_params = vmap_setup(keys, num_players)

        def scan_step(carry, step_idx):
            state, game_done = carry
            active = ~game_done

            # ── Player 0: actor with Gaussian logit noise ──────────────────
            p0, f0, pm0, fm0 = build_env_arrays(state, env_params, 0, num_players)
            logits_p0  = vmap_fwd(actor_params, p0, f0, pm0, fm0)
            noise      = jr.normal(noise_keys[step_idx], logits_p0.shape) * CFG.logit_sigma
            noisy_p0   = logits_p0 + noise
            emb_p0     = vmap_enc(actor_params, p0, f0, fm0)
            owned_p0   = (state.planet_owners == 0)
            lp_per_dim = -0.5 * (noise / CFG.logit_sigma) ** 2
            old_lp     = (lp_per_dim.sum(-1) * owned_p0).sum(-1)
            ships_p0, ds_angle_p0 = logits_to_action(noisy_p0, state.planet_ships.astype(jnp.float32))

            # ── Player 1: opponent ─────────────────────────────────────────
            p1, f1, pm1, fm1 = build_env_arrays(state, env_params, 1, num_players)
            logits_p1  = vmap_fwd(opp_params, p1, f1, pm1, fm1)
            ships_p1, ds_angle_p1 = logits_to_action(logits_p1, state.planet_ships.astype(jnp.float32))
            # ds_angle is in the player's rotated observation frame; de-rotate to global.
            ds_angle_p1 = ds_angle_p1 + 1.0 * (2.0 * jnp.pi / num_players)
            emb_p1     = vmap_enc(opp_params, p1, f1, fm1)   # captured for ESN opponent data

            # ── Players 2, 3 (4p FFA only — resolved at trace time) ────────
            if num_players == 4:
                p2, f2, pm2, fm2 = build_env_arrays(state, env_params, 2, num_players)
                p3, f3, pm3, fm3 = build_env_arrays(state, env_params, 3, num_players)
                logits_p2  = vmap_fwd(opp_params, p2, f2, pm2, fm2)
                logits_p3  = vmap_fwd(opp_params, p3, f3, pm3, fm3)
                ships_p2, ds_angle_p2 = logits_to_action(logits_p2, state.planet_ships.astype(jnp.float32))
                ships_p3, ds_angle_p3 = logits_to_action(logits_p3, state.planet_ships.astype(jnp.float32))
                ds_angle_p2 = ds_angle_p2 + 2.0 * (2.0 * jnp.pi / 4)
                ds_angle_p3 = ds_angle_p3 + 3.0 * (2.0 * jnp.pi / 4)

            # ── Mask inactive body targets ─────────────────────────────────
            ages         = state.step[..., None] - env_params.comet_spawn_steps
            comet_active = env_params.is_comet & (ages >= 0) & (ages < env_params.body_lifespans)
            tgt_active   = env_params.is_static_planet | env_params.is_orbiting_planet | comet_active
            tgt_mask_64  = jnp.concatenate(
                [tgt_active, jnp.ones((*tgt_active.shape[:-1], 4), dtype=bool)], axis=-1)

            ships_p0 = jnp.where(tgt_mask_64[:, None, :], ships_p0, 0.0)
            ships_p1 = jnp.where(tgt_mask_64[:, None, :], ships_p1, 0.0)

            # ── Intercept angles ───────────────────────────────────────────
            intercept_p0 = calculate_intercept_angle(state, env_params, ships_p0[..., :60])
            intercept_p1 = calculate_intercept_angle(state, env_params, ships_p1[..., :60])
            angles_p0    = jnp.concatenate([intercept_p0, ds_angle_p0], axis=-1)
            angles_p1    = jnp.concatenate([intercept_p1, ds_angle_p1], axis=-1)

            ships_p0 = jnp.where(ships_p0 < 1.0, 0.0, ships_p0).astype(jnp.int32)
            ships_p1 = jnp.where(ships_p1 < 1.0, 0.0, ships_p1).astype(jnp.int32)

            is_p0 = (state.planet_owners == 0)[..., None]
            is_p1 = (state.planet_owners == 1)[..., None]

            if num_players == 2:
                combined_ships  = jnp.where(is_p0, ships_p0, jnp.where(is_p1, ships_p1, 0))
                combined_angles = jnp.where(is_p0, angles_p0, jnp.where(is_p1, angles_p1, 0.0))
            else:
                ships_p2 = jnp.where(tgt_mask_64[:, None, :], ships_p2, 0.0)
                ships_p3 = jnp.where(tgt_mask_64[:, None, :], ships_p3, 0.0)
                intercept_p2 = calculate_intercept_angle(state, env_params, ships_p2[..., :60])
                intercept_p3 = calculate_intercept_angle(state, env_params, ships_p3[..., :60])
                angles_p2 = jnp.concatenate([intercept_p2, ds_angle_p2], axis=-1)
                angles_p3 = jnp.concatenate([intercept_p3, ds_angle_p3], axis=-1)
                ships_p2  = jnp.where(ships_p2 < 1.0, 0.0, ships_p2).astype(jnp.int32)
                ships_p3  = jnp.where(ships_p3 < 1.0, 0.0, ships_p3).astype(jnp.int32)
                is_p2     = (state.planet_owners == 2)[..., None]
                is_p3     = (state.planet_owners == 3)[..., None]
                combined_ships  = jnp.where(is_p0, ships_p0,
                                  jnp.where(is_p1, ships_p1,
                                  jnp.where(is_p2, ships_p2,
                                  jnp.where(is_p3, ships_p3, 0))))
                combined_angles = jnp.where(is_p0, angles_p0,
                                  jnp.where(is_p1, angles_p1,
                                  jnp.where(is_p2, angles_p2,
                                  jnp.where(is_p3, angles_p3, 0.0))))

            action = env_lib.EnvAction(ships=combined_ships, angle=combined_angles)
            next_state, scores, _wasted, terminated = vmap_step(state, env_params, action, num_players)
            new_done = game_done | terminated

            # ── Per-step exponentially weighted rank reward ─────────────────
            w      = _weights[step_idx]
            owners = next_state.planet_owners
            own_pl = (owners == 0).sum(-1).astype(jnp.float32)   # [B]
            if num_players == 2:
                opp_pl = (owners == 1).sum(-1).astype(jnp.float32)
                rank_r = jnp.where(own_pl > opp_pl, 1.0,
                         jnp.where(own_pl == opp_pl, 0.5, 0.0))
            else:
                pl_counts = jnp.stack(
                    [(owners == p).sum(-1).astype(jnp.float32) for p in range(4)])  # [4, B]
                rank_p0 = 1.0 + (pl_counts > pl_counts[0:1]).sum(0).astype(jnp.float32)
                rank_r  = (4.0 - rank_p0) / 3.0                 # [B] in {0, 1/3, 2/3, 1}

            shaping = w * rank_r * _reward_norm                   # [B] small per-step signal

            # Terminal win/loss bonus — emitted only at the exact termination step
            if num_players == 2:
                win_bonus = jnp.where(own_pl > opp_pl, 1.0,
                            jnp.where(own_pl == opp_pl, 0.0, -1.0))
            else:
                win_bonus = (4.0 - rank_p0 * 2.0) / 3.0   # [B] in {-1, -1/3, 1/3, 1}
            # Only emit at the first terminal step (not if already done)
            terminal_bonus = jnp.where(terminated & ~game_done, win_bonus, 0.0)

            step_reward = shaping + terminal_bonus             # [B]

            step_out = {
                'planets':        p0,
                'fleets':         f0,
                'pmask':          pm0,
                'fmask':          fm0,
                'old_log_prob':   old_lp,
                'noisy_logits':   noisy_p0.astype(jnp.bfloat16),
                'embeddings':     emb_p0,
                'opp_embeddings': emb_p1,
                'owned_p0':       owned_p0,
                'done':           new_done,
                'scores':         scores,
                'active':         active,
                'reward':         step_reward,
            }
            return (next_state, new_done), step_out

        init_done = jnp.zeros(B, dtype=bool)
        _, traj = jax.lax.scan(scan_step, (init_state, init_done), jnp.arange(CFG.max_steps))
        return traj

    return rollout


# ── Multi-device rollout (pmap) ───────────────────────────────────────────────
def make_pmap_rollout(actor_graph, num_players=2):
    """Broadcast opp_params across devices (for exploiter / single-opp use)."""
    single_rollout = make_rollout_fn(actor_graph, num_players)
    return jax.pmap(single_rollout, axis_name='devices',
                    in_axes=(None, None, 0, 0),
                    static_broadcasted_argnums=())

def make_pmap_rollout_multi_opp(actor_graph, num_players=2):
    """Per-device opp_params — each device plays against a different HoF opponent."""
    single_rollout = make_rollout_fn(actor_graph, num_players)
    return jax.pmap(single_rollout, axis_name='devices',
                    in_axes=(None, 0, 0, 0),
                    static_broadcasted_argnums=())


# ── GAE — pure JAX, reverse lax.scan ─────────────────────────────────────────
@functools.partial(jax.jit, static_argnames=('gamma', 'lam'))
def compute_gae_jax(rewards, dones, active, values,
                    gamma=CFG.gamma, lam=CFG.gae_lambda):
    """All inputs [T, B] jnp. Returns advantages [T, B], returns [T, B]."""
    T, B = rewards.shape

    def _step(carry, t_rev):
        gae, val_next = carry
        t    = T - 1 - t_rev
        r    = rewards[t]
        d    = dones[t].astype(jnp.float32)
        act  = active[t].astype(jnp.float32)
        v_t  = values[t]
        delta = r + gamma * val_next * (1.0 - d) - v_t
        new_gae = (delta + gamma * lam * (1.0 - d) * gae) * act
        return (new_gae, v_t), new_gae

    _, adv_rev = jax.lax.scan(_step, (jnp.zeros(B), jnp.zeros(B)), jnp.arange(T))
    advantages = adv_rev[::-1]
    return advantages, advantages + values


# ── PPO — GAE + epochs + minibatches all in one jit (lax.scan) ───────────────
def make_ppo_train_fn(actor_graph, optimizer):
    vmap_fwd = jax.vmap(make_forward(actor_graph), in_axes=(None, 0, 0, 0, 0))

    def _loss(params, planets, fleets, pmask, fmask,
              noisy_logits, owned_p0, adv_norm, old_lp, act_w):
        new_logits = vmap_fwd(params, planets, fleets, pmask, fmask)
        noisy_f32  = noisy_logits.astype(jnp.float32)
        sigma = CFG.logit_sigma
        lp_all    = -0.5 * ((noisy_f32 - new_logits) / sigma) ** 2  # [MB, 60, 73]
        lp        = (lp_all.sum(-1) * owned_p0).sum(-1)              # [MB]
        ratio     = jnp.exp(lp - old_lp)
        pg1       = -ratio * adv_norm
        pg2       = -jnp.clip(ratio, 1 - CFG.clip_eps, 1 + CFG.clip_eps) * adv_norm
        n_act     = act_w.sum() + 1e-8
        pg_loss   = (jnp.maximum(pg1, pg2) * act_w).sum() / n_act
        dest_p    = jax.nn.softmax(new_logits[..., 1:65], axis=-1)
        entropy   = -(dest_p * jnp.log(dest_p + 1e-8)).sum(-1)
        ent_loss  = -(entropy * owned_p0 * act_w[:, None]).sum() / n_act
        # per-planet entropy: normalize by mean n_owned so cap/floor are scale-invariant
        n_owned_wt    = (owned_p0.sum(-1) * act_w).sum() + 1e-8
        per_planet_ent = (entropy * owned_p0 * act_w[:, None]).sum() / n_owned_wt
        ent_cap   = CFG.ent_max_coef * jnp.maximum(0.0, per_planet_ent - CFG.ent_target)
        ent_floor = CFG.ent_max_coef * jnp.maximum(0.0, CFG.ent_floor - per_planet_ent)
        return pg_loss + CFG.entropy_coef * ent_loss + ent_cap + ent_floor, (pg_loss, per_planet_ent)

    @functools.partial(jax.jit, donate_argnums=(0, 1))
    def ppo_train(params, opt_state,
                  planets, fleets, pmask, fmask,
                  noisy_logits, owned_p0, old_lp,
                  advantages, active, rng_key):
        """Full PPO: ppo_epochs × n_minibatches gradient steps, all in one JIT.
        All inputs are [T, B, ...] jnp arrays (no CPU involved).
        """
        T, B = advantages.shape
        N    = T * B
        n_mb = N // CFG.minibatch_size
        N_use = n_mb * CFG.minibatch_size

        # Global advantage normalisation over active transitions
        act_f = active.reshape(N).astype(jnp.float32)
        adv_f = advantages.reshape(N)
        n_act = act_f.sum() + 1e-8
        mu    = (adv_f * act_f).sum() / n_act
        sig   = jnp.sqrt(((adv_f - mu) ** 2 * act_f).sum() / n_act + 1e-8)
        adv_n = (adv_f - mu) / sig * act_f   # [N] normalised, inactive=0

        # Flat views  [N, ...]
        flat = lambda x: x.reshape(N, *x.shape[2:])
        pl_f = flat(planets);    fl_f = flat(fleets)
        pm_f = flat(pmask);      fm_f = flat(fmask)
        nl_f = flat(noisy_logits); op_f = flat(owned_p0)
        lp_f = flat(old_lp)

        def epoch_fn(carry, _):
            params, opt_state, key = carry
            key, sk = jax.random.split(key)
            perm = jax.random.permutation(sk, N)[:N_use]

            mb = lambda x: x[perm].reshape(n_mb, CFG.minibatch_size, *x.shape[1:])
            xs = (mb(pl_f), mb(fl_f), mb(pm_f), mb(fm_f),
                  mb(nl_f), mb(op_f), adv_n[perm].reshape(n_mb, CFG.minibatch_size),
                  mb(lp_f), act_f[perm].reshape(n_mb, CFG.minibatch_size))

            def mb_step(carry, x):
                params, opt_state = carry
                pl, fl, pm, fm, nl, op, adv, lp, aw = x
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    params, pl, fl, pm, fm, nl, op, adv, lp, aw)
                grads, _ = optax.clip_by_global_norm(CFG.max_grad_norm).update(grads, None)
                updates, new_opt = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), new_opt), (loss, aux)

            (params, opt_state), losses = jax.lax.scan(mb_step, (params, opt_state), xs)
            return (params, opt_state, key), losses

        (params, opt_state, _), losses = jax.lax.scan(
            epoch_fn, (params, opt_state, rng_key), None, length=CFG.ppo_epochs)
        return params, opt_state, losses

    return ppo_train


# ── Episode outcome extraction — pure JAX ────────────────────────────────────
def extract_outcomes_jax(traj, num_players):
    """traj: dict of jnp [T, B, ...]. Returns (lengths [B], outcomes [B], win_rate scalar)."""
    T, B = traj['done'].shape
    done = traj['done']   # [T, B] bool; once True, stays True

    # Length = first done step + 1, else T
    any_done = done.any(axis=0)                            # [B]
    first_done = jnp.argmax(done.astype(jnp.int32), axis=0)   # [B]
    lengths = jnp.where(any_done, first_done + 1, T).astype(jnp.int32)  # [B]

    # Final score at last active step
    end_idx = (lengths - 1).clip(0, T - 1)                # [B]
    sc = traj['scores']                                    # [T, B, n_players]
    final_sc = sc[end_idx, jnp.arange(B)]                 # [B, n_players]

    if num_players == 2:
        won     = (final_sc[:, 0] > final_sc[:, 1]).astype(jnp.float32)   # [B]
        outcomes = won
        win_rate = won.mean()
    else:
        # rank of player 0 (0=1st … 3=4th)
        rank0 = (final_sc > final_sc[:, :1]).sum(-1).astype(jnp.float32)  # [B]
        outcomes = (3.0 - rank0) / 3.0                    # [B] ∈ {0,1/3,2/3,1}
        win_rate = (rank0 == 0).astype(jnp.float32).mean()

    return lengths, outcomes, win_rate


# ── Main training loop ────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',   default='bc_v14bc/weights_bc_v14.pkl')
    parser.add_argument('--local-weights', action='store_true')
    parser.add_argument('--resume',    default=None, help='Local path to RL checkpoint .pkl')
    parser.add_argument('--n-iters',   type=int, default=10_000)
    parser.add_argument('--games-per-device', type=int, default=None)
    parser.add_argument('--out-dir',   default=None, help='Override CFG.out_dir for checkpoints')
    parser.add_argument('--r2-prefix', default=None, help='Override CFG.r2_prefix for R2 sync')
    parser.add_argument('--cpu',       action='store_true')
    args = parser.parse_args()

    if args.games_per_device:
        CFG.games_per_device = args.games_per_device
    if args.out_dir:
        CFG.out_dir = args.out_dir
    if args.r2_prefix:
        CFG.r2_prefix = args.r2_prefix

    os.makedirs(CFG.out_dir, exist_ok=True)
    n_devices = jax.device_count()
    print('train_rl.py  version: v11.24 (per-planet entropy cap/floor + per-device HoF opps)', flush=True)
    print(f'Devices: {n_devices}  ({jax.devices()})')
    print(f'Games per device: {CFG.games_per_device}  Total: {n_devices * CFG.games_per_device}')

    # ── Load BC weights ────────────────────────────────────────────────────────
    if args.local_weights or os.path.exists(args.weights):
        with open(args.weights, 'rb') as f:
            raw = pickle.load(f)
    else:
        import boto3
        s3 = boto3.client('s3',
            endpoint_url=os.environ['R2_ENDPOINT_URL'],
            aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
            region_name='auto')
        tmp = '/tmp/weights_rl_init.pkl'
        print(f'Downloading {args.weights} from R2...')
        s3.download_file(os.environ['R2_BUCKET_NAME'], args.weights, tmp)
        with open(tmp, 'rb') as f:
            raw = pickle.load(f)

    actor_model        = make_actor()
    actor_graph, _     = nnx.split(actor_model)
    params             = np_to_params(raw)
    actor_model        = nnx.merge(actor_graph, params)
    # Shift hold_head bias to counteract the near-perma-closed gate learned during BC.
    # BC training saw sparse launch events → degenerate "never launch" solution.
    actor_model.hold_head.bias[...] = actor_model.hold_head.bias[...] + CFG.launch_gate_bias
    _, params          = nnx.split(actor_model)
    n_params           = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'Actor loaded  ({n_params//1000:.0f}K params, gate bias={CFG.launch_gate_bias:+.1f})')

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer  = optax.adamw(CFG.lr, weight_decay=1e-4)
    opt_state  = optimizer.init(params)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_iter = 0
    current_elo = CFG.hof_init_elo
    hof  = EloHoF()
    crit = S5Critic()

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'rb') as f:
            ckpt = pickle.load(f)
        params      = np_to_params(ckpt['params'])
        opt_state   = jax.tree_util.tree_map(jnp.array, ckpt['opt_state'])
        start_iter  = ckpt['iter']
        current_elo = ckpt.get('elo', CFG.hof_init_elo)
        hof.entries = ckpt.get('hof', [])
        print(f'Resumed from iter {start_iter}  ELO={current_elo:.0f}')

    # ── Seed HoF with BC init ─────────────────────────────────────────────────
    if not hof.entries:
        hof.add(params_to_np(params), tag='bc_init')
        print('HoF seeded with BC init.')

    # ── Build rollout + PPO fns ────────────────────────────────────────────────
    print('Compiling rollout + PPO (first call may take 5-15 min on GPU)...', flush=True)
    pmap_rollout_2p       = make_pmap_rollout(actor_graph, num_players=2)       # single opp (exploiter)
    pmap_rollout_4p       = make_pmap_rollout(actor_graph, num_players=4)
    pmap_rollout_2p_multi = make_pmap_rollout_multi_opp(actor_graph, num_players=2)  # per-device opp
    pmap_rollout_4p_multi = make_pmap_rollout_multi_opp(actor_graph, num_players=4)
    ppo_train             = make_ppo_train_fn(actor_graph, optimizer)
    rng             = np.random.default_rng(42)
    games_per_mode  = CFG.games_per_device // 2

    # ── Exploiter state ────────────────────────────────────────────────────────
    exploiter_params    = None
    exploiter_opt_state = None
    exploiter_iters     = 0
    frozen_target       = None
    exploiter_optimizer = optax.adamw(CFG.lr * 2, weight_decay=1e-4)
    expl_ppo_train      = make_ppo_train_fn(actor_graph, exploiter_optimizer)

    # R2 sync
    r2_sync = R2SyncThread(CFG.out_dir, os.environ.get('R2_BUCKET_NAME', ''))
    r2_sync.start()

    # ── merge pmap output: [n_dev, T, B_local, ...] → [T, n_dev*B_local, ...] ──
    def merge_jax(x):
        nd, T, B_local = x.shape[0], x.shape[1], x.shape[2]
        return x.transpose(1, 0, *range(2, x.ndim)).reshape(T, nd * B_local, *x.shape[3:])

    # ── Training loop ──────────────────────────────────────────────────────────
    prev_r2 = 0.0   # critic R² from previous iter — gates PPO update
    for iteration in range(start_iter, start_iter + args.n_iters):
        t_iter = time.time()
        key = jr.PRNGKey(iteration * 1000)

        # ── Select opponents from HoF — one per device ───────────────────
        fallback_np = params_to_np(params)
        opp_indices_2p, opp_stack_2p = sample_opp_per_device(hof, rng, n_devices, fallback_np)
        opp_indices_4p, opp_stack_4p = sample_opp_per_device(hof, rng, n_devices, fallback_np)

        # ── Collect rollouts (stay on GPU) ────────────────────────────────
        device_keys = jr.split(key, n_devices)
        gkeys_2p    = jnp.stack([jr.split(dk,                games_per_mode) for dk in device_keys])
        gkeys_4p    = jnp.stack([jr.split(jr.fold_in(dk, 7), games_per_mode) for dk in device_keys])
        nkeys_2p    = jr.split(jr.fold_in(key,  99), n_devices)
        nkeys_4p    = jr.split(jr.fold_in(key, 100), n_devices)

        _d0 = jax.devices()[0]
        _to_d0 = lambda x: jax.device_put(x, _d0)
        if iteration == 0: print('  [compile] starting 2p rollout...', flush=True)
        traj_2p = jax.tree_util.tree_map(_to_d0, jax.tree_util.tree_map(merge_jax, pmap_rollout_2p_multi(params, opp_stack_2p, gkeys_2p, nkeys_2p)))
        if iteration == 0: print('  [compile] 2p rollout done, starting 4p rollout...', flush=True)
        traj_4p = jax.tree_util.tree_map(_to_d0, jax.tree_util.tree_map(merge_jax, pmap_rollout_4p_multi(params, opp_stack_4p, gkeys_4p, nkeys_4p)))
        if iteration == 0: print('  [compile] 4p rollout done.', flush=True)

        # ── Extract outcomes + lengths (pure JAX, stays on GPU) ───────────
        lens_2p, out_2p, wr_2p = extract_outcomes_jax(traj_2p, 2)
        lens_4p, out_4p, wr_4p = extract_outcomes_jax(traj_4p, 4)
        win_rate = float(0.5 * (wr_2p + wr_4p))

        # ── Actor embeddings for GRU critic ──────────────────────────────
        emb_actor_2p = traj_2p['embeddings'].transpose(1, 0, 2)    # [B, T, D]
        emb_actor_4p = traj_4p['embeddings'].transpose(1, 0, 2)
        emb_actor  = jnp.concatenate([emb_actor_2p, emb_actor_4p], axis=0)
        lens_actor = jnp.concatenate([lens_2p, lens_4p], axis=0)
        vals_actor = crit.values(emb_actor)                                  # [B_actor, T]

        # Merge traj for PPO (2p + 4p along B axis)
        _ppo_keys = ['planets', 'fleets', 'pmask', 'fmask', 'old_log_prob',
                     'noisy_logits', 'owned_p0', 'active', 'reward', 'done']
        traj = {k: jnp.concatenate([traj_2p[k], traj_4p[k]], axis=1) for k in _ppo_keys}
        T_all, B_total = traj['done'].shape

        # Values aligned with merged traj [T, B_total]
        values = vals_actor.transpose(1, 0)   # [T, B_total] — T dim first

        # ── GAE (pure JAX) ────────────────────────────────────────────────
        advantages, return_targets = compute_gae_jax(
            traj['reward'], traj['done'], traj['active'], values)

        # ── PPO (lax.scan, single JIT — no Python loop) ───────────────────
        if iteration == 0: print('  [compile] starting PPO...', flush=True)
        in_warmup   = iteration < CFG.critic_warmup_iters
        r2_ok       = prev_r2 >= CFG.r2_ppo_threshold
        do_ppo      = (not in_warmup) and r2_ok
        if do_ppo:
            ppo_key = jr.fold_in(key, 42)
            params, opt_state, losses = ppo_train(
                params, opt_state,
                traj['planets'], traj['fleets'], traj['pmask'], traj['fmask'],
                traj['noisy_logits'], traj['owned_p0'].astype(jnp.float32),
                traj['old_log_prob'], advantages, traj['active'], ppo_key,
            )
            pg_loss  = float(losses[0].mean())
            ent_loss = float(losses[1][1].mean())
        else:
            pg_loss, ent_loss = 0.0, 0.0

        # ── S5 critic update ──────────────────────────────────────────────
        if iteration == 0: print('  [compile] PPO done, starting S5 critic update...', flush=True)
        active_BT  = traj['active'].T.astype(jnp.float32)
        targets_BT = return_targets.T
        crit_r2, _, crit_time = crit.update(emb_actor, active_BT, targets_BT)
        prev_r2 = crit_r2

        # ── Exploiter training ────────────────────────────────────────────
        if iteration < 2: print(f'  [log] S5 done, entering post-critic section...', flush=True)
        if iteration % CFG.exploiter_spawn_every == 0 and iteration > 0:
            print(f'  Spawning new exploiter at iter {iteration}')
            exploiter_params    = params_to_np(params)
            exploiter_opt_state = exploiter_optimizer.init(np_to_params(exploiter_params))
            frozen_target       = params_to_np(params)
            exploiter_iters     = 0

        if exploiter_params is not None and exploiter_iters < CFG.exploiter_budget:
            expl_p = np_to_params(exploiter_params)
            frz_p  = np_to_params(frozen_target)
            expl_traj_raw = pmap_rollout_2p(expl_p, frz_p, gkeys_2p, nkeys_2p)
            expl_traj     = jax.tree_util.tree_map(_to_d0, jax.tree_util.tree_map(merge_jax, expl_traj_raw))

            expl_lens, expl_out, expl_wr = extract_outcomes_jax(expl_traj, 2)
            expl_emb  = expl_traj['embeddings'].transpose(1, 0, 2)   # [B, T, D]
            expl_vals = crit.values(expl_emb).transpose(1, 0)        # [T, B]
            expl_adv, _ = compute_gae_jax(
                expl_traj['reward'], expl_traj['done'], expl_traj['active'], expl_vals)
            expl_p, exploiter_opt_state, _ = expl_ppo_train(
                expl_p, exploiter_opt_state,
                expl_traj['planets'], expl_traj['fleets'],
                expl_traj['pmask'],   expl_traj['fmask'],
                expl_traj['noisy_logits'],
                expl_traj['owned_p0'].astype(jnp.float32),
                expl_traj['old_log_prob'], expl_adv, expl_traj['active'],
                jr.fold_in(key, 99),
            )
            exploiter_params = params_to_np(expl_p)
            exploiter_iters += 1
            if exploiter_iters >= CFG.exploiter_budget:
                expl_wr_f = float(expl_wr)
                expl_elo  = CFG.hof_init_elo + (expl_wr_f - 0.5) * 400
                if expl_elo >= CFG.exploiter_hof_min_elo:
                    hof.add(exploiter_params, tag=f'exploiter_i{iteration}', elo=expl_elo)
                    print(f'  Exploiter added to HoF  (approx ELO={expl_elo:.0f}  wr={expl_wr_f:.2f})')
                else:
                    print(f'  Exploiter retired (wr={expl_wr_f:.2f} < threshold)')
                exploiter_params = None

        # ── ELO update (every iter, scaled k so total delta matches old 50-iter batch) ──
        wr_2p_f = float(wr_2p)
        wr_4p_f = float(wr_4p)
        k_per_iter = CFG.hof_k_factor / CFG.hof_update_every
        current_elo += k_per_iter * (win_rate - 0.5)
        # Split k across per-device opponents so total ELO signal per mode is preserved
        k_per_opp = k_per_iter / n_devices
        for idx in opp_indices_2p:
            hof.update_elo(idx, wr_2p_f, current_elo, k_per_opp)
        for idx in opp_indices_4p:
            hof.update_elo(idx, wr_4p_f, current_elo, k_per_opp)

        # ── HoF admission + summary (every hof_update_every iters) ──────────
        # Always try to add; eviction in add() removes lowest ELO when at capacity.
        # New entry starts at current_elo so weak agents don't displace stronger ones.
        if iteration % CFG.hof_update_every == 0 and iteration > 0:
            hof.add(params_to_np(params), tag=f'iter_{iteration}', elo=current_elo)
            print(hof.summary())

        # ── Checkpoint ────────────────────────────────────────────────────
        if iteration % CFG.save_every == 0:
            ckpt = {
                'params':    params_to_np(params),
                'opt_state': jax.tree_util.tree_map(np.array, opt_state),
                'iter':      iteration,
                'elo':       current_elo,
                'hof':       hof.entries,
            }
            ckpt_path = os.path.join(CFG.out_dir, f'rl_ckpt_iter{iteration:06d}.pkl')
            with open(ckpt_path, 'wb') as f:
                pickle.dump(ckpt, f)

        # ── Logging ───────────────────────────────────────────────────────
        if iteration < 2: print(f'  [log] computing avg_ep_len...', flush=True)
        avg_ep_len = float(lens_actor.astype(jnp.float32).mean())
        if iteration < 2: print(f'  [log] all floats done, printing...', flush=True)
        iter_time  = time.time() - t_iter
        warmup_tag = ' [warmup]' if in_warmup else (' [r2gate]' if not r2_ok else '')
        print(
            f'iter {iteration:5d}{warmup_tag} | '
            f'win={win_rate:.3f} (2p={wr_2p_f:.2f}/4p={wr_4p_f:.2f})  elo={current_elo:.0f}  '
            f'pg={pg_loss:.4f}  ent={ent_loss:.4f}  '
            f'crit_r2={crit_r2:.3f}  ep_len={avg_ep_len:.0f}  '
            f'crit={crit_time:.1f}s  total={iter_time:.1f}s',
            flush=True
        )


if __name__ == '__main__':
    main()
