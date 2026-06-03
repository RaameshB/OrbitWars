"""
Build a v13.3bc resume checkpoint from the v13 best weights.

Transfer learning:
  - Keep: SA blocks + CA cross-attention (174 leaves)
  - Reinit: decomposed action head — hold_head, dest_q, dest_k, ds_prob, ds_sincos (10 leaves)

Output:
  tmp/weights_bc_v13_3_resume.pkl  — resume checkpoint (epoch=0, fresh optimizer)
  tmp/weights_bc_v13_3.pkl         — params only

Then uploads both to R2 under bc_v13_3bc/ prefix.
"""

import os, sys, pickle, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from core.networks import Actor

HIDDEN_DIM    = 48
NUM_SA_LAYERS = 6

REINIT_SUBSTRINGS = [
    'hold_head',
    'dest_q', 'dest_k',
    'ds_prob', 'ds_sincos',
    'ca_block_0.fleet_mlp', 'ca_block_0.planet_mlp', 'ca_block_0.rel_bias_mlp',
    'ca_block_1.fleet_mlp', 'ca_block_1.rel_bias_mlp',
]

SRC_PATH   = 'models/weights_bc_v13_best.pkl'
OUT_DIR    = 'tmp'
OUT_RESUME = os.path.join(OUT_DIR, 'weights_bc_v13_3_resume.pkl')
OUT_PARAMS = os.path.join(OUT_DIR, 'weights_bc_v13_3.pkl')
R2_PREFIX  = 'bc_v13_3bc'

os.makedirs(OUT_DIR, exist_ok=True)

print(f'Loading v13 weights from {SRC_PATH}...')
with open(SRC_PATH, 'rb') as f:
    src_params_np = pickle.load(f)
src_params = jax.tree_util.tree_map(jnp.array, src_params_np)

print('Initialising fresh Actor (CPU)...')
actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
actor_graph, dst_params = nnx.split(actor)

src_leaves, _        = jax.tree_util.tree_flatten_with_path(src_params)
dst_leaves, dst_tdef = jax.tree_util.tree_flatten_with_path(dst_params)

if len(src_leaves) != len(dst_leaves):
    print(f'WARNING: leaf count mismatch src={len(src_leaves)} dst={len(dst_leaves)}')
    print('Falling back to name-matched transfer...')
    src_dict = {
        '.'.join(str(k.key if hasattr(k, 'key') else k) for k in path): leaf
        for path, leaf in src_leaves
    }
    new_leaves = []
    transferred, reinitialized, missing = 0, 0, 0
    for dst_path, dst_leaf in dst_leaves:
        path_str = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in dst_path)
        if any(sub in path_str for sub in REINIT_SUBSTRINGS):
            new_leaves.append(dst_leaf)
            reinitialized += 1
        elif path_str in src_dict:
            new_leaves.append(src_dict[path_str])
            transferred += 1
        else:
            new_leaves.append(dst_leaf)
            missing += 1
            print(f'  missing in src: {path_str}')
else:
    new_leaves = []
    transferred, reinitialized = 0, 0
    for (dst_path, dst_leaf), (_, src_leaf) in zip(dst_leaves, src_leaves):
        path_str = '.'.join(str(k.key if hasattr(k, 'key') else k) for k in dst_path)
        if any(sub in path_str for sub in REINIT_SUBSTRINGS):
            new_leaves.append(dst_leaf)
            reinitialized += 1
        else:
            new_leaves.append(src_leaf)
            transferred += 1

params = jax.tree_util.tree_unflatten(dst_tdef, new_leaves)
print(f'  Transferred {transferred} leaves, reinitialized {reinitialized} leaves')

print('Initialising fresh Muon optimizer state...')
schedule  = optax.constant_schedule(3e-4)
opt       = optax.contrib.muon(learning_rate=schedule, weight_decay=0.0)
opt_state = opt.init(params)

ckpt = {
    'epoch':     0,
    'best_val':  float('inf'),
    'history':   [],
    'params':    jax.tree_util.tree_map(np.array, params),
    'opt_state': jax.tree_util.tree_map(np.array, opt_state),
}
print(f'Saving resume checkpoint → {OUT_RESUME}')
with open(OUT_RESUME, 'wb') as f:
    pickle.dump(ckpt, f, protocol=4)

params_np = jax.tree_util.tree_map(np.array, params)
print(f'Saving params-only → {OUT_PARAMS}')
with open(OUT_PARAMS, 'wb') as f:
    pickle.dump(params_np, f, protocol=4)

endpoint = os.environ.get('R2_ENDPOINT_URL', '')
bucket   = os.environ.get('R2_BUCKET_NAME', '')
if not endpoint or not bucket:
    print('R2 creds not found — skipping upload.')
else:
    aws_env = {
        **os.environ,
        'AWS_ACCESS_KEY_ID':     os.environ.get('R2_ACCESS_KEY_ID', ''),
        'AWS_SECRET_ACCESS_KEY': os.environ.get('R2_SECRET_ACCESS_KEY', ''),
        'AWS_DEFAULT_REGION':    'auto',
    }
    for path in [OUT_RESUME, OUT_PARAMS]:
        key = f's3://{bucket}/{R2_PREFIX}/{os.path.basename(path)}'
        result = subprocess.run(
            ['aws', 's3', 'cp', path, key, '--endpoint-url', endpoint],
            capture_output=True, text=True, env=aws_env
        )
        if result.returncode == 0:
            print(f'  [R2] {os.path.basename(path)} → {key}')
        else:
            print(f'  [R2] upload failed: {result.stderr.strip()}')

print('Done.')
