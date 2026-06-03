"""
Warm restart for v13.3bc training.

Downloads the current trained weights from R2, strips the optimizer state,
reinitialises with new hyperparameters, and uploads back to R2.

Usage:
    uv run python scripts/make_v13_3_warmrestart.py
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
NEW_LR        = 3e-2
NEW_EPOCHS    = 150
NEW_BATCH     = 4096
WARMUP_EPOCHS = 5
R2_PREFIX     = 'bc_v13_3bc'

SRC_R2_KEY  = f'{R2_PREFIX}/weights_bc_v13_3.pkl'
OUT_DIR     = 'tmp'
OUT_RESUME  = os.path.join(OUT_DIR, 'weights_bc_v13_3_resume.pkl')
OUT_PARAMS  = os.path.join(OUT_DIR, 'weights_bc_v13_3.pkl')

os.makedirs(OUT_DIR, exist_ok=True)

endpoint = os.environ.get('R2_ENDPOINT_URL', '')
bucket   = os.environ.get('R2_BUCKET_NAME', '')
aws_env  = {
    **os.environ,
    'AWS_ACCESS_KEY_ID':     os.environ.get('R2_ACCESS_KEY_ID', ''),
    'AWS_SECRET_ACCESS_KEY': os.environ.get('R2_SECRET_ACCESS_KEY', ''),
    'AWS_DEFAULT_REGION':    'auto',
}

print(f'Downloading trained weights from R2: {SRC_R2_KEY}...')
result = subprocess.run(
    ['aws', 's3', 'cp', f's3://{bucket}/{SRC_R2_KEY}', OUT_PARAMS,
     '--endpoint-url', endpoint],
    capture_output=True, text=True, env=aws_env
)
if result.returncode != 0:
    print(f'Download failed: {result.stderr.strip()}')
    sys.exit(1)
print(f'  Downloaded → {OUT_PARAMS} ({os.path.getsize(OUT_PARAMS)/1e3:.1f} KB)')

print('Loading params...')
with open(OUT_PARAMS, 'rb') as f:
    src = pickle.load(f)

# Support both params-only dict and full resume checkpoint
if isinstance(src, dict) and 'params' in src:
    params_np = src['params']
    old_epoch = src.get('epoch', '?')
    old_val   = src.get('best_val', float('inf'))
    print(f'  Loaded resume checkpoint (epoch={old_epoch}, best_val={old_val:.4f})')
else:
    params_np = src
    print('  Loaded params-only checkpoint')

params = jax.tree_util.tree_map(jnp.array, params_np)

# Estimate steps_per_epoch from typical dataset size
# 1,642,058 effective train samples / 4096 batch = ~401 steps/epoch
steps_per_epoch = 1_642_058 // NEW_BATCH
total_steps     = NEW_EPOCHS * steps_per_epoch
warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
decay_steps     = max(total_steps - warmup_steps, 1)

print(f'Initialising fresh Muon optimizer (lr={NEW_LR}, batch={NEW_BATCH}, epochs={NEW_EPOCHS})...')
print(f'  steps_per_epoch≈{steps_per_epoch}, total≈{total_steps}, warmup={warmup_steps}')

schedule  = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, NEW_LR, max(warmup_steps, 1)),
        optax.cosine_decay_schedule(NEW_LR, decay_steps, alpha=0.01),
    ],
    boundaries=[warmup_steps],
)
opt       = optax.contrib.muon(learning_rate=schedule, weight_decay=0.0, adam_b2=0.95)
opt_state = opt.init(params)

ckpt = {
    'epoch':     0,
    'best_val':  float('inf'),
    'history':   [],
    'params':    jax.tree_util.tree_map(np.array, params),
    'opt_state': jax.tree_util.tree_map(np.array, opt_state),
}
print(f'Saving warm-restart checkpoint → {OUT_RESUME}')
with open(OUT_RESUME, 'wb') as f:
    pickle.dump(ckpt, f, protocol=4)

print(f'Saving params-only → {OUT_PARAMS}')
with open(OUT_PARAMS, 'wb') as f:
    pickle.dump(jax.tree_util.tree_map(np.array, params), f, protocol=4)

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

print(f'Done. Run with: --epochs {NEW_EPOCHS} --batch-size {NEW_BATCH} --lr {NEW_LR} --resume')
