"""Export one agent from the v6 PGA-ME HoF checkpoint as numpy weights.

Usage:
    uv run python scripts/export_v6_agent.py [--slot 0] [--hof /tmp/checkpoints_v6/qdax_rep_380_hof]

Produces submission/weights.pkl (pickled nnx.State with numpy arrays).
"""
import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from core.networks import Actor

HOF_SIZE = 50
HIDDEN_DIM = 32
NUM_SA_LAYERS = 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot", type=int, default=0, help="HoF slot index to export (0-49)")
    parser.add_argument("--hof", type=str, default="/tmp/checkpoints_v6/qdax_rep_380_hof")
    args = parser.parse_args()

    print(f"Loading v6 HoF from {args.hof} ...")

    # Build template with matching structure
    actor = Actor(hidden_dim=HIDDEN_DIM, num_sa_layers=NUM_SA_LAYERS, rngs=nnx.Rngs(0))
    actor_graph, actor_params_template = nnx.split(actor)

    hof_params = jax.tree_util.tree_map(
        lambda x: jnp.zeros((HOF_SIZE, *x.shape), dtype=x.dtype),
        actor_params_template,
    )

    checkpointer = ocp.PyTreeCheckpointer()
    restore_args = ocp.checkpoint_utils.construct_restore_args(hof_params)
    hof_params = checkpointer.restore(args.hof, item=hof_params, restore_args=restore_args)
    print(f"Loaded HoF ({HOF_SIZE} agents). Extracting slot {args.slot} ...")

    # Extract single agent and convert to numpy
    agent_params = jax.tree_util.tree_map(lambda x: np.array(x[args.slot]), hof_params)

    os.makedirs("submission", exist_ok=True)
    out_path = "submission/weights.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(agent_params, f)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
