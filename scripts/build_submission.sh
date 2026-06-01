#!/usr/bin/env bash
# Build submission.tar.gz from the submission/ directory.
# Run export_v6_agent.py first to generate submission/weights.pkl.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."

if [ ! -f "$ROOT/submission/weights.pkl" ]; then
    echo "ERROR: submission/weights.pkl not found. Run:"
    echo "  uv run python scripts/export_v6_agent.py"
    exit 1
fi

# Sync core files in case they changed
cp "$ROOT/core/networks.py"      "$ROOT/submission/core/networks.py"
cp "$ROOT/core/rollout_utils.py" "$ROOT/submission/core/rollout_utils.py"

cd "$ROOT"
tar -czf submission.tar.gz \
    submission/main.py \
    submission/core/__init__.py \
    submission/core/networks.py \
    submission/core/rollout_utils.py \
    submission/weights.pkl

SIZE=$(du -sh submission.tar.gz | cut -f1)
echo "Built submission.tar.gz ($SIZE)"
echo "Submit via: kaggle competitions submit -c orbit-wars -f submission.tar.gz -m 'v6 PGA-ME gen 380'"
