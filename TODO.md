# TODO

## Website: Pre-Bake Replays + Feature Improvements

**Goal:** Cut "New Game" latency from 3-5 min → ~30s by pre-baking replay HTMLs on Kaggle and having GitHub Actions just copy files (no JAX, no dep install).

### Scripts to create
- `scripts/generate_replays.py` — runs on Kaggle after training; loads HoF checkpoint, runs rollouts, writes replay HTMLs + `manifest.json`, uploads all to R2 under `replays/` prefix
- `scripts/generate_index.py` — pure stdlib, runs in CI; downloads manifest + replay HTMLs from R2, generates `public/index.html` from template

### `manifest.json` shape
```json
{
  "generation": 90, "hof_filled": 50, "has_exploit": true,
  "generated_at": "...",
  "replays": { "ffa-archive": "ffa_archive.html", "duel-archive": "duel_archive.html", ... },
  "stats": { ... },
  "fitness_history": [{"gen": 10, "best": 0.31, "mean": 0.24}, ...]
}
```

### `.github/workflows/deploy_ondemand.yml`
Replace current workflow with: `pip install awscli` → download from R2 → `python scripts/generate_index.py` → Pages deploy. Remove `uv`, JAX, `generate_site.py` from CI path.

### `scripts/train.py` (Kaggle only, never commit)
Add `best_ever_seen` and `mean_score` fields to `meta_{gen}.json` checkpoint saves.

### UI improvements
- Training curve: embed `fitness_history` from manifest as JS array, render with inline `<canvas>` chart (~30 lines Canvas API, no external lib)
- Update loading overlay: "approximately 3 minutes" → "approximately 30 seconds"; progress interval 180s → 30s
