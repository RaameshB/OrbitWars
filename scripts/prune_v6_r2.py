"""Prune v6 R2 checkpoints: keep every 100th gen + latest gen (330), delete the rest.

Reads credentials from .env in the repo root (or from environment).

Usage:
    uv run python scripts/prune_v6_r2.py          # dry run — lists what would be deleted
    uv run python scripts/prune_v6_r2.py --delete  # actually delete
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Load .env from repo root if present
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

import boto3  # noqa: E402

KEEP_EVERY = 100
LATEST_GEN = 380

def get_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

def list_prefixes(client, bucket: str, prefix: str) -> list[str]:
    """List immediate child prefixes (like `aws s3 ls`)."""
    paginator = client.get_paginator("list_objects_v2")
    prefixes = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"].removeprefix(prefix).rstrip("/"))
    return prefixes

def delete_prefix(client, bucket: str, prefix: str):
    """Delete all objects under a prefix."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=bucket, Delete={"Objects": objects})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete (default is dry-run)")
    args = parser.parse_args()

    bucket = os.environ.get("R2_BUCKET_NAME", "orbit-wars-checkpoints")
    client = get_client()

    prefixes = list_prefixes(client, bucket, "v6/")
    print(f"Found {len(prefixes)} prefix(es) under v6/")

    gen_prefixes: dict[int, list[str]] = {}
    for p in prefixes:
        m = re.match(r'^qdax_rep_(\d+)(_hof)?$', p)
        if m:
            gen = int(m.group(1))
            gen_prefixes.setdefault(gen, []).append(p)

    all_gens = sorted(gen_prefixes.keys())
    print(f"Unique generations: {all_gens}")

    keep_gens  = {g for g in all_gens if g % KEEP_EVERY == 0 or g == LATEST_GEN}
    delete_gens = sorted(set(all_gens) - keep_gens)

    print(f"\nKeeping gens (every {KEEP_EVERY} + gen {LATEST_GEN}): {sorted(keep_gens)}")
    print(f"Deleting {len(delete_gens)} gens: {delete_gens}")

    to_delete: list[str] = []
    for gen in delete_gens:
        for pfx in gen_prefixes[gen]:
            to_delete.append(f"v6/{pfx}/")

    if not to_delete:
        print("\nNothing to delete.")
        return

    print(f"\n{'[DRY RUN] Would delete' if not args.delete else 'Deleting'} {len(to_delete)} prefix(es):")
    for path in to_delete:
        print(f"  s3://{bucket}/{path}")

    if args.delete:
        for path in to_delete:
            print(f"  Deleting s3://{bucket}/{path} ...", end=" ", flush=True)
            delete_prefix(client, bucket, path)
            print("done")
        print(f"\nPruned {len(to_delete)} prefix(es). Kept gens: {sorted(keep_gens)}")
    else:
        print("\nRe-run with --delete to actually remove these.")

if __name__ == "__main__":
    main()
