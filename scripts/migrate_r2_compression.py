"""
One-shot migration: compress all R2 checkpoints except the last KEEP_UNCOMPRESSED
into .tar.zst archives at max compression, then delete the original directories.

Run once from the project root:
    uv run python scripts/migrate_r2_compression.py

Idempotent — already-compressed checkpoints are skipped.
"""
import os, re, subprocess, tempfile, threading, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

load_dotenv()

BUCKET       = os.environ['R2_BUCKET_NAME']
ENDPOINT     = os.environ['R2_ENDPOINT_URL']
ACCESS_KEY   = os.environ['R2_ACCESS_KEY_ID']
SECRET_KEY   = os.environ['R2_SECRET_ACCESS_KEY']
KEEP_UNCOMPRESSED = 2   # keep newest N as raw directories

print_lock = threading.Lock()

def log(msg):
    with print_lock:
        print(msg, flush=True)

def make_client():
    return boto3.client(
        's3',
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name='auto',
    )

def compress_one(name, bucket):
    """Download a checkpoint directory, compress to .tar.zst, re-upload, delete original."""
    s3 = make_client()
    paginator = s3.get_paginator('list_objects_v2')
    src_prefix = f'v6/{name}/'
    archive_key = f'v6/{name}.tar.zst'

    # Skip if already compressed
    try:
        s3.head_object(Bucket=bucket, Key=archive_key)
        log(f'  skip (already compressed): {name}')
        return name, 'skipped'
    except Exception:
        pass

    # Confirm source exists
    probe = s3.list_objects_v2(Bucket=bucket, Prefix=src_prefix, MaxKeys=1)
    if not probe.get('Contents'):
        log(f'  skip (source empty): {name}')
        return name, 'empty'

    log(f'  compressing: {name}')
    with tempfile.TemporaryDirectory() as tmp:
        local_dir = os.path.join(tmp, name)
        os.makedirs(local_dir)

        # Download all objects
        for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
            for obj in page.get('Contents', []):
                rel = obj['Key'][len(src_prefix):]
                if not rel:
                    continue
                dest = os.path.join(local_dir, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                s3.download_file(bucket, obj['Key'], dest)

        # Compress: ZSTD_CLEVEL=22 gives maximum compression (ultra mode)
        archive_path = os.path.join(tmp, f'{name}.tar.zst')
        env = {**os.environ, 'ZSTD_CLEVEL': '22'}
        result = subprocess.run(
            ['tar', '--zstd', '-cf', archive_path, '-C', tmp, name],
            env=env, capture_output=True, text=True
        )
        if result.returncode != 0:
            log(f'  ERROR compressing {name}: {result.stderr}')
            return name, 'error'

        orig_mb = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, files in os.walk(local_dir)
            for f in files
        ) / 1e6
        comp_mb = os.path.getsize(archive_path) / 1e6
        log(f'  {name}: {orig_mb:.1f}MB → {comp_mb:.1f}MB ({100*comp_mb/orig_mb:.0f}%)')

        # Upload archive
        s3.upload_file(archive_path, bucket, archive_key)

    # Delete original directory from R2
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix):
        keys = [{'Key': o['Key']} for o in page.get('Contents', [])]
        if keys:
            s3.delete_objects(Bucket=bucket, Delete={'Objects': keys})

    log(f'  done: {name}')
    return name, 'done'


def main():
    s3 = make_client()

    # List all uncompressed checkpoint directories (qdax_rep_N, not _hof)
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix='v6/', Delimiter='/')
    raw_dirs = sorted(
        (int(m.group(1)), p['Prefix'].rstrip('/').split('/')[-1])
        for p in resp.get('CommonPrefixes', [])
        if (m := re.search(r'^qdax_rep_(\d+)$', p['Prefix'].rstrip('/').split('/')[-1]))
    )
    hof_dirs = sorted(
        (int(m.group(1)), p['Prefix'].rstrip('/').split('/')[-1])
        for p in resp.get('CommonPrefixes', [])
        if (m := re.search(r'^qdax_rep_(\d+)_hof$', p['Prefix'].rstrip('/').split('/')[-1]))
    )

    if not raw_dirs:
        print("No uncompressed checkpoints found.")
        return

    keep_gens = {gen for gen, _ in raw_dirs[-KEEP_UNCOMPRESSED:]}
    print(f"Found {len(raw_dirs)} uncompressed checkpoints, keeping gens {sorted(keep_gens)} uncompressed.")

    to_compress = (
        [name for gen, name in raw_dirs if gen not in keep_gens] +
        [name for gen, name in hof_dirs if gen not in keep_gens]
    )
    print(f"Compressing {len(to_compress)} directories: {to_compress}\n")

    if not to_compress:
        print("Nothing to compress.")
        return

    # Run in parallel — each worker is I/O bound (download/upload) with a CPU burst
    # for compression. Use up to 4 workers to avoid saturating the R2 connection.
    results = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(compress_one, name, BUCKET): name for name in to_compress}
        for future in as_completed(futures):
            name, status = future.result()
            results[name] = status

    done    = sum(1 for s in results.values() if s == 'done')
    skipped = sum(1 for s in results.values() if s == 'skipped')
    errors  = sum(1 for s in results.values() if s == 'error')
    print(f"\nMigration complete: {done} compressed, {skipped} skipped, {errors} errors.")

if __name__ == '__main__':
    main()
