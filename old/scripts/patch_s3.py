import sys
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # R2SyncThread push
    content = content.replace('f"s3://{self.bucket_name}/"', 'f"s3://{self.bucket_name}/v4/"')
    # train.py pull
    content = content.replace('f"s3://{r2_bucket}/"', 'f"s3://{r2_bucket}/v4/"')
    
    with open(filepath, 'w') as f:
        f.write(content)

for f in glob.glob("train*.py"):
    patch_file(f)

