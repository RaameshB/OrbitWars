import sys
import glob

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    content = content.replace("checkpoints_v3", "checkpoints_v4")
    
    with open(filepath, 'w') as f:
        f.write(content)

for f in glob.glob("train*.py"):
    patch_file(f)

