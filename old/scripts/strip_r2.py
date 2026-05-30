import json

with open("eval_local.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # We find the cell with the boto3 logic and comment it out
        if "s3.get_paginator('list_objects_v2')" in source:
            lines = source.split('\n')
            new_lines = []
            for line in lines:
                if 'r2_endpoint = os.environ' in line or 'boto3' in line or 's3' in line or 'paginator' in line or 'dest_path' in line or 'file_key' in line or 'page[' in line or 'obj[' in line or "print('Syncing checkpoints from R2...')" in line or "os.makedirs('/tmp/checkpoints_v4'" in line:
                    new_lines.append("# " + line)
                else:
                    new_lines.append(line)
            source = "\n".join(new_lines)
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in source.split("\n")][:-1]

with open("eval_local.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

