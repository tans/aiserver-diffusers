#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
BASE_PYTHON="${H3_BASE_PYTHON:-/opt/conda/envs/vllm/bin/python3.12}"
if [ ! -x .venv/bin/python ]; then
  "$BASE_PYTHON" -m venv .venv
fi
.venv/bin/python - <<'PY'
from pathlib import Path
import site
shared = Path('/srv/lightx2v/venv/lib/python3.12/site-packages')
if not (shared / 'torch').is_dir():
    raise RuntimeError('The audited POWER9 torch binary stack is missing')
(Path(site.getsitepackages()[0]) / 'power9-readonly-stack.pth').write_text(str(shared) + '\n')
PY
# All package writes go into this project's venv. Never invoke the old pip.
.venv/bin/python -m pip install --no-deps --ignore-installed -r requirements.txt
.venv/bin/python - <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
names = ('torch','diffusers','transformers','accelerate','safetensors','fastapi','uvicorn','av','numpy','peft','pydantic','python-multipart')
versions = {name: m.version(name) for name in names}
root = Path('data'); root.mkdir(exist_ok=True)
(root / 'environment-packages.json').write_text(json.dumps(versions,indent=2))
print(json.dumps(versions,indent=2))
PY
