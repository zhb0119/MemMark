# A-MEM Eval-Repo Installer

MemMark's A-MEM backend follows the A-MEM LoCoMo evaluation protocol. That protocol needs an A-MEM implementation exposing:

```python
AgenticMemorySystem.find_related_memories_raw
```

Some pip-installable A-MEM SDK versions do not provide this method. This helper installs the evaluation-repository variant used by the A-MEM LoCoMo experiments and makes it importable as:

```python
from agentic_memory.memory_system import AgenticMemorySystem
```

## Quick Install

From the MemMark repository root, first activate the environment that will run experiments. For example:

```bash
source .venv/bin/activate
which python
python -m pip install -e ".[amem]"
python tools/install_amem_eval/install.py --no-deps
```

The installer uses `sys.executable`, so `which python` must point to the intended virtualenv or conda env. If it points to `/root/miniconda3/bin/python`, the package will be installed into base conda instead.

The installer will:

1. clone `https://github.com/WujiangXu/AgenticMemory.git` to `../A-mem` unless it already exists,
2. add the minimal `setup.py` required for editable/package installation,
3. install the package as `agentic-memory`, and
4. verify that `find_related_memories_raw` is available.

When `--no-deps` is used, the installer does not ask pip to resolve A-MEM dependencies again. This is the recommended path after `pip install -e ".[amem]"`.

## Options

```bash
python tools/install_amem_eval/install.py --target /path/to/A-mem
python tools/install_amem_eval/install.py --no-clone
python tools/install_amem_eval/install.py --break-system-packages
python tools/install_amem_eval/install.py --no-deps
```

Use `--target` to choose a different clone directory. Use `--no-clone` when the repository is already present.

## Verification

```bash
python - <<'PY'
from agentic_memory.memory_system import AgenticMemorySystem
print(AgenticMemorySystem.__module__)
print(hasattr(AgenticMemorySystem, "find_related_memories_raw"))
PY
```

Expected output:

```text
memory_layer
True
```

The exact module name may differ across A-MEM repository revisions; the important condition is that `find_related_memories_raw` is present.

## Why This Exists

MemMark's A-MEM adapter uses the robust LoCoMo QA path from the A-MEM evaluation code: keyword generation, raw memory retrieval, and category-aware answering. Installing the eval-repo variant keeps this path aligned with the original A-MEM LoCoMo evaluation instead of silently falling back to a different retrieval format.
