# MemMark

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="OpenAI-compatible APIs" src="https://img.shields.io/badge/OpenAI--compatible-APIs-412991?logo=openai&logoColor=white">
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-Graph%20Memory-4581C3?logo=neo4j&logoColor=white">
  <img alt="LoCoMo" src="https://img.shields.io/badge/LoCoMo-Benchmark-0F766E">
  <img alt="A-MEM" src="https://img.shields.io/badge/A--MEM-Agentic%20Memory-7C3AED">
  <img alt="Graphiti" src="https://img.shields.io/badge/Graphiti-Temporal%20KG-EA580C">
</p>

Code release for reproducing the MemMark experiments on LoCoMo with the A-MEM and Graphiti memory backends.

MemMark studies watermarking for agent memory systems: the watermark is embedded at memory-evolution decision points while preserving the native behavior of the underlying memory backend. This repository contains the cleaned reproduction harness used for the LoCoMo experiments, including backend adapters, audit/verification utilities, metric computation, and sanitized launch scripts.

> This repository intentionally does not include experiment outputs, API keys, local model caches, or LoCoMo data files.

## What Is Included

```text
memmark/
  backends/              # Json, A-MEM, and Graphiti adapters
  benchmarks/locomo/     # LoCoMo loader, driver, QA prompts, metrics
  core/                  # sampler, commitments, Merkle audit log
  experiments/           # RQ1-RQ5 metric helpers
  llm/                   # OpenAI-compatible clients and watermark wrappers
  sdk/                   # MemoryWatermarker public interface
  verifier/              # full/partial/in-record verification utilities
  examples/run_locomo_full.py
scripts/
  run_locomo_smoke.sh
  run_locomo_amem.sh
  run_locomo_graphiti.sh
tools/install_amem_eval/ # installer for the A-MEM eval-repo package
```

## Supported Backends

- **JsonMemoryStore**: lightweight smoke-test backend with no external services.
- **A-MEM**: agentic-note memory backend, installed from the A-MEM evaluation repository to match the LoCoMo protocol.
- **Graphiti**: temporal knowledge-graph memory backend, backed by Neo4j.

Supported baselines:

- `watermark`
- `no_watermark`
- `signed_metadata_only`
- `random_replace`
- `kgmark_graphiti` for Graphiti only

## Installation

Use Python 3.10 or newer.

```bash
git clone <your-repo-url> memmark
cd memmark
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[amem,graphiti]"
```

`pip install -e ".[amem,graphiti]"` installs MemMark in editable mode and installs the Python dependencies for both backends. If you only need one backend, use `.[amem]` or `.[graphiti]`.

Create a local environment file:

```bash
cp .env.example .env
# Edit .env with your OpenAI-compatible API endpoints and keys.
set -a; source .env; set +a
```

The project uses OpenAI-compatible chat and embedding APIs. Keep `.env` private.

## Data

Download LoCoMo separately from the official repository:

```bash
git clone https://github.com/snap-research/locomo.git ../locomo
export MEMMARK_LOCOMO_PATH=$(realpath ../locomo/data/locomo10.json)
```

The data file is not vendored in this repository.

## Quick Smoke Test

The JSON backend with stub LLM mode checks the package, data loader, driver, and metric pipeline without API calls:

```bash
bash scripts/run_locomo_smoke.sh 0
```

For a direct Python invocation:

```bash
python -m memmark.examples.run_locomo_full \
  --locomo "$MEMMARK_LOCOMO_PATH" \
  --conversation 0 \
  --backend json \
  --llm-mode stub \
  --max-sessions 2 \
  --max-qa 5 \
  --baselines watermark no_watermark \
  --output-mode metrics
```

## Reproducing A-MEM Runs

A-MEM must be installed from its evaluation repository variant, because the paper-aligned LoCoMo QA path needs `find_related_memories_raw`.

```bash
python tools/install_amem_eval/install.py --no-deps
```

Use `--no-deps` after installing `.[amem]`; it installs only the A-MEM eval package and avoids changing already-pinned dependencies.

Then run one baseline:

```bash
bash scripts/run_locomo_amem.sh 0 watermark
```

Run all A-MEM baselines for one conversation:

```bash
for baseline in watermark no_watermark signed_metadata_only random_replace; do
  bash scripts/run_locomo_amem.sh 0 "$baseline"
done
```

The A-MEM embedding model can be configured with:

```bash
export AMEM_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
# or a local path, e.g. /path/to/all-MiniLM-L6-v2
```

## Reproducing Graphiti Runs

Graphiti requires Neo4j. A minimal local compose file is provided:

```bash
docker compose -f docker-compose.neo4j.yml up -d
```

Install Graphiti:

```bash
pip install graphiti-core
```

If the pip release is incompatible with the code path you need, install upstream Graphiti editable:

```bash
git clone https://github.com/getzep/graphiti.git ../graphiti
pip install -e ../graphiti
```

Run one baseline:

```bash
bash scripts/run_locomo_graphiti.sh 0 watermark
```

Run all Graphiti baselines for one conversation:

```bash
for baseline in watermark no_watermark signed_metadata_only random_replace kgmark_graphiti; do
  bash scripts/run_locomo_graphiti.sh 0 "$baseline"
done
```

Graphiti experiments should usually be serialized against a single Neo4j instance.

## Main Entry Point

All scripts call the same runner:

```bash
python -m memmark.examples.run_locomo_full \
  --locomo "$MEMMARK_LOCOMO_PATH" \
  --conversation 0 \
  --backend amem \
  --llm-mode real \
  --progress \
  --async-assess \
  --async-max-concurrency 4 \
  --max-sessions 999 \
  --max-qa 9999 \
  --baselines watermark \
  --output-mode metrics
```

Important options:

- `--backend {json,amem,graphiti}` selects the memory backend.
- `--baselines ...` selects one or more baselines.
- `--max-sessions` and `--max-qa` control run size.
- `--llm-mode stub` is for smoke tests; `--llm-mode real` is required for paper-style runs.
- `--output-mode metrics` writes compact metric JSON; `full` also includes detailed traces.
- `--output PATH` overrides the default output path.
- `--save-checkpoints` enables legacy recovery files (`.partial` and per-baseline JSON); by default the runner writes only one clean output JSON.

## Outputs

By default each run writes exactly one JSON file. The default path is concise and structured:

```text
results/<memory_system>/<model_name>/<time>/convX_<baseline>.json
```

Examples:

```text
results/amem/deepseek-v4-pro/20260526-123456/conv0_watermark.json
results/graphiti/deepseek-v4-pro/20260526-123456/conv0_kgmark_graphiti.json
results/json/model/20260526-123456/conv0_watermark+no_watermark.json
```

`model_name` is resolved from `RESULT_MODEL_NAME`, `TARGET_LLM_MODEL`, `MEMMARK_MODEL`, or `OPENAI_MODEL`. `time` is `RUN_TAG` / `MEMMARK_RUN_TAG` if set, otherwise the current timestamp (`YYYYmmdd-HHMMSS`).

The JSON contains:

- run configuration and LoCoMo conversation metadata
- RQ1 utility metrics
- RQ2 capacity metrics
- RQ3 in-record attribution metrics
- RQ4 robustness metrics
- RQ5 integrity metrics

Use `--output /custom/path.json` to choose a path manually. Use `--save-checkpoints` only when you want the legacy recovery files (`<output>.partial` and `<output_stem>_<baseline>.json`). Outputs are ignored by Git.

## Reproducibility Notes

- Fix `MEMMARK_KEY` for comparable watermark verification across runs.
- Keep backend and target LLM settings stable within a reported experiment.
- A-MEM runs can be slow because each turn may trigger memory evolution through the backend's native LLM path.
- Graphiti runs can be substantially slower and depend on Neo4j state; clear or isolate the graph between independent experiments if needed.
- Do not commit `.env`, `results/`, local model directories, or virtual environments.

## Citation

If you use this code, please cite:

```bibtex
@misc{zhang2026memmarkstateevolutionattributionwatermarking,
      title={MemMark: State-Evolution Attribution Watermarking for Agent Long-Term Memory Systems}, 
      author={Haobo Zhang and Xutao Mao and Guangyuan Dong and Ziwei Li and Xuanbo Su and Kaijie Chen and Jing Yang and Zheng Lin},
      year={2026},
      eprint={2605.25002},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.25002}, 
}
```

## License

This repository is released under the MIT License. Third-party systems such as A-MEM, Graphiti, and LoCoMo are governed by their own licenses.
