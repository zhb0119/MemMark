# MemMark

<p align="center">
  <a href="https://2026.emnlp.org/"><img alt="Findings of EMNLP 2026" src="https://img.shields.io/badge/Findings%20of%20EMNLP-2026-6B4EFF"></a>
  <a href="https://arxiv.org/abs/2605.25002"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.25002-b31b1b.svg"></a>
  <a href="https://henrymao2004.github.io/MemMark/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-4F46E5.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="OpenAI-compatible APIs" src="https://img.shields.io/badge/OpenAI--compatible-APIs-412991?logo=openai&logoColor=white">
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-Graph%20Memory-4581C3?logo=neo4j&logoColor=white">
  <img alt="LoCoMo" src="https://img.shields.io/badge/LoCoMo-Benchmark-0F766E">
  <img alt="A-MEM" src="https://img.shields.io/badge/A--MEM-Agentic%20Memory-7C3AED">
  <img alt="Graphiti" src="https://img.shields.io/badge/Graphiti-Temporal%20KG-EA580C">
</p>

Code release for reproducing the MemMark experiments on LoCoMo with the A-MEM and Graphiti memory backends.

🎉 Accepted to **Findings of EMNLP 2026**.

MemMark studies watermarking for agent memory systems: the watermark is embedded at memory-evolution decision points while preserving the native behavior of the underlying memory backend. This repository contains the cleaned reproduction harness used for the LoCoMo experiments, including backend adapters, audit/verification utilities, metric computation, and sanitized launch scripts.

## What Is Included

```text
memmark/
  backends/              # A-MEM and Graphiti adapters
  benchmarks/locomo/     # LoCoMo loader, driver, QA prompts, metrics
  core/                  # sampler, commitments, Merkle audit log
  experiments/           # RQ1-RQ5 metric helpers
  llm/                   # OpenAI-compatible clients and watermark wrappers
  sdk/                   # MemoryWatermarker public interface
  verifier/              # full/partial/in-record verification utilities
  examples/run_locomo_full.py
scripts/
  run_locomo_amem.sh
  run_locomo_graphiti.sh
tools/install_amem_eval/ # installer for the A-MEM eval-repo package
```

## Supported Backends

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

## API Configuration

MemMark separates two kinds of model APIs:

- `TARGET_LLM_*` / `MEMMARK_*`: the target API used by MemMark candidate sampling, LoCoMo fact extraction, and QA answering.
- `AMEM_LLM_*` / `GRAPHITI_LLM_*`: the memory system's own internal API used by A-MEM or Graphiti while they build and evolve memory.

This separation matters because the memory backend is the system under test, while the target/QA model is the evaluator and watermark candidate generator. The launch scripts map the backend-specific variables to `OPENAI_*` only inside that process when the upstream SDK expects OpenAI-compatible environment names.

Upstream defaults checked against the official repositories:

- A-MEM: `AgenticMemorySystem` defaults to `llm_backend="sglang"`, `llm_model="gpt-4o-mini"`, and `model_name="all-MiniLM-L6-v2"` for SentenceTransformer retrieval. The LoCoMo eval examples commonly run the OpenAI backend with `gpt-4o-mini`.
- Graphiti: current upstream `graphiti_core` creates an OpenAI client by default. In the checked official repo, extraction defaults to `gpt-4.1-mini`, small/reranking calls default to `gpt-4.1-nano`, and embeddings default to `text-embedding-3-small`. Older Graphiti configs may have used `gpt-4o-mini`, so set `GRAPHITI_LLM_MODEL=gpt-4o-mini` if you need that exact setup.

## Data

Download LoCoMo separately from the official repository:

```bash
git clone https://github.com/snap-research/locomo.git ../locomo
export MEMMARK_LOCOMO_PATH=$(realpath ../locomo/data/locomo10.json)
```

The data file is not vendored in this repository.

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

- `--backend {amem,graphiti}` selects the memory backend.
- `--baselines ...` selects one or more baselines.
- `--max-sessions` and `--max-qa` control run size.
- `--llm-mode stub` disables target-side fact extraction and QA calls for debugging; `--llm-mode real` is required for paper-style runs.
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
```

`model_name` is resolved from `RESULT_MODEL_NAME`, `TARGET_LLM_MODEL`, or `MEMMARK_MODEL`. It intentionally names the MemMark/QA target model, not the backend's private internal model. `time` is `RUN_TAG` / `MEMMARK_RUN_TAG` if set, otherwise the current timestamp (`YYYYmmdd-HHMMSS`).

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
- Report both backend-internal LLM settings and target/QA LLM settings.
- A-MEM runs can be slow because each turn may trigger memory evolution through the backend's native LLM path.
- Graphiti runs can be substantially slower and depend on Neo4j state; clear or isolate the graph between independent experiments if needed.
- Do not commit `.env`, `results/`, local model directories, or virtual environments.

## Citation

If you find MemMark useful, please cite:

```bibtex
@article{zhang2026memmark,
  title   = {MemMark: State-Evolution Attribution Watermarking for Agent Long-Term Memory Systems},
  author  = {Zhang, Haobo and Mao, Xutao and Dong, Guangyuan and Li, Ziwei and Su, Xuanbo and Chen, Kaijie and Yang, Jing and Lin, Zheng},
  journal = {arXiv preprint arXiv:2605.25002},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.25002},
  url     = {https://arxiv.org/abs/2605.25002}
}
```

## License

This repository is released under the MIT License. Third-party systems such as A-MEM, Graphiti, and LoCoMo are governed by their own licenses.
