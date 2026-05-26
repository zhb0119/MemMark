"""Minimal setup for installing the WujiangXu/AgenticMemory eval repo.

The eval repo has shipped in two layouts:

  * old flat modules (``memory_layer.py``, ``memory_layer_robust.py``, etc.),
  * current ``agentic_memory`` package layout.

This setup file detects whichever layout is present.

After installing this on top of a clone of WujiangXu/AgenticMemory:

    >>> from agentic_memory.memory_system import AgenticMemorySystem
    >>> hasattr(AgenticMemorySystem, "find_related_memories_raw")
    True

(The agiresearch SDK only exposes ``find_related_memories``; the eval
repo's class additionally has ``find_related_memories_raw`` which the
A-mem paper's LoCoMo eval (test_advanced_robust.py) uses.)
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
PY_MODULE_NAMES = [
    "memory_layer",
    "memory_layer_robust",
    "utils",
    "llm_text_parsers",
    "load_dataset",
]
py_modules = [name for name in PY_MODULE_NAMES if (ROOT / f"{name}.py").exists()]
packages = find_packages(include=["agentic_memory", "agentic_memory.*"])


setup(
    name="agentic-memory",
    version="0.0.3",
    description=(
        "WujiangXu/AgenticMemory eval-repo install (provides"
        " find_related_memories_raw); shimmed under the SDK module path."
    ),
    py_modules=py_modules,
    packages=packages,
    install_requires=[
        "rank_bm25>=0.2,<0.3",
        "torch>=2.0,<2.4",
        # sentence-transformers 5.x pulls transformers 5.x, which requires
        # newer PyTorch than many CUDA 12.1 research images provide.
        "sentence-transformers>=2.2.2,<3.0.0",
        "transformers>=4.30.0,<5.0.0",
        "huggingface-hub>=0.15.1,<1.0",
        "scikit-learn>=1.0,<2.0",
        "nltk>=3.8,<4.0",
        "litellm>=1.0,<2.0",
        "openai>=1.0,<2.0",
        "requests>=2.31,<3.0",
        "chromadb>=0.4.24,<0.6.0",
    ],
    python_requires=">=3.9",
)
