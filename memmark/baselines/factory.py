"""Convenience factory for baseline configurations.

`build_baseline(name, **wm_kwargs)` returns a configured
`MemoryWatermarker`. The actual sampler logic lives inside
`memmark.core.sampler.sample_memory_transition`; this factory just
picks the `sampler_mode` flag.
"""

from __future__ import annotations

from typing import Any

from memmark.sdk.memory_watermarker import MemoryWatermarker


_BASELINE_TO_MODE = {
    # Full MemMark: SDK-internal candidate enumeration + keyed PRF pick
    # + commitment/reveal sidecar. Used as +memory-watermark in RQ1-RQ5.
    "watermark": "watermark",
    "memory-watermark": "watermark",
    "memmark": "watermark",
    # Metadata/provenance ablation: same granularity and audit sidecar,
    # but no embedded bits. This is the RQ3/RQ5 comparison against pure
    # signed provenance.
    "signed_metadata_only": "signed_metadata_only",
    "signed-metadata-only": "signed_metadata_only",
    "metadata-only": "signed_metadata_only",
    # Random choice at the same evolve-decision points, no key and no
    # payload bits. This is the FPR / wrong-key lower-bound baseline.
    "random_replace": "random_replace",
    "random-replace": "random_replace",
    # Native framework baseline: no MemMark prompt wrapping and no audit
    # decisions. For Graphiti this must remain official add_episode
    # behavior; for A-MEM it remains the upstream robust LoCoMo path.
    "no_watermark": "no_watermark",
    "no-watermark": "no_watermark",
    "official": "no_watermark",
    "native-official": "no_watermark",
    # KG-backend baseline for README §9.1: dynamic KG watermarking on
    # Graphiti. This is intentionally separate from MemMark's internal
    # LLM-call watermark; the Graphiti adapter applies it at KG write
    # time after native add_episode() succeeds.
    "kgmark_graphiti": "kgmark_graphiti",
    "kgmark-graphiti": "kgmark_graphiti",
    "kgmark": "kgmark_graphiti",
}


def build_baseline(name: str, **kwargs: Any) -> MemoryWatermarker:
    if name not in _BASELINE_TO_MODE:
        raise ValueError(
            f"Unknown baseline: {name!r}. "
            f"Choose from {sorted(_BASELINE_TO_MODE)}."
        )
    kwargs["sampler_mode"] = _BASELINE_TO_MODE[name]
    return MemoryWatermarker(**kwargs)
