"""Baselines for §10.3 / §10.5 RQ ablations.

* `signed_metadata_only` — same audit trace (commitment + Merkle log)
  but no watermark bits. Direct R3 ablation: does the marginal
  attribution signal come from the watermark or just the metadata?
* `random_replace` — uniform random candidate, no key. FPR / wrong-key
  floor.
* `no_watermark` — top-1 by p_t, no audit trace. Utility upper bound.

All three are implemented as `sampler_mode` flags on
`MemoryWatermarker` (memmark/sdk/memory_watermarker.py); this module
re-exports a thin convenience constructor.
"""

from memmark.baselines.factory import build_baseline

__all__ = ["build_baseline"]
