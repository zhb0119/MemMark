"""Backend adapters bridging MemMark's `(C_t, p_t, ctx_t)` interface to
real memory systems.

Each backend exposes the minimum API from README §4.2.3:
  - snapshot() -> list of memory records
  - apply(operation) -> applied memory record

Optional backends gate their imports on dependency availability.
"""

from memmark.backends.base import MemoryBackendAdapter

__all__ = ["MemoryBackendAdapter"]


def load_amem(**kwargs):
    """Lazy import to avoid hard dependency."""
    from memmark.backends.amem_store import AMemBackend

    return AMemBackend(**kwargs)


def load_graphiti(**kwargs):
    from memmark.backends.graphiti_store import GraphitiBackend

    return GraphitiBackend(**kwargs)
