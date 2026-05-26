from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MemoryBackendAdapter(ABC):
    """Adapter contract per README §4.2.3.

    Required:
      * snapshot()   — list of memory records (dict-shaped)
      * apply(op)    — apply an evolve operation, return the affected
                       memory record. Must propagate any extra keys
                       like `dia_ids`, `session_index`, `speaker`,
                       `session_date_time` on the returned record so
                       RQ5 evidence checks work.

    Optional class attribute:
      * preferred_ingestion_mode  — "turn" / "session" / "fact"
        Driver dispatches on this to feed the backend in the way its
        upstream paper / repo uses for LoCoMo / LongMemEval:
          - "turn":    one memory event per LoCoMo turn  (Graphiti)
          - "session": one memory event per LoCoMo session text
          - "fact":    LoCoMo-style per-session LLM fact extraction
                       with dia_id evidence   (A-MEM, Mem0)

    Optional sampler attachment:
      * attach_sampler(sampler)  — wrap the backend's *internal* LLM
        client with the watermark sampler. Backends that drive their
        own evolution via LLM (A-MEM, Graphiti) override this
        to install :class:`memmark.llm.watermarked.WatermarkedSampler`
        at the SDK's LLM-call boundary. JsonStore is a no-op.
    """

    # Driver fallback when a backend doesn't override.
    preferred_ingestion_mode: str = "turn"

    @abstractmethod
    def snapshot(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------- watermark sampler injection ----------- #
    def attach_sampler(self, sampler: Any) -> None:
        """Default: no internal LLM → nothing to wrap."""

        return None

    # ------------- canonical QA context ------------------ #
    def qa_context(
        self,
        question: str,
        k: int = 10,
        *,
        category: Any = None,
        gold_answer: Any = None,
        llm_client: Any = None,
    ) -> Dict[str, Any]:
        """Return per-question retrieval context for the QA prompt.

        Returns ``{"mode": "context"|"answer", "text": str}``:

          * ``"context"`` — pre-rendered memory text. The driver wraps
            it in LoCoMo's QA_PROMPT and asks our own LLM.
          * ``"answer"``  — the backend ran its native QA pipeline
            (e.g. A-mem robust protocol with cat-aware prompts). The
            driver returns the answer verbatim, skipping the QA prompt.

        Optional kwargs (used by backends that implement a full
        official QA protocol, e.g. A-mem):

          * ``category``     — LoCoMo question category (1..5)
          * ``gold_answer``  — gold answer (only needed for cat-5
                               adversarial A/B protocol)
          * ``llm_client``   — separate raw LLM client (NOT the
                               watermark-wrapped one) for QA-time
                               keyword extraction + final answer

        Default: render the entire snapshot via session-marker
        grouping (matching LoCoMo's full-conversation
        Base/Observation rendering). Backends with a native canonical
        retrieve / QA API override this.
        """

        from memmark.benchmarks.locomo.qa_eval import _default_render_memory

        return {"mode": "context", "text": _default_render_memory(self.snapshot())}


def _string_topk(snapshot: List[Dict[str, Any]], query: str, k: int):
    """Cheap keyword-overlap top-k. Used by ``_default_render_memory``
    helpers that need to surface a few records by query relevance.
    """

    qwords = {w for w in (query or "").lower().split() if len(w) > 2}
    scored = []
    for record in snapshot:
        text = (record.get("text") or "").lower()
        if not text:
            continue
        rwords = {w for w in text.split() if len(w) > 2}
        score = len(qwords & rwords)
        if score == 0:
            continue
        scored.append((score, record))
    scored.sort(key=lambda kv: -kv[0])
    return [r for _, r in scored[:k]]
