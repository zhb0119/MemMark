"""Decoder counterpart to memmark.core.sampler.

Mirrors the AgentMark sampler's encode→decode contract: given the
same `(probabilities, context, round_num)` plus the actually-selected
candidate, recover the bits that were embedded.

Falls back to a keyed-shift decoder if AgentMark (and torch) is not
installed — same caveat as `sampler._sample_fallback`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

from memmark.core.types import DecisionPoint  # noqa: F401

try:
    from agentmark.core.watermark_sampler import differential_based_decoder  # type: ignore
except ModuleNotFoundError:
    differential_based_decoder = None  # type: ignore


def decode_memory_transition(
    decision: DecisionPoint,
    *,
    selected_candidate_id: str,
    override_nonce: Optional[str] = None,
) -> str:
    """Decode the bits embedded in this decision.

    `override_nonce` lets the wrong-key control replay decoding with a
    different nonce; if None we use `decision.nonce` as recorded.
    """

    nonce = decision.nonce if override_nonce is None else override_nonce
    keyed_context = (nonce + "|" + decision.context) if nonce else decision.context
    if differential_based_decoder is not None:
        return differential_based_decoder(
            probabilities=decision.probabilities,
            selected_behavior=selected_candidate_id,
            context_for_key=keyed_context,
            round_num=decision.round_num,
        )
    return _decode_fallback(decision, selected_candidate_id, keyed_context)


def _decode_fallback(
    decision: DecisionPoint,
    selected_candidate_id: str,
    keyed_context: Optional[str] = None,
) -> str:
    candidate_ids = sorted(decision.probabilities)
    n = len(candidate_ids)
    if n < 2 or selected_candidate_id not in candidate_ids:
        return ""
    k = int(math.floor(math.log2(n)))
    if k <= 0:
        return ""
    selected_idx = candidate_ids.index(selected_candidate_id)
    keying = keyed_context if keyed_context is not None else decision.context
    shift = _context_shift(keying, decision.round_num, n)
    idx_value = (selected_idx - shift) % n
    return format(idx_value, f"0{k}b")


def _context_shift(context: str, round_num: int, n: int) -> int:
    digest = hashlib.sha256(f"{context}|{round_num}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n
