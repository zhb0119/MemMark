from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Optional

from memmark.core.types import Candidate, DecisionPoint

try:
    from agentmark.core.watermark_sampler import sample_behavior_differential
except ModuleNotFoundError:
    sample_behavior_differential = None


@dataclass(frozen=True)
class MemorySampleResult:
    selected_candidate: Candidate
    target_candidate_ids: list[str]
    bits_embedded: int
    context_used: str


def sample_memory_transition(
    decision: DecisionPoint,
    *,
    payload_bits: str,
    bit_index: int,
    mode: str = "watermark",
    random_seed: Optional[int] = None,
) -> MemorySampleResult:
    """Pick one candidate from `decision.candidates`.

    Modes (README §10.3 baselines):

    - `watermark` — AgentMark distribution-preserving keyed sampling
      (real research mode); falls back to keyed-shift if AgentMark
      is missing.
    - `signed_metadata_only` — sample by p_t with no key (no bits
      embedded). Audit trace + Merkle log still built. Direct ablation
      vs `watermark` for §10.5 RQ3 marginal gain.
    - `random_replace` — uniform random pick, no key. FPR / wrong-key
      floor.
    - `no_watermark` — top-1 by p_t (deterministic, what the backend
      would have done without MemMark). Utility upper bound.
    """

    candidate_by_id = {c.candidate_id: c for c in decision.candidates}
    if mode == "watermark":
        return _sample_watermarked(decision, payload_bits, bit_index, candidate_by_id)
    if mode == "signed_metadata_only":
        return _sample_distribution(
            decision, candidate_by_id, random_seed=random_seed
        )
    if mode == "random_replace":
        return _sample_uniform(decision, candidate_by_id, random_seed=random_seed)
    if mode == "no_watermark":
        return _sample_argmax(decision, candidate_by_id)
    raise ValueError(f"Unknown sampler mode: {mode}")


# -- watermark mode ------------------------------------------------- #


def _sample_watermarked(
    decision: DecisionPoint,
    payload_bits: str,
    bit_index: int,
    candidate_by_id: dict,
) -> MemorySampleResult:
    # Mix nonce_t into the keying material so the watermark is actually
    # secret-keyed: wrong K ⇒ wrong nonce ⇒ wrong selected candidate
    # ⇒ decode failure on the wrong-key control.
    keyed_context = (decision.nonce + "|" + decision.context) if decision.nonce else decision.context
    if sample_behavior_differential is not None:
        selected_id, target_ids, bits_embedded, context_used = sample_behavior_differential(
            probabilities=decision.probabilities,
            bit_stream=payload_bits,
            bit_index=bit_index,
            context_for_key=keyed_context,
            round_num=decision.round_num,
        )
    else:
        selected_id, target_ids, bits_embedded, context_used = _sample_fallback(
            decision,
            payload_bits=payload_bits,
            bit_index=bit_index,
            keyed_context=keyed_context,
        )
    return MemorySampleResult(
        selected_candidate=candidate_by_id[selected_id],
        target_candidate_ids=list(target_ids),
        bits_embedded=bits_embedded,
        context_used=context_used,
    )


def _sample_fallback(
    decision: DecisionPoint,
    *,
    payload_bits: str,
    bit_index: int,
    keyed_context: Optional[str] = None,
) -> tuple[str, list[str], int, str]:
    """Keyed-shift fallback when AgentMark sampler isn't installed.

    NOTE: not strictly distribution-preserving; flagged for development
    only. See §7.1 for the proper guarantee that AgentMark provides.
    """

    candidate_ids = sorted(decision.probabilities)
    n = len(candidate_ids)
    if n < 2:
        return candidate_ids[0], candidate_ids, 0, decision.context
    k = int(math.floor(math.log2(n)))
    available = max(0, len(payload_bits) - bit_index)
    bits_embedded = min(k, available)
    if bits_embedded == 0:
        idx_value = 0
    else:
        idx_value = int(payload_bits[bit_index : bit_index + bits_embedded], 2)
    keying = keyed_context if keyed_context is not None else decision.context
    shift = _context_shift(keying, decision.round_num, n)
    selected_idx = (idx_value + shift) % n
    return candidate_ids[selected_idx], candidate_ids, bits_embedded, decision.context


def _context_shift(context: str, round_num: int, n: int) -> int:
    digest = hashlib.sha256(f"{context}|{round_num}".encode("utf-8")).hexdigest()
    return int(digest, 16) % n


# -- baseline modes ------------------------------------------------- #


def _sample_distribution(
    decision: DecisionPoint,
    candidate_by_id: dict,
    *,
    random_seed: Optional[int] = None,
) -> MemorySampleResult:
    rng = _rng(decision, random_seed)
    ids, probs = _sorted_probs(decision)
    pick = rng.random()
    cum = 0.0
    selected_id = ids[-1]
    for cid, p in zip(ids, probs):
        cum += p
        if pick <= cum:
            selected_id = cid
            break
    return MemorySampleResult(
        selected_candidate=candidate_by_id[selected_id],
        target_candidate_ids=ids,
        bits_embedded=0,
        context_used=decision.context,
    )


def _sample_uniform(
    decision: DecisionPoint,
    candidate_by_id: dict,
    *,
    random_seed: Optional[int] = None,
) -> MemorySampleResult:
    rng = _rng(decision, random_seed)
    ids = sorted(decision.probabilities)
    selected_id = rng.choice(ids)
    return MemorySampleResult(
        selected_candidate=candidate_by_id[selected_id],
        target_candidate_ids=ids,
        bits_embedded=0,
        context_used=decision.context,
    )


def _sample_argmax(
    decision: DecisionPoint, candidate_by_id: dict
) -> MemorySampleResult:
    ids, probs = _sorted_probs(decision)
    selected_id = max(zip(ids, probs), key=lambda kv: kv[1])[0]
    return MemorySampleResult(
        selected_candidate=candidate_by_id[selected_id],
        target_candidate_ids=ids,
        bits_embedded=0,
        context_used=decision.context,
    )


def _sorted_probs(decision: DecisionPoint) -> tuple[list, list]:
    ids = sorted(decision.probabilities)
    probs = [decision.probabilities[k] for k in ids]
    total = sum(probs)
    if total <= 0:
        # Safety: degrade to uniform
        n = len(probs)
        return ids, [1.0 / max(n, 1)] * n
    return ids, [p / total for p in probs]


def _rng(decision: DecisionPoint, random_seed: Optional[int]) -> random.Random:
    if random_seed is not None:
        return random.Random(random_seed + decision.round_num)
    digest = hashlib.sha256(
        f"baseline|{decision.context}|{decision.round_num}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))
