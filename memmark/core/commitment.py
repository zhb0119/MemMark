from __future__ import annotations

from typing import Dict, List

from memmark.core.context import sha256_text, stable_json
from memmark.core.types import AuditRecord, Candidate, DecisionPoint


def hash_candidates(candidates: List[Candidate]) -> str:
    payload = [
        {
            "candidate_id": c.candidate_id,
            "carrier_type": c.carrier_type,
            "payload": c.payload,
            "operation": c.operation,
            "utility_score": c.utility_score,
        }
        for c in candidates
    ]
    return sha256_text(stable_json(payload))


def hash_probabilities(probabilities: Dict[str, float]) -> str:
    return sha256_text(
        stable_json({k: round(float(v), 12) for k, v in probabilities.items()})
    )


def make_commitment(
    decision: DecisionPoint,
    *,
    selected_candidate_id: str,
    bits_embedded: int,
    bit_index_after: int,
    keep_reveal: bool = True,
    extra_carriers: tuple = (),
) -> AuditRecord:
    """Compute commit_t per README §9.1.

    commit_t = SHA256(
        ctx_hash || H(C_t) || H(p_t)
        || selected_idx || bits_embedded
        || nonce_t || watermark_version
        || round_num
    )

    `nonce_t` and `watermark_version` are taken from the DecisionPoint;
    they were derived via HMAC and the version template respectively in
    the upstream MemoryWatermarker.

    `keep_reveal=True` stores the full reveal record (candidates +
    probabilities) inside the AuditRecord for In-Record Attribution
    Verification (README §10.5 R3). For R1 / R2 reveals can also be
    re-loaded from a separate store.
    """

    candidate_hash = hash_candidates(decision.candidates)
    probability_hash = hash_probabilities(decision.probabilities)
    context_hash = sha256_text(decision.context)
    body = {
        "decision_id": decision.decision_id,
        "tau": decision.tau,
        "candidate_hash": candidate_hash,
        "probability_hash": probability_hash,
        "context_hash": context_hash,
        "selected_candidate_id": selected_candidate_id,
        "bits_embedded": bits_embedded,
        "bit_index_after": bit_index_after,
        "round_num": decision.round_num,
        "nonce": decision.nonce,
        "watermark_version": decision.watermark_version,
    }
    commitment = sha256_text(stable_json(body))
    return AuditRecord(
        decision_id=decision.decision_id,
        tau=decision.tau,
        candidate_hash=candidate_hash,
        probability_hash=probability_hash,
        context=decision.context,
        context_hash=context_hash,
        selected_candidate_id=selected_candidate_id,
        bits_embedded=bits_embedded,
        bit_index_after=bit_index_after,
        round_num=decision.round_num,
        commitment=commitment,
        nonce=decision.nonce,
        watermark_version=decision.watermark_version,
        candidates=list(decision.candidates) if keep_reveal else None,
        probabilities=dict(decision.probabilities) if keep_reveal else None,
        extra_carriers=tuple(extra_carriers),
    )
