"""R3: In-Record Attribution Verification (README §10.5, headline).

The verifier is given **only** a memory snapshot (the audit records,
each carrying its in-record reveal residue: candidates + probabilities
+ ctx + nonce + watermark_version) plus the signed `SessionHeader`
anchor. No external reveal log is consulted.

This implements the §10.5 R3 protocol:

  1. extract `(reveal_t, anchor)` pair from snapshot
  2. for each reveal, replay the keyed sampler and check selected
     candidate is consistent with the watermark bit slice
  3. reconstruct the per-leaf commitment from the in-record reveal and
     check its Merkle proof against the anchor's signed root
  4. aggregate bit-recovery + carrier-level breakdown

The point of R3 is that AgentMark / ActHook *as published* cannot do
this — they require a separate action trajectory that gets discarded
in real forensic scenarios.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from memmark.core.commitment import make_commitment
from memmark.core.context import derive_nonce
from memmark.core.decoder import decode_memory_transition
from memmark.core.merkle_log import (
    merkle_proof,
    merkle_root,
    verify_signature,
)
from memmark.core.types import AuditRecord, DecisionPoint, SessionHeader


@dataclass(frozen=True)
class InRecordVerificationResult:
    anchor_signature_valid: bool
    rebuilt_root: str
    anchor_root: str
    root_matches: bool
    leaf_results: List[dict]
    bits_recovered: int
    bits_total: int
    bit_recovery_rate: float
    carrier_breakdown: Dict[str, dict] = field(default_factory=dict)


def verify_in_record(
    *,
    audit_records: Sequence[AuditRecord],
    anchor: SessionHeader,
    payload_bits: str,
    secret_key: str,
) -> InRecordVerificationResult:
    """Verify with **no external log** — uses only what is co-located
    with the memory records (candidates / probabilities / ctx / nonce
    inside each AuditRecord) plus the signed Merkle anchor.
    """

    if any(r.candidates is None or r.probabilities is None for r in audit_records):
        raise ValueError(
            "verify_in_record requires AuditRecord entries with embedded reveal "
            "(candidates + probabilities). Use make_commitment(..., keep_reveal=True)."
        )

    sig_valid = verify_signature(secret_key, anchor.root, anchor.signature)
    leaves = [r.commitment for r in audit_records]
    rebuilt_root = merkle_root(leaves)

    bits_recovered = 0
    bits_total = 0
    leaf_results: List[dict] = []
    carrier_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"bits_recovered": 0, "bits_total": 0, "leaves": 0}
    )

    # Each audit records its ABSOLUTE position in payload_bits via
    # ``bit_index_after`` (= prefix length after this audit's bits were
    # appended at seal time). We must NOT use a running sum over the
    # iterated audit list — when the attacker prunes leaves the running
    # sum compresses positions and surviving audits get matched against
    # wrong slices of payload_bits, killing bits_match even though
    # Method B inclusion proofs still verify. Use each audit's stored
    # absolute position so partial audit sets stay aligned to the
    # original payload.
    for idx, audit in enumerate(audit_records):
        slice_len = audit.bits_embedded
        bit_start = max(0, audit.bit_index_after - slice_len)
        expected = payload_bits[bit_start : bit_start + slice_len]
        bits_total += slice_len

        # The verifier always re-derives nonce_t from the supplied secret
        # key; this is what makes the wrong-key control meaningful (a
        # wrong K produces a wrong nonce, which produces a wrong decode).
        verifier_nonce = derive_nonce(secret_key, audit.context)
        decision = DecisionPoint(
            decision_id=audit.decision_id,
            tau=audit.tau,
            candidates=list(audit.candidates or []),
            probabilities=dict(audit.probabilities or {}),
            context=audit.context,
            round_num=audit.round_num,
            nonce=verifier_nonce,
            watermark_version=audit.watermark_version,
        )

        # For commitment recomputation we still use the recorded nonce so
        # the commit-binding check is independent of which key the
        # verifier holds.
        decision_for_commit = DecisionPoint(
            decision_id=audit.decision_id,
            tau=audit.tau,
            candidates=list(audit.candidates or []),
            probabilities=dict(audit.probabilities or {}),
            context=audit.context,
            round_num=audit.round_num,
            nonce=audit.nonce,
            watermark_version=audit.watermark_version,
        )
        rebuilt = make_commitment(
            decision_for_commit,
            selected_candidate_id=audit.selected_candidate_id,
            bits_embedded=audit.bits_embedded,
            bit_index_after=audit.bit_index_after,
            keep_reveal=False,
        )
        commit_ok = rebuilt.commitment == audit.commitment

        # Prefer the seal-time per-leaf inclusion proof stored on the
        # audit (Method B): each leaf verifies independently against
        # anchor.root, so structural attacks like pruning / dedup /
        # poisoning produce smooth bit_recovery degradation instead of
        # a rebuilt-root mismatch killing every remaining leaf at once.
        # Fallback to rebuilt-tree proof for legacy audits without the
        # stored proof.
        stored_proof = getattr(audit, "merkle_inclusion_proof", None)
        if stored_proof is not None:
            proof_ok = stored_proof.verify() and stored_proof.root == anchor.root
        else:
            try:
                proof = merkle_proof(leaves, idx)
                proof_ok = proof.verify() and proof.root == anchor.root
            except IndexError:
                proof_ok = False

        decoded_bits = decode_memory_transition(
            decision, selected_candidate_id=audit.selected_candidate_id
        )
        decoded_prefix = decoded_bits[: len(expected)]
        bits_match = decoded_prefix == expected

        leaf_ok = commit_ok and proof_ok and bits_match
        if leaf_ok:
            bits_recovered += slice_len

        carrier_stats[audit.tau]["leaves"] += 1
        carrier_stats[audit.tau]["bits_total"] += slice_len
        if leaf_ok:
            carrier_stats[audit.tau]["bits_recovered"] += slice_len

        leaf_results.append(
            {
                "index": idx,
                "decision_id": audit.decision_id,
                "tau": audit.tau,
                "commitment_valid": commit_ok,
                "merkle_proof_valid": proof_ok,
                "decoded_prefix": decoded_prefix,
                "expected": expected,
                "bits_match": bits_match,
            }
        )

    carrier_breakdown = {
        tau: {
            "leaves": s["leaves"],
            "bits_total": s["bits_total"],
            "bits_recovered": s["bits_recovered"],
            "bit_recovery_rate": (
                float(s["bits_recovered"]) / s["bits_total"]
                if s["bits_total"] else 0.0
            ),
        }
        for tau, s in carrier_stats.items()
    }

    return InRecordVerificationResult(
        anchor_signature_valid=sig_valid,
        rebuilt_root=rebuilt_root,
        anchor_root=anchor.root,
        root_matches=rebuilt_root == anchor.root,
        leaf_results=leaf_results,
        bits_recovered=bits_recovered,
        bits_total=bits_total,
        bit_recovery_rate=(
            float(bits_recovered) / bits_total if bits_total else 0.0
        ),
        carrier_breakdown=carrier_breakdown,
    )
