"""R2: Snapshot + Partial Reveal verifier (README §10.5).

Given a snapshot's signed Merkle root anchor (`SessionHeader`) and an
arbitrary subset of `(DecisionPoint, AuditRecord)` reveal records,
verify that:

1. each provided commitment recomputes to the audit value (binding)
2. its Merkle proof against the rebuilt log root matches the signed
   anchor (membership)
3. each `decoded_prefix` equals the expected payload slice (decode)

The verifier is degradation-tolerant: missing reveals just become
"unverified" entries; the report still produces an aggregate bit
recovery rate over the *covered* slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from memmark.core.commitment import make_commitment
from memmark.core.decoder import decode_memory_transition
from memmark.core.merkle_log import (
    merkle_proof,
    merkle_root,
    verify_signature,
)
from memmark.core.types import AuditRecord, DecisionPoint, SessionHeader


@dataclass(frozen=True)
class PartialLogVerificationResult:
    anchor_signature_valid: bool
    rebuilt_root: str
    anchor_root: str
    root_matches: bool
    leaf_results: List[dict]
    bits_recovered: int
    bits_total: int
    bit_recovery_rate: float


def verify_partial_log(
    *,
    full_decisions: Sequence[DecisionPoint],
    full_audits: Sequence[AuditRecord],
    revealed_indices: Sequence[int],
    anchor: SessionHeader,
    payload_bits: str,
    secret_key: str,
) -> PartialLogVerificationResult:
    """Verify a subset (`revealed_indices`) against the signed anchor.

    `full_decisions` / `full_audits` are aligned per-decision lists from
    the original session (the verifier reconstructs the leaf list from
    them; the *missing* reveals are still represented as commitment-only
    leaves so the Merkle path can be rebuilt).
    """

    leaves = [a.commitment for a in full_audits]
    rebuilt_root = merkle_root(leaves)
    sig_valid = verify_signature(secret_key, anchor.root, anchor.signature)

    leaf_results: List[dict] = []
    bits_recovered = 0
    bits_total = 0
    revealed_set = set(revealed_indices)

    # Use each audit's stored absolute position (bit_index_after - slice)
    # rather than a running sum, so partial audit sets stay aligned to
    # the original payload (see same note in verifier/in_record.py).
    for idx, audit in enumerate(full_audits):
        slice_len = audit.bits_embedded
        bit_start = max(0, audit.bit_index_after - slice_len)
        expected = payload_bits[bit_start : bit_start + slice_len]
        bits_total += slice_len

        if idx not in revealed_set:
            leaf_results.append(
                {
                    "index": idx,
                    "decision_id": audit.decision_id,
                    "revealed": False,
                    "expected": expected,
                }
            )
            continue

        decision = full_decisions[idx]
        rebuilt = make_commitment(
            decision,
            selected_candidate_id=audit.selected_candidate_id,
            bits_embedded=audit.bits_embedded,
            bit_index_after=audit.bit_index_after,
            keep_reveal=False,
        )
        commit_ok = rebuilt.commitment == audit.commitment

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

        if commit_ok and proof_ok and bits_match:
            bits_recovered += slice_len

        leaf_results.append(
            {
                "index": idx,
                "decision_id": audit.decision_id,
                "revealed": True,
                "commitment_valid": commit_ok,
                "merkle_proof_valid": proof_ok,
                "decoded_prefix": decoded_prefix,
                "expected": expected,
                "bits_match": bits_match,
            }
        )

    return PartialLogVerificationResult(
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
    )
