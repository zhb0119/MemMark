"""Shared base result + leaf-level helpers for anchor-based verifiers.

Both the §10.5 R2 (partial-log) and R3 (in-record) verifiers walk over
``AuditRecord`` leaves and check the same three things — commitment
binding, Merkle inclusion against the signed anchor, and decoded
bit-slice match. This module factors out the pieces that don't depend
on per-verifier policy (R2 skips non-revealed leaves, R3 re-derives
nonces) so the two verifier modules only carry their differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from memmark.core.merkle_log import merkle_proof
from memmark.core.types import AuditRecord


@dataclass(frozen=True)
class AnchorVerificationResult:
    """Common shape for verifiers that check leaves against a signed root."""

    anchor_signature_valid: bool
    rebuilt_root: str
    anchor_root: str
    root_matches: bool
    leaf_results: List[dict]
    bits_recovered: int
    bits_total: int
    bit_recovery_rate: float


def expected_payload_slice(audit: AuditRecord, payload_bits: str) -> Tuple[str, int]:
    """Return ``(expected_slice, slice_len)`` for an audit's absolute position.

    Uses ``audit.bit_index_after - bits_embedded`` rather than a running
    sum over the iterated audit list — see the note in
    ``verifier/in_record.py`` for why running sums break under pruning.
    """

    slice_len = audit.bits_embedded
    bit_start = max(0, audit.bit_index_after - slice_len)
    return payload_bits[bit_start : bit_start + slice_len], slice_len


def verify_inclusion_proof(
    audit: AuditRecord,
    leaves: Sequence[str],
    idx: int,
    anchor_root: str,
) -> bool:
    """Verify a leaf's Merkle inclusion against ``anchor_root``.

    Prefer the seal-time per-leaf proof stored on the audit (Method B);
    fall back to a fresh proof rebuilt from ``leaves`` for legacy audits.
    """

    stored_proof = getattr(audit, "merkle_inclusion_proof", None)
    if stored_proof is not None:
        return stored_proof.verify() and stored_proof.root == anchor_root
    try:
        proof = merkle_proof(list(leaves), idx)
    except IndexError:
        return False
    return proof.verify() and proof.root == anchor_root


def bit_recovery_rate(bits_recovered: int, bits_total: int) -> float:
    return float(bits_recovered) / bits_total if bits_total else 0.0
