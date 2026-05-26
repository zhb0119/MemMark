from __future__ import annotations

from dataclasses import dataclass

from memmark.core.commitment import make_commitment
from memmark.core.decoder import decode_memory_transition
from memmark.core.types import AuditRecord, DecisionPoint


@dataclass(frozen=True)
class FullLogVerificationResult:
    commitment_valid: bool
    decoded_bits: str
    expected_prefix: str
    bits_match: bool


def verify_full_log(
    decision: DecisionPoint,
    audit: AuditRecord,
    *,
    payload_bits: str,
    previous_bit_index: int,
) -> FullLogVerificationResult:
    rebuilt = make_commitment(
        decision,
        selected_candidate_id=audit.selected_candidate_id,
        bits_embedded=audit.bits_embedded,
        bit_index_after=audit.bit_index_after,
    )
    decoded_bits = decode_memory_transition(
        decision,
        selected_candidate_id=audit.selected_candidate_id,
    )
    expected = payload_bits[previous_bit_index : previous_bit_index + audit.bits_embedded]
    decoded_prefix = decoded_bits[: len(expected)]
    return FullLogVerificationResult(
        commitment_valid=rebuilt.commitment == audit.commitment,
        decoded_bits=decoded_bits,
        expected_prefix=expected,
        bits_match=decoded_prefix == expected,
    )
