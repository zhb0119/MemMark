"""RQ3: In-Record / Partial-Log / Full-Log Verification (README §10.5).

This is the headline RQ. Produces three verifier outputs:

  R1 (Full External Log):    100% reveals available
  R2 (Partial External Log): r ∈ {0.9, 0.7, 0.5, 0.3, 0.1}
  R3 (In-Record):            no external log, in-record sidecar only

Plus a wrong-key control: the same R3 protocol with a different secret
gives the FPR floor. AgentMark @ action layer is reported separately
in the cross-baseline runner, since it consumes a different trace.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from memmark.benchmarks.locomo.driver import LoCoMoDriverResult
from memmark.verifier.full_log import verify_full_log
from memmark.verifier.in_record import verify_in_record
from memmark.verifier.partial_log import verify_partial_log


@dataclass
class AgentMarkRQ3Report:
    """RQ3-style summary for AgentMark / action-layer baselines.

    AgentMark is trajectory-bound, so the headline R3 score is
    expected to collapse to 0 when the action trace is unavailable.
    This adapter keeps the reporting shape aligned with MemMark's RQ3
    tables while making the baseline explicit.
    """

    r1: Dict[str, float] = field(default_factory=dict)
    r2: Dict[str, Dict[str, float]] = field(default_factory=dict)
    r3: Dict[str, float] = field(default_factory=dict)
    r3_wrong_key: Dict[str, float] = field(default_factory=dict)
    source: str = "agentmark"


_DEFAULT_R2_RATES = (0.9, 0.7, 0.5, 0.3, 0.1)


@dataclass
class RQ3Report:
    r1: Dict[str, float] = field(default_factory=dict)
    r2: Dict[str, Dict[str, float]] = field(default_factory=dict)
    r3: Dict[str, float] = field(default_factory=dict)
    r3_wrong_key: Dict[str, float] = field(default_factory=dict)
    r3_carrier_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)


def run_rq3_in_record(
    *,
    driver_result: LoCoMoDriverResult,
    secret_key: str,
    wrong_key: str = "memmark-wrong-key-control",
    r2_rates: Sequence[float] = _DEFAULT_R2_RATES,
    seed: int = 0,
) -> RQ3Report:
    payload_bits = driver_result.payload_bits
    audits = driver_result.audits
    decisions = driver_result.decisions
    anchor = driver_result.anchor
    if anchor is None or not audits:
        return RQ3Report()

    # ---- R1 ---- #
    r1_match = 0
    r1_total = 0
    bit_index = 0
    r1_commit_ok = 0
    for decision, audit in zip(decisions, audits):
        slice_len = audit.bits_embedded
        prev_bit_index = bit_index
        bit_index += slice_len
        verification = verify_full_log(
            decision,
            audit,
            payload_bits=payload_bits,
            previous_bit_index=prev_bit_index,
        )
        r1_total += slice_len
        if verification.bits_match:
            r1_match += slice_len
        if verification.commitment_valid:
            r1_commit_ok += 1
    r1 = {
        "bit_recovery_rate": r1_match / r1_total if r1_total else 0.0,
        "commitment_pass_rate": (
            r1_commit_ok / len(audits) if audits else 0.0
        ),
        "bits_total": float(r1_total),
        "bits_recovered": float(r1_match),
    }

    # ---- R2 ---- #
    rng = random.Random(seed)
    r2: Dict[str, Dict[str, float]] = {}
    n = len(audits)
    for rate in r2_rates:
        keep = max(1, int(round(rate * n)))
        revealed_indices = rng.sample(range(n), keep)
        result = verify_partial_log(
            full_decisions=decisions,
            full_audits=audits,
            revealed_indices=revealed_indices,
            anchor=anchor,
            payload_bits=payload_bits,
            secret_key=secret_key,
        )
        r2[f"r={rate}"] = {
            "kept_leaves": float(keep),
            "anchor_signature_valid": float(result.anchor_signature_valid),
            "root_matches": float(result.root_matches),
            "bit_recovery_rate": result.bit_recovery_rate,
            "bits_recovered": float(result.bits_recovered),
            "bits_total": float(result.bits_total),
        }

    # ---- R3 ---- #
    r3_result = verify_in_record(
        audit_records=audits,
        anchor=anchor,
        payload_bits=payload_bits,
        secret_key=secret_key,
    )
    r3 = {
        "anchor_signature_valid": float(r3_result.anchor_signature_valid),
        "root_matches": float(r3_result.root_matches),
        "bit_recovery_rate": r3_result.bit_recovery_rate,
        "bits_recovered": float(r3_result.bits_recovered),
        "bits_total": float(r3_result.bits_total),
    }

    # ---- R3 wrong-key control ---- #
    r3_wrong = verify_in_record(
        audit_records=audits,
        anchor=anchor,
        payload_bits=payload_bits,
        secret_key=wrong_key,
    )
    r3_wrong_key = {
        "anchor_signature_valid": float(r3_wrong.anchor_signature_valid),
        "bit_recovery_rate": r3_wrong.bit_recovery_rate,
    }

    return RQ3Report(
        r1=r1,
        r2=r2,
        r3=r3,
        r3_wrong_key=r3_wrong_key,
        r3_carrier_breakdown=r3_result.carrier_breakdown,
    )
