"""RQ2: Capacity (README §10.4).

Reports per-carrier:
  * average |C_t| (candidate set size)
  * average H(p_t)  (LLM-assigned distribution entropy)
  * acceptance rate Pr[|C_t| ≥ 2]
  * realised bits/decision

Plus aggregate session-level numbers. The capacity stats are already
collected during driver execution; this RQ runner just shapes them
into the §10.4 table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from memmark.benchmarks.locomo.driver import LoCoMoDriverResult


@dataclass
class RQ2Report:
    overall: Dict[str, float] = field(default_factory=dict)
    by_carrier: Dict[str, Dict[str, float]] = field(default_factory=dict)


def run_rq2_capacity(driver_result: LoCoMoDriverResult) -> RQ2Report:
    stats = driver_result.capacity_stats or {}
    overall = {
        "decisions": float(stats.get("decisions", 0)),
        "bits_embedded": float(stats.get("bits_embedded", 0)),
        "bits_per_decision": float(stats.get("bits_per_decision", 0.0)),
        "avg_candidate_set_size": float(stats.get("avg_candidate_set_size", 0.0)),
        "avg_entropy": float(stats.get("avg_entropy", 0.0)),
        "acceptance_rate": float(stats.get("acceptance_rate", 0.0)),
    }
    by_carrier = {
        tau: {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in info.items()
        }
        for tau, info in (stats.get("by_carrier") or {}).items()
    }
    return RQ2Report(overall=overall, by_carrier=by_carrier)
