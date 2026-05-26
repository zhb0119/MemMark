"""RQ5: Memory Integrity (README §10.7).

Reports whether the watermark introduces wrong-target updates, broken
links, duplicates, or contradictions. Some of these need ground-truth
labels; for LoCoMo we use simple heuristics over the final memory
snapshot + audit trace:

  * duplication_rate       — fraction of memory records with text
                              appearing >1 times in the snapshot
  * contradiction_rate     — placeholder (LoCoMo doesn't ship clean
                              contradiction labels; reserved for the
                              §10.7 appendix using LongMemEval's
                              knowledge-update splits).
  * update_target_accuracy — when carrier=update_target was chosen,
                              whether the picked target's text shares
                              ≥1 content keyword with the source event
                              prompt (heuristic; LoCoMo doesn't ship
                              ground-truth update targets).
  * link_target_accuracy   — same heuristic for carrier=link_target:
                              whether the chosen link target shares
                              ≥1 content keyword with the source event.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from memmark.benchmarks.locomo.driver import LoCoMoDriverResult


@dataclass
class RQ5Report:
    duplication_rate: float = 0.0
    duplicate_count: int = 0
    update_target_accuracy: float = 0.0
    update_target_total: int = 0
    update_target_correct: int = 0
    link_target_accuracy: float = 0.0
    link_target_total: int = 0
    link_target_correct: int = 0
    contradiction_rate: float = 0.0  # placeholder
    overall_records: int = 0
    by_carrier_counts: Dict[str, int] = field(default_factory=dict)
    # Evidence-grounded checks (P2 #4): use LoCoMo's `evidence` dia_id
    # markers to verify that the memory used for QA actually contains
    # the right source dialogue turns.
    evidence_recall_mean: float = 0.0
    evidence_required_qas: int = 0
    qa_with_full_evidence: int = 0


def run_rq5_integrity(driver_result: LoCoMoDriverResult) -> RQ5Report:
    snapshot = driver_result.memory_snapshot_final or []
    overall = len(snapshot)
    report = RQ5Report(overall_records=overall)
    if overall == 0:
        return report

    # Duplication: same `text` appearing more than once
    counts = Counter((r.get("text") or "").strip() for r in snapshot)
    duplicate_count = sum(c for c in counts.values() if c > 1) - sum(
        1 for c in counts.values() if c > 1
    )  # extra occurrences beyond the first
    report.duplicate_count = duplicate_count
    report.duplication_rate = duplicate_count / overall

    # Carrier breakdown of decisions — multi-label: count an audit
    # in every carrier bucket it lists (primary tau + extra_carriers).
    by_carrier: Counter = Counter()
    for a in driver_result.audits:
        for c in _carriers_of(a):
            by_carrier[c] += 1
    report.by_carrier_counts = dict(by_carrier)

    # update_target / link_target accuracy heuristic: when the
    # carrier was chosen, did the picked target's text share ≥1
    # content keyword with the source event prompt? Both carriers use
    # the same scheme. ``selected.payload["text"]`` is the chosen LLM
    # output (the SDK schema-shaped string); the source event prompt
    # is reconstructed from ``audit.context`` (a JSON-serialized
    # ctx_payload from WatermarkedSampler.intercept).
    update_total = update_correct = 0
    link_total = link_correct = 0
    for decision, audit in zip(driver_result.decisions, driver_result.audits):
        carriers = _carriers_of(audit)
        is_update = "update_target" in carriers
        is_link = "link_target" in carriers
        if not (is_update or is_link):
            continue
        selected = next(
            (c for c in decision.candidates if c.candidate_id == audit.selected_candidate_id),
            None,
        )
        target_text = (selected.payload.get("text", "") if selected else "") or ""
        event_text = _event_text_from_context(audit.context)
        ok = _shares_keywords(target_text, event_text)
        if is_update:
            update_total += 1
            if ok:
                update_correct += 1
        if is_link:
            link_total += 1
            if ok:
                link_correct += 1
    report.update_target_total = update_total
    report.update_target_correct = update_correct
    report.update_target_accuracy = (
        update_correct / update_total if update_total else 0.0
    )
    report.link_target_total = link_total
    report.link_target_correct = link_correct
    report.link_target_accuracy = (
        link_correct / link_total if link_total else 0.0
    )

    # ---- evidence-grounded checks (P2 #4) ---- #
    qa_predictions = driver_result.qa_predictions or []
    evidence_qas = [q for q in qa_predictions if q.get("evidence")]
    if evidence_qas:
        recalls = [float(q.get("evidence_recall", 0.0)) for q in evidence_qas]
        report.evidence_required_qas = len(evidence_qas)
        report.evidence_recall_mean = sum(recalls) / len(recalls)
        report.qa_with_full_evidence = sum(
            1 for q in evidence_qas if float(q.get("evidence_recall", 0.0)) >= 1.0
        )
    return report


def _shares_keywords(a: str, b: str, *, min_share: int = 1) -> bool:
    a_words = {w for w in a.lower().split() if len(w) > 3}
    b_words = {w for w in b.lower().split() if len(w) > 3}
    return len(a_words & b_words) >= min_share


def _event_text_from_context(context: str) -> str:
    """Extract the source event prompt text from an audit's serialized
    ctx_payload (see WatermarkedSampler.intercept). Falls back to the
    raw context string if parsing fails."""
    if not context:
        return ""
    try:
        payload = json.loads(context)
    except (ValueError, TypeError):
        return context
    parts = []
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)
    event_ctx = payload.get("event_context") or {}
    if isinstance(event_ctx, dict):
        for key in ("text", "speaker"):
            v = event_ctx.get(key)
            if isinstance(v, str):
                parts.append(v)
    return " ".join(parts)


def _carriers_of(audit) -> List[str]:
    """Return all carrier labels associated with an audit
    (primary ``tau`` + any ``extra_carriers``)."""
    out = [audit.tau] if audit.tau else []
    extras = getattr(audit, "extra_carriers", ()) or ()
    for c in extras:
        if c and c not in out:
            out.append(c)
    return out
