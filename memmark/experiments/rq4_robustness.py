"""RQ4: Robustness against memory-specific attacks (README §10.6).

Each attack is a deterministic post-processing on the audit trace.
After the attack we re-run R3 in-record verification (since R3 is the
hardest case for the watermark) and report:
  * pre/post bit recovery
  * **commitment_fail_rate** — fraction of leaves whose commitment hash
    no longer matches its reveal record. Catches CONTENT-level tamper
    (manual_edits, paraphrase, edge_relabel, supersession,
    subgraph_reanchor) where the leaf still exists but its bytes have
    changed.
  * **merkle_proof_fail_rate** — fraction of leaves whose Merkle proof
    no longer chains to the signed anchor root. Catches STRUCTURAL
    tamper (pruning, dedup, poisoning, compaction) where the leaf set
    itself has been resized so the rebuilt root diverges from the
    signed anchor.

These two signals are deliberately reported separately: a paper
reviewer asking "did MemMark detect this attack?" needs to know
*which* layer of the audit caught it. Some attacks trigger only one
of the two; the union is implicitly available as
``max(commitment_fail_rate, merkle_proof_fail_rate)`` per row.

The 4 main paper attacks: compaction, supersession, pruning,
manual_edits. Three appendix attacks: dedup, paraphrase rewrite,
poisoning. Two KGMark-style attacks: edge_relabel, subgraph_reanchor.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from memmark.core.types import AuditRecord
from memmark.verifier.in_record import verify_in_record
from memmark.benchmarks.locomo.driver import LoCoMoDriverResult


@dataclass
class AttackOutcome:
    name: str
    strength: float
    leaves_affected: int
    bit_recovery_pre: float
    bit_recovery_post: float
    commitment_fail_rate: float
    merkle_proof_fail_rate: float
    # Structural-tamper signal: how many leaves disappeared from the
    # audit set between pre-attack (anchor.leaf_count) and post-attack
    # (len(audits_after)). Commitment and Merkle-proof failures only
    # fire on leaves the verifier still SEES; pruning / dedup remove
    # leaves outright so they previously slipped past both signals.
    # Reported as count + rate so a paper table can pick either.
    missing_leaves_count: int = 0
    missing_leaves_rate: float = 0.0


@dataclass
class RQ4Report:
    pre_recovery: float = 0.0
    outcomes: List[AttackOutcome] = field(default_factory=list)


def run_rq4_robustness(
    *,
    driver_result: LoCoMoDriverResult,
    secret_key: str,
    main_attacks: Sequence[str] = (
        "compaction",
        "supersession",
        "pruning",
        "manual_edits",
    ),
    extra_attacks: Sequence[str] = (
        "dedup",
        "paraphrase_rewrite",
        "poisoning",
        "edge_relabel",
        "subgraph_reanchor",
    ),
    strengths: Sequence[float] = (0.1, 0.3, 0.5),
    seed: int = 0,
) -> RQ4Report:
    audits = driver_result.audits
    anchor = driver_result.anchor
    if anchor is None or not audits:
        return RQ4Report()

    pre = verify_in_record(
        audit_records=audits,
        anchor=anchor,
        payload_bits=driver_result.payload_bits,
        secret_key=secret_key,
    )
    report = RQ4Report(pre_recovery=pre.bit_recovery_rate)

    for name in list(main_attacks) + list(extra_attacks):
        attack_fn = _ATTACK_REGISTRY[name]
        for strength in strengths:
            attacked, affected = attack_fn(
                audits, strength=strength, seed=seed + hash(name) % 1000
            )
            post = verify_in_record(
                audit_records=attacked,
                anchor=anchor,
                payload_bits=driver_result.payload_bits,
                secret_key=secret_key,
            )
            n_leaves = max(len(post.leaf_results), 1)
            commit_fail = sum(
                1 for leaf in post.leaf_results if not leaf.get("commitment_valid")
            )
            proof_fail = sum(
                1 for leaf in post.leaf_results if not leaf.get("merkle_proof_valid")
            )
            # Compare expected leaf count (anchor at seal time) to what
            # the verifier was actually handed. Negative deltas (the
            # poisoning attack inserts extra leaves) clamp to 0 — that
            # case is already covered by commitment / proof signals on
            # the injected leaves.
            anchor_leaves = anchor.leaf_count or len(audits)
            missing = max(0, anchor_leaves - len(attacked))
            report.outcomes.append(
                AttackOutcome(
                    name=name,
                    strength=strength,
                    leaves_affected=affected,
                    bit_recovery_pre=pre.bit_recovery_rate,
                    bit_recovery_post=post.bit_recovery_rate,
                    commitment_fail_rate=commit_fail / n_leaves,
                    merkle_proof_fail_rate=proof_fail / n_leaves,
                    missing_leaves_count=missing,
                    missing_leaves_rate=missing / anchor_leaves if anchor_leaves else 0.0,
                )
            )
    return report


# --------------------------------------------------------------- #
# Attack implementations. Each takes a list[AuditRecord] and returns
# (attacked_list, num_affected). They mutate copies, never the original.
# --------------------------------------------------------------- #


def _attack_compaction(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        # Simulate compaction: collapse candidate set by removing one
        candidates = list(rec.candidates or [])
        if len(candidates) > 1:
            candidates.pop()
            object.__setattr__(rec, "candidates", candidates)
        out[idx] = rec
    return out, affected


def _attack_supersession(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        # Simulate supersede: rewrite selected_candidate_id to a sibling
        candidates = list(rec.candidates or [])
        if len(candidates) > 1:
            other = next(
                (c for c in candidates if c.candidate_id != rec.selected_candidate_id),
                None,
            )
            if other is not None:
                object.__setattr__(rec, "selected_candidate_id", other.candidate_id)
        out[idx] = rec
    return out, affected


def _attack_pruning(audits: List[AuditRecord], *, strength: float, seed: int):
    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = set(rng.sample(range(n), affected))
    # Keep only the unpruned leaves
    pruned = [a for i, a in enumerate(audits) if i not in indices]
    return pruned, affected


def _attack_manual_edits(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        # Edit the in-record probabilities silently
        probs = dict(rec.probabilities or {})
        if probs:
            keys = list(probs.keys())
            target = keys[0]
            probs[target] = max(0.0, probs[target] - 0.1)
            object.__setattr__(rec, "probabilities", probs)
        out[idx] = rec
    return out, affected


def _attack_dedup(audits: List[AuditRecord], *, strength: float, seed: int):
    """Real dedup: find audits whose selected candidate text matches
    another audit's selected text (i.e. the system stored two notes
    with the same content) and drop the duplicates, keeping one
    canonical leaf per content group. ``strength`` controls what
    fraction of duplicate-eligible groups we collapse (so we can sweep
    strength even when there are few exact duplicates in a real run).
    """

    def _selected_text(audit: AuditRecord) -> str:
        cands = audit.candidates or []
        sel = next(
            (c for c in cands if c.candidate_id == audit.selected_candidate_id),
            None,
        )
        if sel is None:
            return audit.commitment  # fallback
        return str(sel.payload.get("text", "")) or audit.commitment

    rng = random.Random(seed)
    # Group indices by selected-text bucket
    buckets: Dict[str, List[int]] = {}
    for i, a in enumerate(audits):
        buckets.setdefault(_selected_text(a), []).append(i)
    duplicate_groups = [idxs for idxs in buckets.values() if len(idxs) > 1]
    # If the trace has no real duplicates, synthesize a fraction of
    # near-duplicate groups by pairing random audits — this keeps the
    # strength sweep meaningful on conv traces with low natural dedup.
    if not duplicate_groups:
        n = len(audits)
        if n < 2:
            return list(audits), 0
        target_pairs = max(1, int(round(strength * n / 2)))
        indices = list(range(n))
        rng.shuffle(indices)
        duplicate_groups = [
            indices[i : i + 2]
            for i in range(0, min(2 * target_pairs, n - n % 2), 2)
        ]

    rng.shuffle(duplicate_groups)
    n_groups = len(duplicate_groups)
    n_collapse = max(1, int(round(strength * n_groups)))
    drop: set = set()
    for group in duplicate_groups[:n_collapse]:
        # Keep the first; drop the rest as "deduped"
        for victim_idx in group[1:]:
            drop.add(victim_idx)
    out = [a for i, a in enumerate(audits) if i not in drop]
    return out, len(drop)


def _attack_paraphrase_rewrite(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        # Append a marker to the context to simulate paraphrase
        object.__setattr__(rec, "context", rec.context + " [PARAPHRASE]")
        out[idx] = rec
    return out, affected


def _attack_poisoning(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    rng = random.Random(seed)
    n = len(audits)
    inject = max(1, int(round(strength * n)))
    out = list(audits)
    if not out:
        return out, 0
    # Inject duplicate (forged) audit records at random positions
    for _ in range(inject):
        idx = rng.randrange(0, len(out))
        forged = copy.deepcopy(out[idx])
        object.__setattr__(forged, "decision_id", forged.decision_id + "_forged")
        out.append(forged)
    return out, inject


def _attack_edge_relabel(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    """KGMark-style edge relabel: rewrite the SELECTED candidate's
    payload text (i.e. the actual edge label that ended up in the
    graph) so commitment recompute fails AND any decoder that reads
    the selected payload sees a tampered string. Previous version
    relabeled only ``candidates[0]`` which is a no-op when the chosen
    candidate isn't index 0 (which is the common case).
    """

    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        candidates = list(rec.candidates or [])
        target = next(
            (c for c in candidates if c.candidate_id == rec.selected_candidate_id),
            candidates[0] if candidates else None,
        )
        if target is not None and hasattr(target, "payload"):
            new_payload = dict(target.payload)
            existing = new_payload.get("text")
            new_payload["text"] = (
                f"{existing} [RELABEL]" if existing else "[RELABEL]"
            )
            object.__setattr__(target, "payload", new_payload)
        out[idx] = rec
    return out, affected


def _attack_subgraph_reanchor(
    audits: List[AuditRecord], *, strength: float, seed: int
):
    """KGMark-style subgraph reanchor: an attacker re-roots a portion
    of the KG so each affected audit's binding context (which records
    the cluster the new memory was attached to) shifts. We mutate
    ``audit.context`` (the JSON-serialized ctx_payload) and the
    ``link_target``-bearing candidate set so commitment recompute
    fails — a faithful "the attached subgraph root changed" signal.
    Previous implementation only tweaked ``round_num`` which is not
    folded into the commitment hash, so the attack was a no-op.
    """

    rng = random.Random(seed)
    n = len(audits)
    affected = max(1, int(round(strength * n)))
    indices = sorted(rng.sample(range(n), affected))
    out = list(audits)
    for idx in indices:
        rec = copy.deepcopy(out[idx])
        # 1. Mutate ctx_payload to mark the reanchor (changes commitment hash)
        new_ctx = (rec.context or "") + " [REANCHOR]"
        object.__setattr__(rec, "context", new_ctx)
        # 2. Re-root: rotate all candidate ids by one (the chosen anchor
        #    no longer corresponds to its original index in the candidate
        #    list — a structural shift on the KG side).
        candidates = list(rec.candidates or [])
        if len(candidates) >= 2:
            rotated = candidates[1:] + candidates[:1]
            object.__setattr__(rec, "candidates", rotated)
        out[idx] = rec
    return out, affected


_ATTACK_REGISTRY: Dict[str, Callable] = {
    "compaction": _attack_compaction,
    "supersession": _attack_supersession,
    "pruning": _attack_pruning,
    "manual_edits": _attack_manual_edits,
    "dedup": _attack_dedup,
    "paraphrase_rewrite": _attack_paraphrase_rewrite,
    "poisoning": _attack_poisoning,
    "edge_relabel": _attack_edge_relabel,
    "subgraph_reanchor": _attack_subgraph_reanchor,
}
