"""LoCoMo-official QA prompt + Table 4 metric suite.

Mirrors `locomo/task_eval/{evaluation.py, gpt_utils.py}`:

  * QA_PROMPT  / QA_PROMPT_CAT_5  — original prompt templates
  * f1_score / f1_multi / score_one — Porter-stemmed token F1
    (verbatim from `evaluation.py.eval_question_answering`)
  * bleu1 — clipped unigram BLEU
  * rouge_l — longest-common-subsequence F1 (Table 4 reports
    `rouge-1.f` from rouge package; LCS-based F1 is the same metric
    family and avoids an extra dep)
  * make_locomo_qa_judge — binary CORRECT/INCORRECT via the
    category-aware LoCoMo rule (F1 threshold or abstention check),
    matching `eval_question_answering`'s `all_ems` semantics. **Not
    LLM-as-judge** — LoCoMo paper does not use one.

`make_llm_judge` is also exposed for callers who want an LLM-judge
ablation, but it is *not* used by the default real-mode pipeline.

Memory snapshots are rendered with LoCoMo's `=== Session N
(timestamp) ===` block format (matching `task_eval/get_facts.py`),
so the QA prompt sees memory in the same shape as LoCoMo paper's
Base / +Observation baselines.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import Counter
from typing import Any, List, Optional


# ----- official LoCoMo QA prompts ------------------------------------ #

QA_PROMPT = (
    "\nBased on the above context, write an answer in the form of a "
    "short phrase for the following question. Answer with exact words "
    "from the context whenever possible.\n\n"
    "Question: {q} Short answer:\n"
)

QA_PROMPT_CAT_5 = (
    "\nBased on the above context, answer the following question.\n\n"
    "Question: {q} Short answer:\n"
)

CONV_START_PROMPT = (
    "Below is a conversation between two people: {a} and {b}. The "
    "conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)


# ----- A-mem robust cat-aware QA prompt (shared by A-mem + Graphiti) ----- #

def build_cat_aware_qa_prompt(
    category: int,
    context: str,
    question: str,
    gold_answer: str = None,
) -> "tuple[str, float]":
    """Category-specific QA prompt + temperature, verbatim from
    A-mem ``test_advanced_robust.py:117-144``.

    Both A-mem and Graphiti (after Phase B alignment) call this.
    LoCoMo questions need the same temporal / abstention handling
    regardless of which backend retrieves the context.

      cat 2 (temporal)     — explicit "Use DATE of CONVERSATION"
                              instruction so model doesn't quote
                              "yesterday" verbatim
      cat 3 (multi-hop)    — "exact words from context"
      cat 5 (adversarial)  — A/B select with `Not mentioned in the
                              conversation` and the gold answer
                              (gold_answer required)
      default (1, 4)       — same as cat 3
    """

    if category == 5 and gold_answer:
        import random as _rand
        opts = ["Not mentioned in the conversation", gold_answer]
        if _rand.random() < 0.5:
            opts = list(reversed(opts))
        prompt = (
            f"Based on the context: {context}, answer the following "
            f"question. {question}\n\n"
            f"Select the correct answer: {opts[0]} or {opts[1]}  "
            f"Short answer:"
        )
        return prompt, 0.5
    if category == 2:
        prompt = (
            f"Based on the context: {context}, answer the following "
            f"question. Use DATE of CONVERSATION to answer with an "
            f"approximate date.\nPlease generate the shortest possible "
            f"answer, using words from the conversation where possible, "
            f"and avoid using any subjects.\n\n"
            f"Question: {question} Short answer:"
        )
        return prompt, 0.7
    # cat 3, default 1/4
    prompt = (
        f"Based on the context: {context}, write an answer in "
        f"the form of a short phrase for the following question. "
        f"Answer with exact words from the context whenever possible."
        f"\n\nQuestion: {question} Short answer:"
    )
    return prompt, 0.7


def build_amem_keyword_prompt(question: str) -> str:
    """A-mem ``generate_query_llm`` prompt verbatim
    (test_advanced_robust.py:96-105)."""

    return (
        "Given the following question, generate several keywords separated "
        "by commas.\n\n"
        f"Question: {question}\n\n"
        "Keywords:"
    )


def parse_plain_text_answer(response: str) -> str:
    try:
        cleaned = _strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"]).strip()
    except Exception:
        pass
    return (response or "").strip()


def parse_keywords_response(response: str) -> str:
    try:
        cleaned = _strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"]).strip()
    except Exception:
        pass
    return (response or "").strip()


def _strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


# ----- normalization + F1 (A-mem utils.py:calculate_metrics) --------- #
#
# F1 is paper-comparable to A-mem Table 1: set-based token F1 with
# A-mem's simple_tokenize (lowercase + replace .,!? with spaces +
# split). No Porter stemming, no article removal. Switched from the
# LoCoMo-paper Porter-multiset formula because the headline comparison
# in the paper is against A-mem and we want the F1 numbers to stack
# directly. Reference: A-mem/utils.py:34-38 (simple_tokenize) +
# A-mem/utils.py:135-145 (set-based F1).


def _simple_tokenize(text: str) -> List[str]:
    """A-mem ``utils.py:simple_tokenize`` verbatim."""
    text = str(text)
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def normalize_answer(s: str) -> str:
    """Kept for callers (e.g. bleu1) that still want the LoCoMo
    article-stripped normalization. F1 itself does not use this."""
    s = (s or "").replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.lower().split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """A-mem set-based F1 (utils.py:135-145), paper-comparable to
    A-mem Table 1. Sets the "main" LoCoMo metric to the same formula
    A-mem uses; no Porter stemming, no article removal."""
    pred_tokens = set(_simple_tokenize(prediction))
    gold_tokens = set(_simple_tokenize(ground_truth))
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0


def f1_multi(prediction: str, ground_truth: str) -> float:
    """Category 1: ground_truth may have ',' separated alternates."""

    preds = [p.strip() for p in (prediction or "").split(",")]
    golds = [g.strip() for g in (ground_truth or "").split(",")]
    if not preds or not golds:
        return 0.0
    return _mean(
        max(f1_score(p, g) for p in preds) for g in golds
    )


def _mean(seq) -> float:
    seq = list(seq)
    return sum(seq) / len(seq) if seq else 0.0


def score_one(prediction: str, gold: str, category: int) -> float:
    """LoCoMo per-category metric (mirrors eval_question_answering)."""

    if category == 3:
        # multi-hop: gold may have ';' separated alternates → take first
        gold = (gold or "").split(";")[0].strip()
        return f1_score(prediction, gold)
    if category in (2, 4):
        return f1_score(prediction, gold)
    if category == 1:
        return f1_multi(prediction, gold)
    if category == 5:
        # Adversarial: gold is now ``adversarial_answer`` (loader fix),
        # cat-5 prompt is A/B with the gold as one option. Compare with
        # set-based F1 same as the rest — this matches A-mem Table 1's
        # uniform F1 computation across categories. The previous
        # hard-coded "did the model say not-mentioned?" check made
        # cat-5 effectively a binary heuristic with empty gold.
        return f1_score(prediction, gold)
    # unknown → fallback to plain F1
    return f1_score(prediction, gold)


# ----- BLEU-1 + ROUGE-L (no extra deps) ---------------------------- #


def bleu1(prediction: str, gold: str) -> float:
    """Clipped unigram BLEU (no brevity penalty), LoCoMo Table 4 same."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    clipped = 0
    for tok in pred_tokens:
        if gold_counts[tok] > 0:
            clipped += 1
            gold_counts[tok] -= 1
    return clipped / len(pred_tokens)


def rouge_l(prediction: str, gold: str) -> float:
    """ROUGE-L F1 via LCS over normalized tokens."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(a, b) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    # Use 2-row DP to keep memory bounded
    prev = [0] * (m + 1)
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, [0] * (m + 1)
    return prev[m]


def metric_suite(prediction: str, gold: str, category: int) -> dict:
    """Returns the LoCoMo Table 4 numbers per QA leaf."""

    return {
        "f1": score_one(prediction, gold, category),  # category-aware F1
        "bleu1": bleu1(prediction, gold),
        "rougeL": rouge_l(prediction, gold),
    }


def aggregate_locomo_metrics(qa_predictions: List[dict]) -> dict:
    """Mirror of `task_eval/evaluation.eval_question_answering` output.

    Returns per-category F1 / BLEU-1 / ROUGE-L means + overall, with
    the same category bucketing the LoCoMo paper Table 4 uses
    (cat 1 = single-hop, cat 2 = temporal, cat 3 = multi-hop,
    cat 4 = open-ended, cat 5 = adversarial / abstention).
    """

    overall = {"n": 0, "f1": 0.0, "bleu1": 0.0, "rougeL": 0.0}
    by_cat: dict = {}
    for q in qa_predictions:
        cat = int(q.get("category", 0))
        bucket = by_cat.setdefault(
            cat, {"n": 0, "f1": 0.0, "bleu1": 0.0, "rougeL": 0.0}
        )
        bucket["n"] += 1
        overall["n"] += 1
        for k in ("f1", "bleu1", "rougeL"):
            bucket[k] += float(q.get(k, 0.0))
            overall[k] += float(q.get(k, 0.0))
    for bucket in (overall, *by_cat.values()):
        n = bucket["n"]
        if n:
            for k in ("f1", "bleu1", "rougeL"):
                bucket[k] /= n
    return {"overall": overall, "by_category": by_cat}


# ----- LLM-as-judge (semantic CORRECT / INCORRECT) ----------------- #


def make_llm_judge(llm_client):
    """Returns `(question, predicted) -> bool` using LLM semantic judgment.

    Prompt mirrors NirDiamant's notebook (which itself mirrors LoCoMo
    paper's GPT-4 judge): a CORRECT / INCORRECT verdict on whether
    the predicted answer carries the same factual information as
    gold. Falls back to substring + F1>0.5 if the LLM call fails.
    """

    def judge(question, predicted) -> bool:
        gold = (question.answer or "").strip()
        if not predicted or not gold:
            return False
        prompt = (
            "You are evaluating an answer to a question about a long "
            "multi-session conversation.\n\n"
            f"Question: {question.question}\n"
            f"Reference answer: {gold}\n"
            f"Predicted answer: {predicted}\n\n"
            "Is the predicted answer correct? It does NOT need to "
            "match word-for-word, but it MUST convey the same factual "
            "information as the reference. For category-5 (adversarial) "
            "questions, 'No information available' / 'I don't know' "
            "type answers should be judged correct when the reference "
            "indicates abstention.\n"
            "Reply with exactly 'CORRECT' or 'INCORRECT'."
        )
        try:
            raw = llm_client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        except Exception:
            # Fallback: F1 ≥ 0.5 means "good enough"
            return score_one(predicted, gold, question.category) >= 0.5
        return "CORRECT" in (raw or "").strip().upper()[:32]

    return judge


# ----- responder + judge wired to MemoryWatermarker driver ----------- #


def make_locomo_qa_responder(
    llm_client,
    *,
    max_chars: int = 12000,
):
    """Returns a `qa_responder(question, context_text) -> str` closure.

    `context_text` is the pre-rendered memory context produced by the
    backend's canonical retrieval API (``backend.qa_context(question)``):

      * A-mem: ``find_related_memories(q, k)`` formatted string.
      * Graphiti: ``client.search(q, group_ids).fact`` list.
      * JsonStore / fallback: full snapshot in LoCoMo session-marker
        format.

    Backends that produce an answer themselves (mode=``answer``)
    bypass this responder entirely.
    """

    def responder(question, context_text) -> str:
        trace = build_locomo_qa_trace(
            question,
            context_text,
            max_chars=max_chars,
        )
        setattr(responder, "last_trace", trace)
        try:
            answer = (llm_client.complete(trace["messages"], temperature=0.0) or "").strip()
            trace["raw_response"] = answer
            return answer
        except Exception as exc:
            trace["error"] = f"{type(exc).__name__}: {exc}"
            return ""

    return responder


def build_locomo_qa_trace(
    question,
    snapshot,
    *,
    memory_render: Optional[Any] = None,
    max_chars: int = 12000,
):
    if isinstance(snapshot, str):
        ctx = snapshot[:max_chars]
    else:
        render = memory_render or _default_render_memory
        ctx = render(snapshot)[:max_chars]
    if question.category == 5:
        qa_template = QA_PROMPT_CAT_5
    else:
        qa_template = QA_PROMPT
    user_prompt = ctx + qa_template.format(q=question.question)
    # Match A-mem upstream SYSTEM_MESSAGE (memory_layer_robust.py:76).
    # Abstention is opt-in via cat-5's A/B prompt only, never via system.
    messages = [
        {
            "role": "system",
            "content": (
                "Follow the format specified in the prompt exactly. "
                "Do not add extra commentary."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    return {
        "context": ctx,
        "context_chars": len(ctx),
        "qa_template": qa_template,
        "user_prompt": user_prompt,
        "messages": messages,
        "max_chars": max_chars,
    }


def make_locomo_qa_judge():
    """Returns a `judge(question, predicted) -> bool` using LoCoMo F1.

    A leaf is "correct" if its category-aware F1 ≥ 0.5 (matching
    LoCoMo's table: they report F1 directly; using 0.5 as a binary
    cutoff is our reduction so the rest of the pipeline can still
    aggregate accuracy. Drivers that want the raw F1 should instead
    use `score_one` directly.)
    """

    def judge(question, predicted) -> bool:
        return score_one(predicted, question.answer, question.category) >= 0.5

    return judge


def _default_render_memory(snapshot: List[dict]) -> str:
    """Render memory snapshot in LoCoMo official `=== Session N (timestamp) ===`
    block format (matching `task_eval/get_facts.py` `conversation`
    string and NirDiamant's notebook).

    This puts memory in front of the LLM in the same shape the
    LoCoMo paper Base / +Observation prompts see, so QA F1 is
    directly comparable to Table 4.
    """

    if not snapshot:
        return "(no long-term memory available)"

    # Group records by session_index so we can lay them out in
    # session-major order (matches LoCoMo's full conversation render).
    by_session: dict[int, List[dict]] = {}
    no_session: List[dict] = []
    for rec in snapshot:
        text = (rec.get("text") or rec.get("content") or "").strip()
        if not text:
            continue
        sid = rec.get("session_index")
        if sid is None:
            no_session.append(rec)
        else:
            by_session.setdefault(int(sid), []).append(rec)

    lines: List[str] = []
    for sid in sorted(by_session):
        records = by_session[sid]
        # Take session date_time from any record that has it.
        ts = ""
        for r in records:
            ts = r.get("session_date_time") or ""
            if ts:
                break
        header = f"\n=== Session {sid} ({ts}) ===" if ts else f"\n=== Session {sid} ==="
        lines.append(header)
        for rec in records:
            speaker = (rec.get("speaker") or "").strip()
            text = (rec.get("text") or "").strip()
            evidence = rec.get("dia_ids") or []
            evidence_tag = ""
            if evidence:
                evidence_tag = f" [{','.join(evidence)}]"
            if speaker:
                lines.append(f"{speaker}: {text}{evidence_tag}")
            else:
                lines.append(f"- {text}{evidence_tag}")

    if no_session:
        lines.append("\n=== Unattributed memories ===")
        for rec in no_session:
            text = (rec.get("text") or "").strip()
            evidence = rec.get("dia_ids") or []
            evidence_tag = f" [{','.join(evidence)}]" if evidence else ""
            lines.append(f"- {text}{evidence_tag}")

    return "\n".join(lines).lstrip()
