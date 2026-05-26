"""LoCoMo driver — replays a conversation through MemMark.

Each backend has its own preferred ingestion granularity (matching
its upstream LoCoMo / LongMemEval official protocol):

  - "turn"    : per-turn episode with session date_time
                Graphiti  (graphiti/tests/evals/eval_e2e_graph_building.py)
                A-MEM     (A-mem/test_advanced_robust.py)
                JsonStore (smoke default)
  - "fact"    : LoCoMo-official `CONVERSATION2FACTS_PROMPT` per
                session → N facts each with dia_id evidence
                (kept for Mem0-style ablations only)

  ("session" mode is still implemented for completeness.)

The driver reads `backend.preferred_ingestion_mode` and dispatches
automatically, so each backend gets the input format its upstream
benchmarks use.

After the conversation ends, run all `qa` questions through the
LoCoMo-official QA prompt + F1 judge against the final snapshot.
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from memmark.benchmarks.locomo.loader import (
    LoCoMoConversation,
    LoCoMoQuestion,
    LoCoMoSession,
    LoCoMoTurn,
)
from memmark.benchmarks.locomo.qa_eval import (
    bleu1,
    build_locomo_qa_trace,
    make_locomo_qa_judge,
    make_locomo_qa_responder,
    rouge_l,
    score_one,
)
from memmark.core.types import AuditRecord, DecisionPoint, SessionHeader
from memmark.sdk.memory_watermarker import EvolveResult, MemoryWatermarker


TurnFilter = Callable[[LoCoMoTurn, str], bool]
QAJudge = Callable[[LoCoMoQuestion, str], bool]
# qa_responder takes the question + a pre-rendered memory context
# string (produced by `backend.qa_context(question)`).
QAResponder = Callable[[LoCoMoQuestion, str], str]


@dataclass
class LoCoMoDriverResult:
    sample_id: str
    decisions: List[DecisionPoint] = field(default_factory=list)
    audits: List[AuditRecord] = field(default_factory=list)
    anchor: Optional[SessionHeader] = None
    memory_snapshot_final: List[Dict[str, Any]] = field(default_factory=list)
    qa_predictions: List[Dict[str, Any]] = field(default_factory=list)
    capacity_stats: Dict[str, Any] = field(default_factory=dict)
    extracted_events: List[Dict[str, Any]] = field(default_factory=list)
    payload_bits: str = ""

    @property
    def bits_embedded_total(self) -> int:
        return sum(a.bits_embedded for a in self.audits)

    @property
    def qa_accuracy(self) -> float:
        if not self.qa_predictions:
            return 0.0
        correct = sum(1 for q in self.qa_predictions if q.get("correct"))
        return correct / len(self.qa_predictions)

    @property
    def qa_f1_mean(self) -> float:
        return self._mean_metric("f1")

    @property
    def qa_bleu1_mean(self) -> float:
        return self._mean_metric("bleu1")

    @property
    def qa_rougeL_mean(self) -> float:
        return self._mean_metric("rougeL")

    @property
    def qa_judge_accuracy(self) -> float:
        return self._mean_metric("judge_correct", boolean=True)

    @property
    def qa_metrics_by_category(self) -> Dict[int, Dict[str, float]]:
        """Per-category {f1, bleu1, rougeL, judge_acc, n} (LoCoMo Table 4 shape)."""

        out: Dict[int, Dict[str, float]] = {}
        groups: Dict[int, List[Dict[str, Any]]] = {}
        for q in self.qa_predictions:
            groups.setdefault(int(q.get("category", 0)), []).append(q)
        for cat, items in groups.items():
            n = len(items)
            out[cat] = {
                "n": float(n),
                "f1": sum(q.get("f1", 0.0) for q in items) / n,
                "bleu1": sum(q.get("bleu1", 0.0) for q in items) / n,
                "rougeL": sum(q.get("rougeL", 0.0) for q in items) / n,
                "judge_acc": sum(
                    1 for q in items if q.get("judge_correct")
                ) / n,
            }
        return out

    def _mean_metric(self, key: str, *, boolean: bool = False) -> float:
        if not self.qa_predictions:
            return 0.0
        if boolean:
            return sum(1 for q in self.qa_predictions if q.get(key)) / len(self.qa_predictions)
        return sum(q.get(key, 0.0) for q in self.qa_predictions) / len(self.qa_predictions)


def _progress_bar(current: int, total: int, width: int = 18) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    current = max(0, min(current, total))
    filled = int(width * current / total)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"


class _ProgressPrinter:
    """进度打印器，支持 tqdm 进度条（可选）和原始文本输出"""
    
    def __init__(self, enabled: bool, baseline: str = "", use_tqdm: Optional[bool] = None) -> None:
        self.enabled = enabled
        if use_tqdm is None:
            use_tqdm = os.getenv("MEMMARK_PROGRESS_TQDM", "0").lower() in {"1", "true", "yes"}
        self.use_tqdm = use_tqdm and HAS_TQDM and enabled
        self._last_len = 0
        self._line_open = False
        self.baseline = baseline
        self.session_pbar = None
        self.qa_pbar = None

    def init_session_bar(self, total: int) -> None:
        if not self.use_tqdm:
            return
        self.session_pbar = tqdm(
            total=total,
            desc=f"Sessions ({self.baseline})",
            position=0,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, qid={postfix}]'
        )
    
    def update_session(self, n: int = 1, **kwargs) -> None:
        if self.session_pbar:
            self.session_pbar.update(n)
            if kwargs:
                self.session_pbar.set_postfix(**kwargs)
            self.session_pbar.refresh()
        elif self.enabled and kwargs:
            fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[progress] baseline={self.baseline} phase=session {fields}", flush=True)
    
    def close_session_bar(self) -> None:
        if self.session_pbar:
            self.session_pbar.close()
            self.session_pbar = None
    
    def init_qa_bar(self, total: int) -> None:
        if not self.use_tqdm:
            return
        self.qa_pbar = tqdm(
            total=total,
            desc=f"Questions ({self.baseline})",
            position=1,
            leave=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, qid={postfix}]'
        )
    
    def update_qa(self, n: int = 1, **kwargs) -> None:
        if self.qa_pbar:
            self.qa_pbar.update(n)
            if kwargs:
                self.qa_pbar.set_postfix(**kwargs)
            self.qa_pbar.refresh()
        elif self.enabled and kwargs:
            fields = " ".join(f"{k}={v}" for k, v in kwargs.items())
            print(f"[progress] baseline={self.baseline} phase=qa {fields}", flush=True)
    
    def close_qa_bar(self) -> None:
        if self.qa_pbar:
            self.qa_pbar.close()
            self.qa_pbar = None

    def update(self, message: str) -> None:
        """原始的行内更新方法（用于非tqdm模式）"""
        if not self.enabled:
            return
        if self.use_tqdm:
            return  # tqdm模式下不使用此方法
        print(message, flush=True)

    def line(self, message: str = "") -> None:
        """输出一行消息"""
        if not self.enabled:
            return
        if self.use_tqdm and (self.session_pbar or self.qa_pbar):
            tqdm.write(message)
        elif self._line_open:
            pad = " " * max(0, self._last_len - len(message))
            sys.stdout.write("\r" + message + pad + "\n")
            sys.stdout.flush()
            self._last_len = 0
            self._line_open = False
        else:
            print(message, flush=True)


class LoCoMoDriver:
    """Replay a single LoCoMo conversation through a MemoryWatermarker.

    `turn_filter` lets you drop chitchat / greetings; default keeps
    every turn (so the backend sees the full dialog and does its own
    native ingestion / extraction).
    """

    def __init__(
        self,
        *,
        watermarker: MemoryWatermarker,
        turn_filter: Optional[TurnFilter] = None,
        qa_responder: Optional[QAResponder] = None,
        qa_judge: Optional[QAJudge] = None,
        max_sessions: Optional[int] = None,
        max_qa: Optional[int] = None,
        max_turns_per_session: Optional[int] = None,
        fact_extractor_llm: Optional[Any] = None,
        async_assess: bool = False,
        async_max_concurrency: int = 4,
        progress: bool = False,
        progress_context: Optional[Dict[str, Any]] = None,
        # Backwards-compat: accept the deprecated `memory_extractor`
        # kwarg but ignore it.
        memory_extractor: Optional[Any] = None,
    ) -> None:
        self.wm = watermarker
        self.turn_filter = turn_filter or _keep_all_substantive_turns
        self.qa_responder = qa_responder or _default_qa_responder
        self.qa_judge = qa_judge or _default_qa_judge
        self.max_sessions = max_sessions
        self.max_qa = max_qa
        self.max_turns_per_session = max_turns_per_session
        self.fact_extractor_llm = fact_extractor_llm
        self.memory_candidate_llm = fact_extractor_llm
        self.async_assess = bool(async_assess)
        self.async_max_concurrency = max(1, int(async_max_concurrency))
        self.progress = progress
        self.progress_context = progress_context or {}
        baseline = self.progress_context.get("baseline", "")
        self._progress = _ProgressPrinter(self.progress, baseline=baseline)
        self._active_session_i = 0
        self._active_session_count = 0
        self._active_conversation = self.progress_context.get("conversation", "")
        self._active_baseline = baseline
        # We intentionally ignore memory_extractor — backend is the
        # source of truth for what becomes a memory record.
        self._legacy_extractor_warned = bool(memory_extractor)

    def run(self, conversation: LoCoMoConversation) -> LoCoMoDriverResult:
        result = LoCoMoDriverResult(
            sample_id=conversation.sample_id,
            payload_bits=self.wm.payload_bits,
        )
        sessions = conversation.sessions
        if self.max_sessions is not None:
            sessions = sessions[: self.max_sessions]

        ingestion_mode = getattr(self.wm.backend, "preferred_ingestion_mode", "turn")
        recent_dialog_ids: List[str] = []
        qa_list = conversation.qa
        if self.max_qa is not None:
            qa_list = qa_list[: self.max_qa]
        total_turns = 0
        for session in sessions:
            turns = session.turns
            if self.max_turns_per_session is not None:
                turns = turns[: self.max_turns_per_session]
            total_turns += len(turns)
        self._progress.line(
            f"[start] baseline={self._active_baseline} conversation={self._active_conversation} "
            f"sample={conversation.sample_id} sessions={len(sessions)} turns={total_turns} "
            f"qa={len(qa_list)} ingestion={ingestion_mode}"
        )
        
        if hasattr(self.wm.backend, "begin_conversation"):
            self.wm.backend.begin_conversation(conversation.sample_id)

        # 初始化 Sessions 进度条
        self._progress.init_session_bar(len(sessions))
        
        for session_i, session in enumerate(sessions, start=1):
            self._active_session_i = session_i
            self._active_session_count = len(sessions)
            turns = session.turns
            if self.max_turns_per_session is not None:
                turns = turns[: self.max_turns_per_session]
            events_before = len(result.extracted_events)
            audits_before = len(self.wm.audits)
            self._progress.line(
                f"[session] conversation={self._active_conversation} "
                f"{_progress_bar(session_i, len(sessions), 12)} "
                f"session_id={session.index} turns={len(turns)}"
            )
            if ingestion_mode == "session":
                self._ingest_session_mode(
                    conversation, session, turns, result, recent_dialog_ids
                )
            elif ingestion_mode == "fact":
                self._ingest_fact_mode(
                    conversation, session, turns, result, recent_dialog_ids
                )
            else:
                self._ingest_turn_mode(
                    conversation, session, turns, result, recent_dialog_ids
                )
            # 更新 Sessions 进度条
            self._progress.update_session(
                1, 
                qid=conversation.sample_id,
                turns=len(turns),
                events=len(result.extracted_events),
                bits=self.wm.bit_index
            )

        # 关闭 Sessions 进度条
        self._progress.close_session_bar()
        
        self._progress.line("[seal] finalizing watermark anchor")
        result.anchor = self.wm.seal_session()
        # Snapshot the canonical audit set AFTER seal so it matches
        # anchor.leaf_count exactly. Driver-side incremental extend was
        # missing partial audits when wm.backend.apply() raised mid-call
        # (see note in _evolve_one). Single source of truth = sampler.
        result.audits = list(self.wm.audits)
        result.decisions = [
            DecisionPoint(
                decision_id=a.decision_id,
                tau=a.tau,
                candidates=a.candidates,
                probabilities=a.probabilities,
                context=a.context,
                round_num=a.round_num,
                nonce=a.nonce,
                watermark_version=a.watermark_version,
            )
            for a in result.audits
            if a.candidates and a.probabilities
        ]
        self._progress.line("[snapshot] reading final memory state")
        result.memory_snapshot_final = self.wm.backend.snapshot()
        result.capacity_stats = _capacity_stats(result.audits, result.decisions)

        self._progress.line(
            f"[qa] questions={len(qa_list)} memory_records={len(result.memory_snapshot_final)}"
        )
        
        # 初始化 Questions 进度条
        self._progress.init_qa_bar(len(qa_list))
        
        if self.async_assess and len(qa_list) > 1:
            try:
                result.qa_predictions.extend(self._run_qa_parallel(qa_list, result))
            except Exception as exc:
                self._progress.line(f"[qa fallback] {type(exc).__name__}: {exc}")
                for qa_i, q in enumerate(qa_list, start=1):
                    result.qa_predictions.append(
                        self._run_one_qa(q, qa_i, len(qa_list), result)
                    )
                    self._progress.update_qa(1, qid=qa_i, f1=result.qa_f1_mean)
        else:
            for qa_i, q in enumerate(qa_list, start=1):
                result.qa_predictions.append(
                    self._run_one_qa(q, qa_i, len(qa_list), result)
                )
                self._progress.update_qa(1, qid=qa_i, f1=result.qa_f1_mean)
        
        # 关闭 Questions 进度条
        self._progress.close_qa_bar()
        
        self._progress.line(
            f"[done] f1={result.qa_f1_mean:.3f} bleu1={result.qa_bleu1_mean:.3f} "
            f"rougeL={result.qa_rougeL_mean:.3f} judge_acc={result.qa_judge_accuracy:.3f}"
        )
        return result

    def _run_qa_parallel(
        self,
        qa_list: List[LoCoMoQuestion],
        result: LoCoMoDriverResult,
    ) -> List[Dict[str, Any]]:
        workers = min(self.async_max_concurrency, len(qa_list))
        self._progress.line(f"[qa async] workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_one_qa, q, qa_i, len(qa_list), result): (qa_i, q)
                for qa_i, q in enumerate(qa_list, start=1)
            }
            out = []
            completed = 0
            for future in as_completed(futures):
                qa_i, q = futures[future]
                try:
                    out.append(future.result())
                except Exception as exc:
                    out.append(self._qa_error_record(qa_i, q, result, exc))
                completed += 1
                self._progress.update_qa(1, qid=qa_i)
            return sorted(out, key=lambda item: int(item.get("index", 0)))

    def _run_one_qa(
        self,
        q: LoCoMoQuestion,
        qa_i: int,
        qa_count: int,
        result: LoCoMoDriverResult,
    ) -> Dict[str, Any]:
        try:
            ctx = self.wm.backend.qa_context(
                q.question,
                k=10,
                category=q.category,
                gold_answer=q.answer,
                llm_client=self.fact_extractor_llm,
            )
            if ctx.get("mode") == "answer":
                answer = (ctx.get("text") or "").strip()
                context_text = ctx.get("context") or ctx.get("retrieved_context") or ""
                qa_trace = {
                    "context": context_text,
                    "context_chars": len(context_text),
                    "mode": "answer",
                    "keywords": ctx.get("keywords", ""),
                    "keyword_raw": ctx.get("keyword_raw", ""),
                    "retrieval_error": ctx.get("retrieval_error", ""),
                    "retrieval_fallback": bool(ctx.get("retrieval_fallback")),
                    "user_prompt": ctx.get("user_prompt", ""),
                    "raw_response": answer,
                }
            else:
                context_text = ctx.get("text") or ""
                answer = self.qa_responder(q, context_text)
                qa_trace = build_locomo_qa_trace(q, context_text)
                qa_trace["raw_response"] = answer
        except Exception as exc:
            return self._qa_error_record(qa_i, q, result, exc)

        f1 = score_one(answer, q.answer, q.category)
        bleu = bleu1(answer, q.answer)
        rouge = rouge_l(answer, q.answer)
        correct = bool(self.qa_judge(q, answer))
        # QA-time evidence recall: did the rendered context the LLM
        # actually saw contain the question's evidence dia_ids? Was
        # previously a snapshot-wide check that returned the same
        # number for every baseline regardless of QA quality.
        evidence_recall = _evidence_recall(q, context_text)
        return {
            "index": qa_i,
            "question": q.question,
            "answer_gold": q.answer,
            "answer_pred": answer,
            "category": q.category,
            "evidence": q.evidence,
            "f1": f1,
            "bleu1": bleu,
            "rougeL": rouge,
            "judge_correct": correct,
            "correct": correct,
            "evidence_recall": evidence_recall,
            "memory_record_count": len(result.memory_snapshot_final),
            "qa_trace": qa_trace,
        }

    def _qa_error_record(
        self,
        qa_i: int,
        q: LoCoMoQuestion,
        result: LoCoMoDriverResult,
        exc: Exception,
    ) -> Dict[str, Any]:
        qa_trace = {
            "context": "",
            "context_chars": 0,
            "mode": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        return {
            "index": qa_i,
            "question": q.question,
            "answer_gold": q.answer,
            "answer_pred": "",
            "category": q.category,
            "evidence": q.evidence,
            "f1": 0.0,
            "bleu1": 0.0,
            "rougeL": 0.0,
            "judge_correct": False,
            "correct": False,
            "evidence_recall": _evidence_recall(q, ""),
            "memory_record_count": len(result.memory_snapshot_final),
            "qa_trace": qa_trace,
        }

    def _ingest_turn_mode(
        self,
        conversation: LoCoMoConversation,
        session: LoCoMoSession,
        turns: List[LoCoMoTurn],
        result: "LoCoMoDriverResult",
        recent_dialog_ids: List[str],
    ) -> None:
        """Per-turn ingestion (Graphiti / JsonStore default).

        Each LoCoMo turn becomes one memory event; the operation
        carries the session's date_time so Graphiti uses it as
        `reference_time` (matching `eval_e2e_graph_building.py`).
        """

        total_turns = len(turns)
        for turn_i, turn in enumerate(turns, start=1):
            if not self.turn_filter(turn, session.summary):
                self._progress.update(
                    f"[turn skip] conversation={self._active_conversation} "
                    f"session={self._active_session_i}/{self._active_session_count} "
                    f"{_progress_bar(turn_i, total_turns)} dia={turn.dia_id}"
                )
                continue
            recent_dialog_ids[:] = (recent_dialog_ids + [turn.dia_id])[-8:]
            event_text = _format_turn(turn, session.date_time)
            self._progress.update(
                f"[turn] conversation={self._active_conversation} "
                f"session={self._active_session_i}/{self._active_session_count} "
                f"{_progress_bar(turn_i, total_turns)} dia={turn.dia_id} "
                f"events={len(result.extracted_events)} llm={len(self.wm.audits)} "
                f"bits={self.wm.bit_index}"
            )
            self._progress.update_session(
                0,
                qid=turn.dia_id,
                turn=f"{turn_i}/{total_turns}",
                events=len(result.extracted_events),
                bits=self.wm.bit_index,
            )
            self._evolve_one(
                event_text,
                dia_ids=[turn.dia_id],
                session_index=session.index,
                session_date_time=session.date_time,
                speaker=turn.speaker,
                recent_dialog_ids=recent_dialog_ids,
                result=result,
                source_label=f"{turn.dia_id}",
                progress_label=f"turn:{session.index}:{turn.dia_id}",
            )
            self._progress.update(
                f"[turn] conversation={self._active_conversation} "
                f"session={self._active_session_i}/{self._active_session_count} "
                f"{_progress_bar(turn_i, total_turns)} dia={turn.dia_id} "
                f"events={len(result.extracted_events)} llm={len(self.wm.audits)} "
                f"bits={self.wm.bit_index}"
            )

    def _target_llm_watermark_text(self, event_text: str, *, source_label: str) -> str:
        if not getattr(self.wm.backend, "watermark_with_target_llm", False):
            return event_text
        if self.wm.sampler_mode in {"no_watermark", "kgmark_graphiti"}:
            return event_text
        if self.memory_candidate_llm is None:
            return event_text
        prompt = (
            "You are the memory-writing agent. Generate semantically equivalent memory "
            "records for the same dialogue turn. Preserve the speaker, dialogue id, "
            "date/time, entities, and factual meaning. Return JSON only in this form: "
            '{"candidates":[{"decision":"<full memory record text>","weight":0.6},'
            '{"decision":"<alternative full memory record text>","weight":0.4}],'
            '"carrier":"semantic_realization"}.\n\n'
            f"Original memory record:\n{event_text}"
        )
        try:
            raw = self.memory_candidate_llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=900,
            )
        except Exception:
            return event_text
        try:
            from memmark.llm.watermarked import parse_agentmark_response

            decisions_raw, weights, carriers = parse_agentmark_response(raw)
        except Exception:
            return event_text
        decisions: List[str] = []
        cleaned_weights: List[float] = []
        for decision, weight in zip(decisions_raw, weights):
            text = self._candidate_text(decision)
            if not text:
                continue
            decisions.append(text)
            cleaned_weights.append(weight)
        if event_text not in decisions:
            decisions.insert(0, event_text)
            cleaned_weights.insert(0, max(cleaned_weights) if cleaned_weights else 1.0)
        if len(decisions) < 2:
            return event_text
        chosen = self.wm.sampler.intercept(
            decisions,
            cleaned_weights,
            ctx_text=f"{source_label}\n{event_text}",
            prompt_name="target_memory_writer",
            tau=carriers or ["semantic_realization"],
        )
        return chosen or event_text

    @staticmethod
    def _candidate_text(decision: Any) -> str:
        if isinstance(decision, str):
            return decision.strip()
        if isinstance(decision, dict):
            for key in ("text", "memory", "content", "record"):
                value = decision.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return json.dumps(decision, ensure_ascii=False, sort_keys=True)
        return str(decision).strip() if decision is not None else ""

    def _ingest_session_mode(
        self,
        conversation: LoCoMoConversation,
        session: LoCoMoSession,
        turns: List[LoCoMoTurn],
        result: "LoCoMoDriverResult",
        recent_dialog_ids: List[str],
    ) -> None:
        """Document-level ingestion (one filtered session = one doc).

        The whole filtered session is concatenated and pushed as one
        document; cognify() / its native pipeline does the entity /
        relation extraction internally.
        """

        kept = [t for t in turns if self.turn_filter(t, session.summary)]
        if not kept:
            self._progress.line(f"[session skip] session_id={session.index} no kept turns")
            return
        body = _format_session_text(session, kept)
        all_dia_ids = [t.dia_id for t in kept]
        self._progress.update(
            f"[session ingest] conversation={self._active_conversation} "
            f"session={self._active_session_i}/{self._active_session_count} kept_turns={len(kept)}"
        )
        recent_dialog_ids[:] = (recent_dialog_ids + all_dia_ids)[-8:]
        self._evolve_one(
            body,
            dia_ids=all_dia_ids,
            session_index=session.index,
            session_date_time=session.date_time,
            speaker="",
            recent_dialog_ids=recent_dialog_ids,
            result=result,
            source_label=f"session_{session.index}",
            progress_label=f"session:{session.index}",
        )

    def _ingest_fact_mode(
        self,
        conversation: LoCoMoConversation,
        session: LoCoMoSession,
        turns: List[LoCoMoTurn],
        result: "LoCoMoDriverResult",
        recent_dialog_ids: List[str],
    ) -> None:
        """LoCoMo / Mem0 / A-MEM style fact extraction.

        Runs `CONVERSATION2FACTS_PROMPT` (LoCoMo official) on the
        session, producing N facts each with their dia_id evidence,
        then pushes each fact as a memory event.
        """

        kept = [t for t in turns if self.turn_filter(t, session.summary)]
        if not kept or self.fact_extractor_llm is None:
            reason = "no_kept_turns" if not kept else "no_fact_extractor_llm"
            self._progress.line(f"[extract fallback] session_id={session.index} reason={reason}")
            self._ingest_turn_mode(
                conversation, session, kept, result, recent_dialog_ids
            )
            return

        from memmark.extractors import extract_session_facts

        self._progress.update(
            f"[extract] conversation={self._active_conversation} "
            f"session={self._active_session_i}/{self._active_session_count} turns={len(kept)}"
        )
        facts = extract_session_facts(
            llm_client=self.fact_extractor_llm,
            speaker_a=conversation.speaker_a,
            speaker_b=conversation.speaker_b,
            session_index=session.index,
            session_date_time=session.date_time,
            turns=kept,
        )

        if not facts:
            self._progress.line(f"[extract fallback] session_id={session.index} reason=no_facts")
            self._ingest_turn_mode(
                conversation, session, kept, result, recent_dialog_ids
            )
            return

        self._progress.line(f"[extract done] session_id={session.index} facts={len(facts)}")
        for fact_i, fact in enumerate(facts, start=1):
            self._progress.update(
                f"[fact] conversation={self._active_conversation} "
                f"session={self._active_session_i}/{self._active_session_count} "
                f"{_progress_bar(fact_i, len(facts))} events={len(result.extracted_events)}"
            )
            recent_dialog_ids[:] = (recent_dialog_ids + fact.dia_ids)[-8:]
            self._evolve_one(
                fact.as_event_text(),
                dia_ids=fact.dia_ids or [t.dia_id for t in kept[:1]],
                session_index=session.index,
                session_date_time=session.date_time,
                speaker=fact.speaker,
                recent_dialog_ids=recent_dialog_ids,
                result=result,
                source_label=f"fact_{session.index}",
                progress_label=f"fact:{session.index}:{fact_i}/{len(facts)}",
            )

    def _evolve_one(
        self,
        event_text: str,
        *,
        dia_ids: List[str],
        session_index: int,
        session_date_time: str,
        speaker: str,
        recent_dialog_ids: List[str],
        result: "LoCoMoDriverResult",
        source_label: str,
        progress_label: Optional[str] = None,
    ) -> None:
        label = progress_label or source_label
        # Native LLM-hook architecture: watermark bits are embedded
        # inside backend.apply() via SDK-internal LLM-call interception.
        # Driver hands the event text to backend.apply, audits accumulate
        # in self.wm.audits, we slice the new ones for this event.
        self.wm.set_event_context(
            dia_ids=list(dia_ids),
            session_index=session_index,
            session_date_time=session_date_time,
            speaker=speaker,
            recent_dialog_ids=list(recent_dialog_ids),
            source_label=source_label,
        )
        before = len(self.wm.audits)
        event_text = self._target_llm_watermark_text(
            event_text,
            source_label=source_label,
        )
        operation = {
            "op": "add_memory",
            "text": event_text,
            "dia_ids": list(dia_ids),
            "session_index": session_index,
            "session_date_time": session_date_time,
            "speaker": speaker,
        }
        try:
            record = self.wm.backend.apply(operation)
        except Exception as exc:
            err_msg = str(exc)[:100] if str(exc) else ""
            self._progress.line(f"[apply failed] source={label} reason={exc.__class__.__name__}: {err_msg}")
            result.extracted_events.append(
                {
                    "session": session_index,
                    "source": source_label,
                    "speaker": speaker,
                    "text": event_text,
                    "applied": False,
                    "reason": f"apply_fail: {exc.__class__.__name__}",
                }
            )
            self.wm.clear_event_context()
            return
        finally:
            self.wm.clear_event_context()

        new_audits = self.wm.audits[before:]
        # NOTE: we deliberately do NOT extend result.audits here. The
        # canonical audit set is wm.sampler.audit_log (== merkle_log
        # leaves at seal time). Driver-side per-event extension misses
        # partial audits when apply() raises mid-call, causing
        # anchor.leaf_count > result.audits and breaking R2/R3
        # root_matches downstream. result.audits / result.decisions
        # are populated once after seal_session below.
        audit = new_audits[-1] if new_audits else None
        bits_for_event = sum(a.bits_embedded for a in new_audits)
        # Pick the last new audit as the representative one to log on the
        # extracted event (one event can trigger multiple SDK-internal
        # LLM calls → multiple audits; we summarize via the last one).
        last_audit = new_audits[-1] if new_audits else None
        result.extracted_events.append(
            {
                "session": session_index,
                "source": source_label,
                "speaker": speaker,
                "text": event_text,
                "applied": True,
                "llm_calls": len(new_audits),
                "selected": last_audit.selected_candidate_id if last_audit else "",
                "tau": last_audit.tau if last_audit else "",
                "bits_embedded": bits_for_event,
                "memory_record": record,
            }
        )


# --------------------------------------------------------------- #
# Capacity stats — copied from before, identical contract
# --------------------------------------------------------------- #


def _capacity_stats(
    audits: List[AuditRecord], decisions: List[DecisionPoint]
) -> Dict[str, Any]:
    if not audits or not decisions:
        return {
            "decisions": 0,
            "bits_embedded": 0,
            "bits_per_decision": 0.0,
            "avg_candidate_set_size": 0.0,
            "avg_entropy": 0.0,
            "acceptance_rate": 0.0,
            "by_carrier": {},
        }
    total_bits = sum(a.bits_embedded for a in audits)
    avg_size = sum(len(d.candidates) for d in decisions) / len(decisions)
    avg_entropy = sum(_entropy(d.probabilities) for d in decisions) / len(decisions)
    acceptance = sum(1 for d in decisions if len(d.candidates) >= 2) / len(decisions)
    by_carrier: Dict[str, Dict[str, Any]] = {}
    for d, a in zip(decisions, audits):
        carriers = [d.tau]
        for extra in getattr(a, "extra_carriers", ()) or ():
            if extra and extra not in carriers:
                carriers.append(extra)
        for carrier in carriers:
            bucket = by_carrier.setdefault(
                carrier,
                {
                    "decisions": 0,
                    "bits_embedded": 0,
                    "candidate_size_sum": 0,
                    "entropy_sum": 0.0,
                    "accepted": 0,
                },
            )
            bucket["decisions"] += 1
            bucket["bits_embedded"] += a.bits_embedded
            bucket["candidate_size_sum"] += len(d.candidates)
            bucket["entropy_sum"] += _entropy(d.probabilities)
            if len(d.candidates) >= 2:
                bucket["accepted"] += 1
    by_carrier_out = {
        tau: {
            "decisions": v["decisions"],
            "bits_embedded": v["bits_embedded"],
            "bits_per_decision": v["bits_embedded"] / v["decisions"],
            "avg_candidate_set_size": v["candidate_size_sum"] / v["decisions"],
            "avg_entropy": v["entropy_sum"] / v["decisions"],
            "acceptance_rate": v["accepted"] / v["decisions"],
        }
        for tau, v in by_carrier.items()
    }
    return {
        "decisions": len(audits),
        "bits_embedded": total_bits,
        "bits_per_decision": total_bits / len(audits),
        "avg_candidate_set_size": avg_size,
        "avg_entropy": avg_entropy,
        "acceptance_rate": acceptance,
        "by_carrier": by_carrier_out,
    }


def _entropy(probabilities: Dict[str, float]) -> float:
    h = 0.0
    for p in probabilities.values():
        if p > 0:
            h -= p * math.log2(p)
    return h


# --------------------------------------------------------------- #
# Defaults (offline, no LLM)
# --------------------------------------------------------------- #


def _format_turn(turn: LoCoMoTurn, session_date_time: str = "") -> str:
    """Canonical per-turn text for backend ingestion.

    Graphiti's official eval uses exactly ``role: content`` episodes and
    passes the session date separately as ``reference_time``. A-MEM's
    LoCoMo eval similarly passes the timestamp via ``add_note(...,
    time=...)``. Keep dia_id/date out of the body so ``no_watermark`` is
    a native-framework baseline; dia_ids remain in operation metadata
    and QA context source markers.
    """

    text = (turn.text or "").strip()
    speaker = (turn.speaker or "").strip()
    return f"{speaker}: {text}" if speaker else text


def _format_session_text(session: LoCoMoSession, turns: List[LoCoMoTurn]) -> str:
    lines = []
    if session.date_time:
        lines.append(f"Session {session.index} — {session.date_time}")
    if session.summary:
        lines.append(f"Summary: {session.summary}")
    for turn in turns:
        lines.append(_format_turn(turn))
    return "\n".join(lines)


def _keep_all_substantive_turns(turn: LoCoMoTurn, session_summary: str) -> bool:
    """Default filter: drop empty or 1-word chitchat ("Hey!", "ok")
    but keep everything else. The backend itself decides what's
    durable.
    """

    text = (turn.text or "").strip()
    if not text:
        return False
    if len(text.split()) < 3:
        return False
    return True


def _default_qa_responder(
    question: LoCoMoQuestion, context_text: str
) -> str:
    """Substring lookup; only used when LLM responder isn't wired.
    Real runs should use `make_locomo_qa_responder(llm_client)` from
    qa_eval.py.

    `context_text` is whatever the backend's `qa_context` returned
    (canonical retrieval rendering).
    """

    if isinstance(context_text, list):
        context_text = "\n".join(
            str(item.get("text", item)) if isinstance(item, dict) else str(item)
            for item in context_text
        )
    keywords = [w for w in question.question.lower().split() if len(w) > 3]
    if not keywords:
        return ""
    # Cheap match: pick the line in the rendered context with the
    # most keyword hits.
    best_line = ""
    best_score = 0
    for line in (context_text or "").splitlines():
        line_lower = line.lower()
        score = sum(1 for kw in keywords if kw in line_lower)
        if score > best_score:
            best_score = score
            best_line = line
    return best_line


def _default_qa_judge(question: LoCoMoQuestion, answer: str) -> bool:
    if not answer:
        return False
    gold = (question.answer or "").lower()
    return bool(gold) and gold in answer.lower()


def _evidence_recall(
    question: LoCoMoQuestion, retrieved_context: str
) -> float:
    """Fraction of QA's ``evidence`` dia_ids that appear in the
    **retrieved memory context** the LLM actually saw at QA time.

    Previously this scanned the full memory snapshot's dia_ids, which
    only measured ingestion completeness — every baseline (including
    random_replace with empty answers) returned the same number on
    conv 8 because they all ingested the same turns. This new version
    measures *retrieval* fidelity by checking whether each
    ``question.evidence`` dia_id (e.g. ``"D1:10"``) appears as a
    substring in the rendered ``retrieved_context`` string the QA
    pipeline produced. Backends emit dia_ids inline in their context
    rendering (A-mem: ``[D1:10]`` markers; Graphiti: edge name /
    valid_at lines), so substring match is sufficient.
    """

    if not question.evidence:
        return 1.0  # no evidence required
    if not retrieved_context:
        return 0.0
    hit = sum(1 for d in question.evidence if d in retrieved_context)
    return hit / len(question.evidence)


# Backwards-compat alias used by older example scripts.
def keyword_memory_extractor(turn, session_summary):  # pragma: no cover
    """Legacy stub: callers should switch to backend-native ingestion.

    Kept for import compatibility with `examples/run_real_llm_agent.py`
    and any external scripts that still expect this function.
    """

    if not _keep_all_substantive_turns(turn, session_summary):
        return []
    return [_format_turn(turn)]
