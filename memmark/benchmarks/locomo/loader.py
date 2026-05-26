"""LoCoMo dataset loader.

Reads `locomo10.json` (10 long conversations × ~27 sessions each) and
exposes typed Conversation / Session / Turn / Question objects so the
driver can iterate without re-parsing.

LoCoMo official: https://github.com/snap-research/locomo
Paper: arXiv 2402.17753
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class LoCoMoTurn:
    speaker: str
    dia_id: str  # e.g. "D1:3"
    text: str
    session_index: int


@dataclass(frozen=True)
class LoCoMoSession:
    index: int
    date_time: str
    turns: List[LoCoMoTurn]
    summary: str = ""
    observations: List[str] = None  # type: ignore[assignment]
    events: List[str] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class LoCoMoQuestion:
    question: str
    answer: str
    evidence: List[str]
    category: int


@dataclass(frozen=True)
class LoCoMoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[LoCoMoSession]
    qa: List[LoCoMoQuestion]


_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def load_locomo(path: str | Path) -> List[LoCoMoConversation]:
    """Load `locomo10.json` and return parsed conversations."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    conversations: List[LoCoMoConversation] = []
    for sample in raw:
        conversations.append(_parse_sample(sample))
    return conversations


def _parse_sample(sample: dict) -> LoCoMoConversation:
    convo = sample.get("conversation", {})
    speaker_a = convo.get("speaker_a", "")
    speaker_b = convo.get("speaker_b", "")

    session_indices = sorted(
        int(m.group(1))
        for k in convo.keys()
        if (m := _SESSION_KEY_RE.match(k))
    )
    summaries = sample.get("session_summary", {})
    obs = sample.get("observation", {})
    events = sample.get("event_summary", {})

    sessions: List[LoCoMoSession] = []
    for idx in session_indices:
        turns_raw = convo.get(f"session_{idx}", []) or []
        date_time = convo.get(f"session_{idx}_date_time", "")
        turns = [
            LoCoMoTurn(
                speaker=t.get("speaker", ""),
                dia_id=t.get("dia_id", ""),
                text=t.get("text", ""),
                session_index=idx,
            )
            for t in turns_raw
            if isinstance(t, dict)
        ]
        sessions.append(
            LoCoMoSession(
                index=idx,
                date_time=date_time,
                turns=turns,
                summary=str(summaries.get(f"session_{idx}_summary", ""))
                or str(summaries.get(f"session_{idx}", "")),
                observations=_flatten_strings(obs.get(f"session_{idx}_observation")),
                events=_flatten_strings(events.get(f"events_session_{idx}")),
            )
        )

    qa: List[LoCoMoQuestion] = []
    for q in sample.get("qa", []) or []:
        if not isinstance(q, dict):
            continue
        category = int(q.get("category", 0))
        # LoCoMo cat-5 (adversarial) ships the gold under
        # ``adversarial_answer`` rather than ``answer``; A-mem's
        # load_dataset.py:17-21 selects between the two by category
        # and we mirror that so cat-5 gold isn't silently empty.
        if category == 5:
            gold = str(q.get("adversarial_answer") or q.get("answer") or "")
        else:
            gold = str(q.get("answer", ""))
        qa.append(
            LoCoMoQuestion(
                question=str(q.get("question", "")),
                answer=gold,
                evidence=[str(e) for e in (q.get("evidence") or [])],
                category=category,
            )
        )

    return LoCoMoConversation(
        sample_id=str(sample.get("sample_id", "")),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        sessions=sessions,
        qa=qa,
    )


def _flatten_strings(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("event") or item.get("observation")
                if isinstance(t, str):
                    out.append(t)
        return out
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten_strings(v))
        return out
    return [str(value)]
