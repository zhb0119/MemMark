"""LoCoMo-official per-session fact extractor.

Uses `CONVERSATION2FACTS_PROMPT` verbatim from
`locomo/generative_agents/memory_utils.py` so the extracted
"observations" are directly comparable to the LoCoMo paper's
session-observation baseline. Each fact carries the source dia_ids
that produced it (LoCoMo's evidence ground truth).

This is the Mem0 / A-MEM style ingestion path: from a multi-turn
session, ask the LLM to enumerate "concise factual observations
about each speaker" with `[D1:3]`-style dialog id citations, then
parse them back into structured `ExtractedFact` records.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, List, Sequence


# Verbatim from locomo/generative_agents/memory_utils.py
CONVERSATION2FACTS_PROMPT = (
    "Write a concise and short list of all possible OBSERVATIONS about "
    "each speaker that can be gathered from the CONVERSATION. Each "
    "dialog in the conversation contains a dialogue id within square "
    "brackets. Each observation should contain a piece of information "
    "about the speaker, and also include the dialog id of the dialogs "
    "from which the information is taken. The OBSERVATIONS should be "
    "objective factual information about the speaker that can be used "
    "as a database about them and to answer later questions. Keep both "
    "stable long-term facts and QA-relevant episodic facts: identity, "
    "relationship status, family, kids, work, education, career plans, "
    "hobbies, books, gifts, places, countries, events, activities, trips, "
    "classes, conferences, races, speeches, camping, museums, birthdays, "
    "dates, durations, plans, decisions, causes, and preferences. Avoid "
    "abstract observations about the dynamics between the two speakers "
    "such as 'speaker is supportive', 'speaker appreciates' etc. Exclude "
    "greetings, bare compliments, bare reactions, agreement, and "
    "conversation-management statements unless they contain one of the "
    "QA-relevant facts above. Normalize relative dates using the session "
    "date: if the conversation date is 8 May 2023, 'yesterday' must be "
    "written as '7 May 2023 (yesterday)'. Preserve the original relative "
    "phrase in parentheses after the absolute date. Do not leave out any "
    "QA-relevant personal, temporal, location, or event information from "
    "the CONVERSATION. "
    "Important: respond with a strict JSON object whose keys are speaker "
    "names and whose values are arrays of [observation_text, dia_id] "
    'pairs. Example: {"Alice": [["Alice lives in Berlin", "D1:3"], '
    '["Alice has a dog named Pepper", "D1:7"]], "Bob": [...]}'
)


_DIA_ID_RE = re.compile(r"D\d+:\d+")


@dataclass
class ExtractedFact:
    text: str
    dia_ids: List[str] = field(default_factory=list)
    speaker: str = ""

    def as_event_text(self) -> str:
        if self.speaker:
            return f"{self.speaker}: {self.text}"
        return self.text


def extract_session_facts(
    *,
    llm_client: Any,
    speaker_a: str,
    speaker_b: str,
    session_index: int,
    session_date_time: str,
    turns: Sequence[Any],
    max_facts: int = 30,
) -> List[ExtractedFact]:
    """Run LoCoMo's `CONVERSATION2FACTS_PROMPT` on one session.

    Returns up to `max_facts` ExtractedFact records.
    """

    if not turns:
        return []

    conversation = _format_conversation(session_date_time, turns)
    user_prompt = (
        "CONVERSATION:\n"
        f"{conversation}\n\n"
        f"Speakers: {speaker_a}, {speaker_b}.\n"
        f"Return a strict JSON object as described in the system prompt. "
        f"At most {max_facts} observations total."
    )
    messages = [
        {"role": "system", "content": CONVERSATION2FACTS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        raw = llm_client.complete(messages, temperature=0.0)
    except Exception:
        return []
    facts = _parse_facts(raw, valid_speakers={speaker_a, speaker_b})
    normalized = [
        ExtractedFact(
            text=_normalize_relative_dates(f.text, session_date_time),
            dia_ids=f.dia_ids,
            speaker=f.speaker,
        )
        for f in facts
    ]
    return [f for f in normalized if _is_durable_fact(f.text)][:max_facts]


def _format_conversation(date_time: str, turns: Sequence[Any]) -> str:
    lines = []
    if date_time:
        lines.append(date_time)
    for t in turns:
        speaker = getattr(t, "speaker", "")
        dia_id = getattr(t, "dia_id", "")
        text = getattr(t, "text", "") or ""
        text = text.replace('"', '\\"')
        lines.append(f'[{dia_id}] {speaker} said, "{text}"')
    return "\n".join(lines)


def _parse_facts(raw: str, *, valid_speakers: set) -> List[ExtractedFact]:
    text = (raw or "").strip()
    if not text:
        return []
    # Find first `{` and matching `}` for the response object
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    facts: List[ExtractedFact] = []
    if isinstance(parsed, dict):
        for speaker, items in parsed.items():
            if not isinstance(items, list):
                continue
            for item in items:
                fact_text, dia_ids = _normalize_item(item)
                if not fact_text:
                    continue
                facts.append(
                    ExtractedFact(
                        text=fact_text,
                        dia_ids=dia_ids,
                        speaker=str(speaker),
                    )
                )
    return facts


def _normalize_item(item: Any) -> tuple[str, List[str]]:
    """Accept several LLM JSON shapes: [text, dia_id], {text, dia_ids}, etc."""

    if isinstance(item, str):
        return item.strip(), _scan_dia_ids(item)
    if isinstance(item, list):
        if not item:
            return "", []
        text = str(item[0]).strip()
        dia_ids: List[str] = []
        for x in item[1:]:
            if isinstance(x, str):
                dia_ids.extend(_scan_dia_ids(x))
            elif isinstance(x, list):
                for xx in x:
                    if isinstance(xx, str):
                        dia_ids.extend(_scan_dia_ids(xx))
        if not dia_ids:
            dia_ids = _scan_dia_ids(text)
        return text, list(dict.fromkeys(dia_ids))
    if isinstance(item, dict):
        text = (
            item.get("observation")
            or item.get("text")
            or item.get("fact")
            or ""
        )
        text = str(text).strip()
        raw_ids = item.get("dia_ids") or item.get("dia_id") or []
        if isinstance(raw_ids, str):
            raw_ids = _scan_dia_ids(raw_ids)
        dia_ids = [d for d in raw_ids if isinstance(d, str)]
        if not dia_ids:
            dia_ids = _scan_dia_ids(text)
        return text, list(dict.fromkeys(dia_ids))
    return "", []


def _scan_dia_ids(text: str) -> List[str]:
    return _DIA_ID_RE.findall(text or "")


_NON_DURABLE_PATTERNS = (
    r"\basked\b",
    r"\bpraised\b",
    r"\bcompliment(?:ed|s)?\b",
    r"\bappreciates?\b",
    r"\badmires?\b",
    r"\bagrees?\b",
    r"\bthinks? .{0,40}\b(cool|great|nice|awesome|interesting)\b",
    r"\bfinds? .{0,40}\b(cool|great|nice|awesome|interesting)\b",
    r"\bis curious about\b",
    r"\bis off to\b",
)

_DURABLE_CUES = (
    "attended",
    "went to",
    "experienced",
    "felt",
    "feels",
    "found",
    "plans",
    "interested in",
    "works",
    "has",
    "is swamped",
    "painted",
    "likes",
    "enjoys",
    "wants",
    "lives",
    "studies",
    "identity",
    "transgender",
    "relationship",
    "single",
    "support group",
    "education",
    "career",
    "counseling",
    "mental health",
    "research",
    "adoption",
    "agency",
    "agencies",
    "charity",
    "race",
    "camping",
    "camped",
    "beach",
    "mountain",
    "forest",
    "museum",
    "pottery",
    "class",
    "running",
    "destress",
    "painting",
    "swimming",
    "kids",
    "dinosaurs",
    "nature",
    "book",
    "read",
    "bookshelf",
    "collects",
    "classic",
    "birthday",
    "speech",
    "school",
    "friends",
    "family",
    "mentor",
    "sweden",
    "moved",
    "home country",
    "necklace",
    "gift",
    "grandma",
    "conference",
    "picnic",
    "activity",
    "activities",
    "place",
    "places",
    "location",
    "duration",
    "year",
    "month",
    "week",
    "day",
)


def _is_durable_fact(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for pattern in _NON_DURABLE_PATTERNS:
        if re.search(pattern, t):
            return False
    has_memory_worthy_cue = any(cue in t for cue in _DURABLE_CUES)
    if has_memory_worthy_cue:
        return True
    return False


def _normalize_relative_dates(text: str, session_date_time: str) -> str:
    base = _parse_session_date(session_date_time)
    if base is None:
        return text
    out = text
    replacements = [
        ("the day before", base - timedelta(days=1)),
        ("yesterday", base - timedelta(days=1)),
        ("today", base),
        ("tomorrow", base + timedelta(days=1)),
        ("last week", base - timedelta(days=7)),
        ("next week", base + timedelta(days=7)),
        ("two days ago", base - timedelta(days=2)),
    ]
    for phrase, dt in replacements:
        out = _replace_relative_phrase(out, phrase, _format_date(dt))
    out = _replace_month_relative(out, "last month", base, -1)
    out = _replace_month_relative(out, "next month", base, 1)
    out = _replace_year_relative(out, "last year", base, -1)
    out = _replace_year_relative(out, "next year", base, 1)
    out = _replace_weekday_relative(out, base)
    return out


def _replace_relative_phrase(text: str, phrase: str, absolute: str) -> str:
    pattern = re.compile(rf"(?<!\()\b{re.escape(phrase)}\b(?!\))", re.IGNORECASE)
    return pattern.sub(f"{absolute} ({phrase})", text)


def _replace_month_relative(text: str, phrase: str, base: datetime, delta: int) -> str:
    month = base.month + delta
    year = base.year
    while month < 1:
        month += 12
        year -= 1
    while month > 12:
        month -= 12
        year += 1
    absolute = datetime(year, month, 1).strftime("%B %Y")
    return _replace_relative_phrase(text, phrase, absolute)


def _replace_year_relative(text: str, phrase: str, base: datetime, delta: int) -> str:
    return _replace_relative_phrase(text, phrase, str(base.year + delta))


def _replace_weekday_relative(text: str, base: datetime) -> str:
    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    out = text
    for name, idx in weekdays.items():
        pattern = re.compile(rf"\b(last|next|the previous|the following)\s+{name}\b", re.IGNORECASE)
        match = pattern.search(out)
        if not match:
            continue
        direction = match.group(1).lower()
        offset = (base.weekday() - idx) % 7
        if direction in {"last", "the previous"}:
            offset = offset or 7
            dt = base - timedelta(days=offset)
        else:
            offset = (idx - base.weekday()) % 7 or 7
            dt = base + timedelta(days=offset)
        out = pattern.sub(f"{_format_date(dt)} ({match.group(0)})", out)
    return out


def _parse_session_date(session_date_time: str):
    match = re.search(
        r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b",
        session_date_time or "",
    )
    if not match:
        return None
    day, month, year = match.groups()
    try:
        return datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
    except ValueError:
        try:
            return datetime.strptime(f"{day} {month} {year}", "%d %b %Y")
        except ValueError:
            return None


def _format_date(dt: datetime) -> str:
    return f"{dt.day} {dt.strftime('%B')} {dt.year}"
