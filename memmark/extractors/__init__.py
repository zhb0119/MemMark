"""LoCoMo / Mem0 style fact extractors used by ingestion modes that
require pre-extracted facts (A-MEM follows this pattern).

The prompt is `CONVERSATION2FACTS_PROMPT` from the LoCoMo official
repo (`generative_agents/memory_utils.py`); we keep it verbatim so
extracted facts are directly comparable to the LoCoMo paper's
session-observation baseline.
"""

from memmark.extractors.locomo_facts import (
    ExtractedFact,
    extract_session_facts,
)

__all__ = ["ExtractedFact", "extract_session_facts"]
