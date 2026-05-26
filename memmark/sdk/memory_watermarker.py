"""Public SDK entry point.

Thin facade around :class:`memmark.llm.watermarked.WatermarkedSampler`.
The watermark itself runs *inside* each backend's SDK via LLM-call
interception (see ``memmark/llm/watermarked.py``); this class owns
the shared sampler, the per-session Merkle log, and the audit list,
and threads them through to the backend on construction.

Old ``evolve()`` / carrier-planner path is gone — bits are embedded
in the memory system's own internal LLM decisions, not in an
external candidate-overlay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from memmark.backends.base import MemoryBackendAdapter
from memmark.core.context import make_watermark_version, resolve_secret_key
from memmark.core.types import AuditRecord, SessionHeader
from memmark.llm.watermarked import WatermarkedSampler


SAMPLER_MODES = (
    "watermark",
    "signed_metadata_only",
    "random_replace",
    "no_watermark",
    "kgmark_graphiti",
)


@dataclass(frozen=True)
class EvolveResult:
    """Backwards-compat structure for the few call sites that still
    reference it. The new pipeline does not produce per-event
    EvolveResults — audits are per-LLM-call, accumulated in
    ``MemoryWatermarker.audits``.
    """

    decision: object = None
    audit: object = None
    memory_record: dict = None  # type: ignore[assignment]


class MemoryWatermarker:
    """Owns the :class:`WatermarkedSampler` and attaches it to the
    backend's SDK at construction time.

    Surface API kept compatible with the old version where possible:
    callers still do ``MemoryWatermarker(backend=..., payload_bits=...,
    agent_id=..., user_id=..., session_id=..., secret_key=...,
    sampler_mode=...)`` and read back ``.audits`` / ``.seal_session()``.

    Removed: ``evolve()``, ``_plan_carrier()``, ``carrier_planner``,
    ``carrier``. Bits flow through the backend's own LLM calls.
    """

    def __init__(
        self,
        *,
        backend: MemoryBackendAdapter,
        payload_bits: str,
        agent_id: str = "memmark-agent",
        user_id: str = "memmark-user",
        session_id: str = "default-session",
        secret_key: Optional[str] = None,
        watermark_version: Optional[str] = None,
        sampler_mode: str = "watermark",
        random_seed: Optional[int] = None,
        target_k: int = 4,
    ) -> None:
        if sampler_mode not in SAMPLER_MODES:
            raise ValueError(
                f"sampler_mode must be one of {SAMPLER_MODES}, got {sampler_mode}"
            )
        self.backend = backend
        self.payload_bits = self._validate_bits(payload_bits)
        self.agent_id = agent_id
        self.user_id = user_id
        self.session_id = session_id
        self.sampler_mode = sampler_mode
        self.random_seed = random_seed
        resolved_key = resolve_secret_key(
            secret_key or os.getenv("MEMMARK_KEY") or "memmark-default-dev-key"
        )
        self.secret_key = resolved_key
        self.watermark_version = watermark_version or make_watermark_version(
            sdk_version="memmark-native-v0.1",
            model=os.getenv("MEMMARK_MODEL") or "unknown",
        )
        self.sampler = WatermarkedSampler(
            secret_key=resolved_key,
            payload_bits=self.payload_bits,
            agent_id=agent_id,
            user_id=user_id,
            session_id=session_id,
            sampler_mode=sampler_mode,
            watermark_version=self.watermark_version,
            target_k=target_k,
        )
        # Inject the sampler into the backend's SDK. Backends without
        # an internal LLM (JsonStore) implement ``attach_sampler`` as a
        # no-op so this is safe even for the smoke baseline.
        if hasattr(backend, "attach_sampler"):
            backend.attach_sampler(self.sampler)

    # ----- accessors used by driver / RQ runners ----- #
    @property
    def audits(self) -> List[AuditRecord]:
        return list(self.sampler.audit_log)

    @property
    def bit_index(self) -> int:
        return self.sampler.bit_index

    def seal_session(self) -> SessionHeader:
        return self.sampler.seal_session()

    def set_event_context(self, **fields) -> None:
        """Driver hook: tag the next batch of LLM calls with LoCoMo
        event metadata (dia_ids, session_index, turn_id, ...). The
        sampler folds these into ``ctx_t`` for nonce derivation, so
        audits can be linked back to the LoCoMo turn that triggered
        them.
        """

        self.sampler.set_event_context(**fields)

    def clear_event_context(self) -> None:
        self.sampler.clear_event_context()

    # ----- backwards-compat shim ----- #
    def evolve(self, text: str, **_kwargs) -> EvolveResult:
        """Compatibility wrapper for callers that used the old
        per-event ``evolve()`` API.

        Behavior under native LLM-hook:
          * Watermark bits are embedded inside the backend's internal
            LLM calls during apply(); this method just routes the
            event text through ``backend.apply()``.
          * The returned ``EvolveResult`` is mostly informational —
            real audits live in ``self.audits``.
        """

        operation = {
            "op": "add_memory",
            "text": text,
            "dia_ids": list(_kwargs.get("dia_ids", []) or []),
            "session_index": _kwargs.get("session_index"),
            "speaker": _kwargs.get("speaker", ""),
            "session_date_time": _kwargs.get("session_date_time", ""),
        }
        record = self.backend.apply(operation)
        latest = self.audits[-1] if self.audits else None
        return EvolveResult(decision=None, audit=latest, memory_record=record)

    @staticmethod
    def _validate_bits(bits: str) -> str:
        cleaned = "".join(ch for ch in bits if ch in {"0", "1"})
        if not cleaned:
            raise ValueError("payload_bits must contain at least one bit.")
        return cleaned
