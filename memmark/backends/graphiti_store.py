"""Graphiti backend adapter (https://github.com/getzep/graphiti).

Graphiti is a temporal context graph: each memory becomes an *episode*
with a `reference_time`. The graph evolves through fact invalidation
and supersession. We expose:

  * `add_memory` → graphiti.add_episode()
  * `update_memory` → adds a new episode that supersedes the prior fact
    (Graphiti handles this natively when a contradicting fact arrives)
  * `delete_memory` → graphiti.remove_episode()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import os
import re
import threading
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence

from memmark.backends.base import MemoryBackendAdapter
from memmark.core.context import make_watermark_version, sha256_text, stable_json
from memmark.core.types import AuditRecord, Candidate, DecisionPoint

try:
    from graphiti_core import Graphiti  # type: ignore
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient  # type: ignore
    from graphiti_core.embedder.client import EmbedderClient  # type: ignore
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # type: ignore
    from graphiti_core.llm_client import LLMConfig, OpenAIClient  # type: ignore
    from graphiti_core.nodes import EpisodeType  # type: ignore
    HAS_GRAPHITI = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Graphiti = None  # type: ignore
    OpenAIRerankerClient = None  # type: ignore
    EmbedderClient = object  # type: ignore
    OpenAIEmbedder = None  # type: ignore
    OpenAIEmbedderConfig = None  # type: ignore
    LLMConfig = None  # type: ignore
    OpenAIClient = None  # type: ignore
    EpisodeType = None  # type: ignore
    HAS_GRAPHITI = False


class GraphitiBackend(MemoryBackendAdapter):
    """Backend adapter wrapping `graphiti_core.Graphiti`.

    Per Graphiti's official eval (`tests/evals/eval_e2e_graph_building.py`),
    the LongMemEval / LoCoMo ingestion is **per-turn**: each
    dialog turn becomes one episode whose `reference_time` is the
    session's date_time (not now()). We therefore set
    `preferred_ingestion_mode = "turn"` and read
    `operation["session_date_time"]` to populate `reference_time`.
    """

    preferred_ingestion_mode = "turn"
    watermark_with_target_llm = True

    def __init__(
        self,
        *,
        graphiti: Optional[Any] = None,
        group_id: Optional[str] = None,
        source_description: str = "memmark watermark",
    ) -> None:
        if graphiti is None:
            if not HAS_GRAPHITI:
                raise RuntimeError(
                    "graphiti_core not installed. `pip install graphiti-core` "
                    "or pass `graphiti=` explicitly."
                )
            llm_client = OpenAIClient(
                config=LLMConfig(
                    api_key=os.getenv("OPENAI_API_KEY") or os.getenv("MEMMARK_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("MEMMARK_BASE_URL"),
                    model=os.getenv("OPENAI_MODEL") or os.getenv("MEMMARK_MODEL"),
                    small_model=os.getenv("OPENAI_MODEL") or os.getenv("MEMMARK_MODEL"),
                )
            )
            embedder = _build_openai_embedder()
            cross_encoder = _build_openai_reranker()
            graphiti = _build_graphiti_instance(
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder,
            )
        self.graphiti = graphiti
        self.group_id = group_id or os.getenv("MEMMARK_GRAPHITI_GROUP", "memmark")
        self._base_group_id = self.group_id
        self.source_description = source_description
        self._memories: List[Dict[str, Any]] = []
        self._kgmark_sampler: Optional[Any] = None
        self._kgmark_audits: List[AuditRecord] = []
        self._loop = asyncio.new_event_loop()
        self._loop_lock = threading.RLock()
        if _env_bool("MEMMARK_GRAPHITI_BUILD_INDICES", True) and hasattr(
            self.graphiti, "build_indices_and_constraints"
        ):
            self._run_async(
                self.graphiti.build_indices_and_constraints(
                    delete_existing=_env_bool("MEMMARK_GRAPHITI_DELETE_EXISTING_INDICES", False)
                )
            )

    # -- MemoryBackendAdapter ------------------------------------- #
    def begin_conversation(self, sample_id: str) -> None:
        """Use one Graphiti group per LoCoMo conversation.

        Graphiti's official eval partitions graphs by user/sample. Doing
        the same here prevents cross-conversation retrieval leakage while
        keeping the no-watermark baseline native to Graphiti.
        """

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id or "sample").strip("_")
        if safe:
            self.group_id = f"{self._base_group_id}_{safe}"

    def snapshot(self) -> List[Dict[str, Any]]:
        return [dict(m) for m in self._memories]

    def apply(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        return self._run_async(self.apply_async(operation))

    async def apply_async(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        op = operation.get("op")
        evidence = list(operation.get("dia_ids", []))
        session_index = operation.get("session_index")
        speaker = operation.get("speaker", "")
        if op == "add_memory":
            text = operation["text"]
            session_date_time = operation.get("session_date_time", "")
            ref_time = _require_reference_time(session_date_time)
            # Aligned with Graphiti's official eval
            # (graphiti/tests/evals/eval_e2e_graph_building.py:53-61):
            # name='', source_description='', source=EpisodeType.message,
            # reference_time = session date as datetime, group_id per
            # user. NO previous_episode_uuids — that's a podcast_runner
            # demo optimization, not the eval protocol.
            results = await self.graphiti.add_episode(
                name="",
                episode_body=text,
                source_description="",
                reference_time=ref_time,
                source=EpisodeType.message if EpisodeType is not None else None,
                group_id=self.group_id,
            )
            ep_uuid = getattr(getattr(results, "episode", None), "uuid", None) or (
                results.episode.uuid if hasattr(results, "episode") else f"g{len(self._memories) + 1}"
            )
            record = {
                "id": ep_uuid,
                "text": text,
                "raw_episode_body": text,
                "links": list(operation.get("links", [])),
                "reference_time": ref_time.isoformat(),
                "parsed_reference_time_source": "session_date_time",
                "dia_ids": evidence,
                "session_index": session_index,
                "speaker": speaker,
                "session_date_time": session_date_time,
            }
            await self._apply_kgmark_record(record)
            self._memories.append(record)
            return record
        if op == "update_memory":
            target_id = operation["memory_id"]
            new_text = operation["text"]
            now = datetime.now(timezone.utc)
            results = await self.graphiti.add_episode(
                name=f"memmark_update_{target_id}",
                episode_body=new_text,
                source_description=self.source_description + " (update)",
                reference_time=now,
                source=EpisodeType.message if EpisodeType is not None else None,
                group_id=self.group_id,
            )
            ep_uuid = getattr(getattr(results, "episode", None), "uuid", None) or target_id
            for record in self._memories:
                if record["id"] == target_id:
                    record["text"] = new_text
                    record["last_update_id"] = ep_uuid
                    if evidence:
                        record["dia_ids"] = list(
                            dict.fromkeys(list(record.get("dia_ids", [])) + evidence)
                        )
                    if session_index is not None:
                        record["session_index"] = session_index
                    if speaker:
                        record["speaker"] = speaker
                    break
            return {"id": target_id, "text": new_text, "supersede_id": ep_uuid}
        if op == "delete_memory":
            target_id = operation["memory_id"]
            try:
                await self.graphiti.remove_episode(target_id)
            except Exception:
                pass
            self._memories = [m for m in self._memories if m["id"] != target_id]
            return {"id": target_id, "deleted": True}
        raise ValueError(f"Unsupported operation: {op}")

    # ----- watermark sampler injection ----- #
    def attach_sampler(self, sampler: Any) -> None:
        """Hot-swap Graphiti's ``llm_client`` with the watermark
        wrapper for MemMark baselines.

        ``no_watermark`` deliberately leaves Graphiti completely native:
        no AgentMark prompt wrapping, no candidate enumeration, no audit
        trace side effects. The other three baselines operate at the
        same SDK-internal LLM-call granularity so RQ1-RQ5 compare like
        with like: watermark = keyed pick, random_replace = random pick,
        signed_metadata_only = signed sidecar without embedded bits.
        """

        from memmark.llm.watermarked import make_watermarked_graphiti_client

        mode = getattr(sampler, "sampler_mode", "watermark")
        if mode == "no_watermark":
            return None
        if mode == "kgmark_graphiti":
            self._kgmark_sampler = sampler
            return None
        if self.graphiti is not None and hasattr(self.graphiti, "llm_client"):
            wm_client = make_watermarked_graphiti_client(
                sampler, self.graphiti.llm_client
            )
            self.graphiti.llm_client = wm_client
            if hasattr(self.graphiti, "clients") and hasattr(self.graphiti.clients, "llm_client"):
                self.graphiti.clients.llm_client = wm_client

    async def _apply_kgmark_record(self, record: Dict[str, Any]) -> None:
        """Apply KGMark-style latent watermarking to a Graphiti write.

        This baseline follows the public KGMark pipeline at minimum
        viable granularity for LoCoMo RQ1: Graphiti first builds its KG
        natively; we then derive a local episode/subgraph latent from
        the same embedding path used by Graphiti retrieval, overwrite a
        keyed Fourier/masked subset with a signature, and persist the
        watermarked latent as record metadata. The watermarked latent is
        also used as a deterministic tie-break/boost during QA retrieval
        reranking, so the embedding watermark participates in Graphiti's
        downstream answer path instead of being an unused sidecar.
        """

        sampler = self._kgmark_sampler
        if sampler is None:
            return
        idx = len(self._kgmark_audits)
        secret_key = getattr(sampler, "secret_key", "")
        payload_bits = getattr(sampler, "payload_bits", "") or "0"
        bit_index = int(getattr(sampler, "bit_index", 0))
        bit = payload_bits[bit_index % len(payload_bits)] if payload_bits else "0"
        local_embedding = await self._kgmark_local_embedding(record)
        ctx_payload = {
            "baseline": "kgmark_graphiti",
            "group_id": self.group_id,
            "episode_id": record.get("id", ""),
            "text_hash": sha256_text(record.get("text", "")),
            "reference_time": record.get("reference_time", ""),
            "dia_ids": record.get("dia_ids", []),
            "prev_state_hash": self._kgmark_state_hash(),
            "round": idx,
            "embedding_dim": len(local_embedding),
            "strength": _kgmark_strength(),
            "mask_ratio": _kgmark_mask_ratio(),
        }
        context = stable_json(ctx_payload)
        keyed = hmac.new(secret_key.encode("utf-8"), context.encode("utf-8"), hashlib.sha256).hexdigest()
        watermarked_embedding, mask_indices, signature_values = _kgmark_embed_latent(
            local_embedding,
            secret_key=secret_key,
            context=context,
            payload_bit=bit,
        )
        watermark_bit = str((int(keyed, 16) ^ int(bit)) & 1)
        tag = sha256_text(stable_json({"keyed": keyed, "bit": watermark_bit}))
        record["kgmark"] = {
            "scheme": "kgmark_graphiti_embedding",
            "bit": watermark_bit,
            "tag": tag,
            "context_hash": sha256_text(context),
            "round": idx,
            "injection_point": "episode_local_subgraph_embedding",
            "latent_dim": len(watermarked_embedding),
            "mask_ratio": _kgmark_mask_ratio(),
            "mask_size": len(mask_indices),
            "strength": _kgmark_strength(),
            "seed_alias": sha256_text(secret_key)[:12],
            "embedding_hash_before": _hash_float_vector(local_embedding),
            "embedding_hash_after": _hash_float_vector(watermarked_embedding),
            "mask_indices": mask_indices[:64],
            "signature_preview": [round(v, 6) for v in signature_values[:16]],
        }
        record["kgmark_embedding"] = watermarked_embedding
        decision = DecisionPoint(
            decision_id=f"kgmark_graphiti:{self.group_id}:{idx}",
            tau="kg_embedding_latent",
            candidates=[
                Candidate(
                    candidate_id="kgmark_bit_0",
                    carrier_type="kg_embedding_latent",
                    payload={"bit": "0"},
                    operation={"op": "kgmark_embed", "record_id": record.get("id", "")},
                ),
                Candidate(
                    candidate_id="kgmark_bit_1",
                    carrier_type="kg_embedding_latent",
                    payload={"bit": "1"},
                    operation={"op": "kgmark_embed", "record_id": record.get("id", "")},
                ),
            ],
            probabilities={"kgmark_bit_0": 0.5, "kgmark_bit_1": 0.5},
            context=context,
            round_num=idx,
            nonce=keyed,
            watermark_version=make_watermark_version(
                sdk_version="kgmark-graphiti-v0.2",
                model=os.getenv("GRAPHITI_EMBEDDING_MODEL") or os.getenv("OPENAI_MODEL") or "graphiti",
                extra="carrier=episode_local_subgraph_embedding",
            ),
        )
        from memmark.core.commitment import make_commitment

        audit = make_commitment(
            decision,
            selected_candidate_id=f"kgmark_bit_{watermark_bit}",
            bits_embedded=1,
            bit_index_after=bit_index + 1,
        )
        self._kgmark_audits.append(audit)
        sampler.audit_log.append(audit)
        sampler.bit_index = bit_index + 1

    async def _kgmark_local_embedding(self, record: Dict[str, Any]) -> List[float]:
        text = record.get("text", "") or ""
        local_context = stable_json(
            {
                "episode": text,
                "links": record.get("links", []),
                "speaker": record.get("speaker", ""),
                "session_index": record.get("session_index"),
                "reference_time": record.get("reference_time", ""),
            }
        )
        embedder = getattr(self.graphiti, "embedder", None) or getattr(
            getattr(self.graphiti, "clients", None), "embedder", None
        )
        if embedder is not None and hasattr(embedder, "create"):
            try:
                return [float(x) for x in await embedder.create(local_context)]
            except Exception:
                pass
        dim = _embedding_dim() or 1536
        return _deterministic_embedding(local_context, dim)

    def _kgmark_state_hash(self) -> str:
        state = [
            {
                "id": item.get("id", ""),
                "text_hash": sha256_text(item.get("text", "")),
                "kgmark": item.get("kgmark", {}),
            }
            for item in self._memories
        ]
        return sha256_text(stable_json(state))

    async def search_async(self, query: str, top_k: int = 5):
        edges = await self.graphiti.search(
            query=query, group_ids=[self.group_id], num_results=top_k
        )
        if self._kgmark_sampler is None or not edges:
            return edges
        return await self._kgmark_rerank_edges(query, list(edges))

    async def _kgmark_rerank_edges(self, query: str, edges: List[Any]) -> List[Any]:
        query_embedding = await self._kgmark_query_embedding(query)

        def score(edge: Any) -> float:
            fact = getattr(edge, "fact", "") or ""
            source = self._source_record_for_fact(fact)
            if not source:
                return 0.0
            return _cosine_similarity(query_embedding, source.get("kgmark_embedding") or [])

        ranked = sorted(enumerate(edges), key=lambda item: (score(item[1]), -item[0]), reverse=True)
        return [edge for _, edge in ranked]

    async def _kgmark_query_embedding(self, query: str) -> List[float]:
        embedder = getattr(self.graphiti, "embedder", None) or getattr(
            getattr(self.graphiti, "clients", None), "embedder", None
        )
        if embedder is not None and hasattr(embedder, "create"):
            try:
                return [float(x) for x in await embedder.create(query)]
            except Exception:
                pass
        dim = _embedding_dim() or 1536
        return _deterministic_embedding(query, dim)

    def search(self, query: str, top_k: int = 5):
        return self._run_async(self.search_async(query, top_k=top_k))

    # ----- canonical QA context ----- #
    def qa_context(
        self,
        question: str,
        k: int = 10,
        *,
        category: Any = None,
        gold_answer: Any = None,
        llm_client: Any = None,
    ) -> Dict[str, Any]:
        """Graphiti-native QA context.

        This keeps the original Graphiti search path unchanged.
        """

        try:
            edges = self._run_async(self.search_async(question, top_k=k))
        except Exception:
            from memmark.benchmarks.locomo.qa_eval import _default_render_memory

            return {
                "mode": "context",
                "text": _default_render_memory(self.snapshot()),
            }

        if not edges:
            context_text = "(no related facts in graph)"
        else:
            lines: List[str] = []
            for edge in edges:
                fact = getattr(edge, "fact", "") or ""
                name = getattr(edge, "name", "") or ""
                source_record = self._source_record_for_fact(fact)
                marker = self._format_source_marker(source_record)
                talk_time = (source_record or {}).get("session_date_time", "")
                head = f"[{name}] " if name else ""
                time_part = f" DATE OF CONVERSATION: {talk_time}." if talk_time else ""
                source = f" {marker}" if marker else ""
                lines.append(f"- {head}{fact}.{time_part}{source}")
            context_text = "\n".join(lines)

        if llm_client is None or category is None:
            return {"mode": "context", "text": context_text}

        from memmark.benchmarks.locomo.qa_eval import build_cat_aware_qa_prompt

        user_prompt, temperature = build_cat_aware_qa_prompt(
            category, context_text, question, gold_answer=gold_answer,
        )
        try:
            answer = llm_client.complete(
                [{"role": "user", "content": user_prompt}], temperature=temperature
            )
        except Exception:
            answer = ""
        return {
            "mode": "answer",
            "text": (answer or "").strip(),
            "context": context_text,
            "context_chars": len(context_text),
            "user_prompt": user_prompt,
            "retrieved_context": context_text,
        }

    def _source_record_for_fact(self, fact: str) -> Optional[Dict[str, Any]]:
        best_score = 0.0
        best_record: Optional[Dict[str, Any]] = None
        fact_norm = self._normalize_for_match(fact)
        if not fact_norm:
            return None
        fact_tokens = set(fact_norm.split())
        for record in self._memories:
            dia_ids = record.get("dia_ids") or []
            text_norm = self._normalize_for_match(record.get("text") or "")
            if not dia_ids or not text_norm:
                continue
            text_tokens = set(text_norm.split())
            overlap = len(fact_tokens & text_tokens) / max(1, len(fact_tokens | text_tokens))
            ratio = SequenceMatcher(None, fact_norm, text_norm).ratio()
            contains = 1.0 if fact_norm in text_norm or text_norm in fact_norm else 0.0
            score = max(contains, overlap, ratio)
            if score > best_score:
                best_score = score
                best_record = record
        if best_record is None or best_score < 0.28:
            return None
        return best_record

    def _source_marker_for_fact(self, fact: str) -> str:
        return self._format_source_marker(self._source_record_for_fact(fact))

    @staticmethod
    def _format_source_marker(record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return ""
        dia_ids = record.get("dia_ids") or []
        return f"[{','.join(str(d) for d in dia_ids)}]" if dia_ids else ""

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def _run_async(self, coro):
        with self._loop_lock:
            return self._loop.run_until_complete(coro)


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


def _require_reference_time(text: str) -> datetime:
    dt = _parse_reference_time(text)
    if dt is None:
        raise ValueError(
            "Graphiti LoCoMo ingestion requires a parseable session_date_time; "
            "refusing to fall back to runtime datetime.now(). "
            f"Got: {text!r}"
        )
    return dt


def _parse_reference_time(text: str):
    """Parse LoCoMo / LongMemEval session date_time string to UTC datetime.

    LoCoMo format examples:  '7 May 2023, 11:38 am', 'May 8, 2023 at 09:00'
    LongMemEval format:      '2023/05/07 (Sun) 11:38'
    Returns None on unrecognized format; callers must not silently use
    the runtime clock because that corrupts Graphiti temporal facts.
    """

    if not text:
        return None
    raw = text.strip()
    candidates = [
        "%d %B %Y, %I:%M %p",       # "7 May 2023, 11:38 am"
        "%d %B %Y, %H:%M",          # "7 May 2023, 11:38"
        "%I:%M %p on %d %B, %Y",    # "1:56 pm on 8 May, 2023"
        "%I:%M%p on %d %B, %Y",     # "1:56pm on 8 May, 2023"
        "%H:%M on %d %B, %Y",       # "13:56 on 8 May, 2023"
        "%B %d, %Y at %H:%M",        # "May 8, 2023 at 09:00"
        "%B %d, %Y, %I:%M %p",       # "May 8, 2023, 9:00 am"
        "%Y/%m/%d (%a) %H:%M",       # "2023/05/07 (Sun) 11:38"
        "%Y/%m/%d (%a) %H:%M UTC",   # Graphiti official eval style
        "%Y-%m-%d %H:%M:%S",         # ISO-ish
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in candidates:
        for candidate in (raw, _normalize_ampm(raw)):
            try:
                dt = datetime.strptime(candidate, fmt)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except ValueError:
                continue
    try:
        iso = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _normalize_ampm(text: str) -> str:
    return re.sub(
        r"\b(am|pm)\b",
        lambda match: match.group(1).upper(),
        text,
        flags=re.IGNORECASE,
    )


def _first_env(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _build_graphiti_instance(*, llm_client: Any, embedder: Any, cross_encoder: Any) -> Any:
    kwargs = {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "neo4j"),
        "llm_client": llm_client,
        "embedder": embedder,
        "cross_encoder": cross_encoder,
    }

    return Graphiti(**kwargs)


def _build_openai_embedder():
    model = _first_env(
        "GRAPHITI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "MEMMARK_EMBEDDING_MODEL",
        "EMBEDDING_MODEL",
    )
    dim = _embedding_dim()
    if not model:
        if dim is None:
            return _BatchLimitedEmbedder(OpenAIEmbedder())
        return _BatchLimitedEmbedder(OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_dim=dim)))
    return _BatchLimitedEmbedder(OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=_first_env(
                "GRAPHITI_EMBEDDING_API_KEY",
                "OPENAI_EMBEDDING_API_KEY",
                "MEMMARK_EMBEDDING_API_KEY",
                "OPENAI_API_KEY",
                "MEMMARK_API_KEY",
            ),
            base_url=_first_env(
                "GRAPHITI_EMBEDDING_BASE_URL",
                "OPENAI_EMBEDDING_BASE_URL",
                "MEMMARK_EMBEDDING_BASE_URL",
                "OPENAI_BASE_URL",
                "MEMMARK_BASE_URL",
            ),
            embedding_model=model,
            embedding_dim=dim or 1536,
        )
    ))


class _BatchLimitedEmbedder(EmbedderClient):  # type: ignore[misc, valid-type]
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._max_batch_size = _embedding_batch_size()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def create(self, input_data: Any) -> List[float]:
        return await self._inner.create(input_data)

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        if len(input_data_list) <= self._max_batch_size:
            return await self._inner.create_batch(input_data_list)
        outputs: List[List[float]] = []
        for chunk in self._chunks(input_data_list):
            result = await self._inner.create_batch(chunk)
            outputs.extend(result)
        return outputs

    async def embed(self, input_data: Any) -> Any:
        if not self._should_split(input_data):
            return await self._inner.embed(input_data)
        for chunk in self._chunks(input_data):
            pass
        return await self.create_batch(input_data)

    def _should_split(self, input_data: Any) -> bool:
        return isinstance(input_data, list) and len(input_data) > self._max_batch_size

    def _chunks(self, input_data: Sequence[Any]):
        for start in range(0, len(input_data), self._max_batch_size):
            yield list(input_data[start:start + self._max_batch_size])


def _embedding_dim() -> Optional[int]:
    raw = _first_env("GRAPHITI_EMBEDDING_DIM", "OPENAI_EMBEDDING_DIM", "EMBEDDING_DIM")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _embedding_batch_size() -> int:
    raw = _first_env("GRAPHITI_EMBEDDING_BATCH_SIZE", "OPENAI_EMBEDDING_BATCH_SIZE", "EMBEDDING_BATCH_SIZE")
    if not raw:
        return 10
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _kgmark_strength() -> float:
    raw = _first_env("KGMARK_GRAPHITI_STRENGTH", "KGMARK_STRENGTH")
    if not raw:
        return 0.08
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.08


def _kgmark_mask_ratio() -> float:
    raw = _first_env("KGMARK_GRAPHITI_MASK_RATIO", "KGMARK_MASK_RATIO")
    if not raw:
        return 0.08
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return 0.08


def _kgmark_embed_latent(
    latent: Sequence[float],
    *,
    secret_key: str,
    context: str,
    payload_bit: str,
) -> tuple[List[float], List[int], List[float]]:
    values = [float(x) for x in latent]
    if not values:
        return [], [], []
    dim = len(values)
    mask_size = max(1, min(dim, int(round(dim * _kgmark_mask_ratio()))))
    mask_indices = _kgmark_mask_indices(dim, mask_size, secret_key, context)
    signature_values = [
        _kgmark_signature_value(secret_key, context, idx, payload_bit)
        for idx in mask_indices
    ]
    strength = _kgmark_strength()
    watermarked = list(values)
    for idx, sig in zip(mask_indices, signature_values):
        watermarked[idx] = (1.0 - strength) * watermarked[idx] + strength * sig
    return watermarked, mask_indices, signature_values


def _kgmark_mask_indices(dim: int, mask_size: int, secret_key: str, context: str) -> List[int]:
    scored = []
    for idx in range(dim):
        digest = hmac.new(
            secret_key.encode("utf-8"),
            f"kgmark-mask|{context}|{idx}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        scored.append((digest, idx))
    scored.sort()
    return sorted(idx for _, idx in scored[:mask_size])


def _kgmark_signature_value(secret_key: str, context: str, idx: int, payload_bit: str) -> float:
    digest = hmac.new(
        secret_key.encode("utf-8"),
        f"kgmark-signature|{payload_bit}|{context}|{idx}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    unsigned = int.from_bytes(digest[:8], "big") / float(2 ** 64 - 1)
    return 2.0 * unsigned - 1.0


def _hash_float_vector(values: Sequence[float]) -> str:
    rounded = [round(float(x), 6) for x in values]
    return sha256_text(stable_json(rounded))


def _deterministic_embedding(text: str, dim: int) -> List[float]:
    values: List[float] = []
    for idx in range(dim):
        digest = hashlib.sha256(f"graphiti-fallback-embed|{idx}|{text}".encode("utf-8")).digest()
        unsigned = int.from_bytes(digest[:8], "big") / float(2 ** 64 - 1)
        values.append(2.0 * unsigned - 1.0)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    n = min(len(left), len(right))
    if n <= 0:
        return 0.0
    dot = sum(float(left[i]) * float(right[i]) for i in range(n))
    left_norm = math.sqrt(sum(float(left[i]) ** 2 for i in range(n)))
    right_norm = math.sqrt(sum(float(right[i]) ** 2 for i in range(n)))
    denom = left_norm * right_norm
    if denom <= 0:
        return 0.0
    return dot / denom


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _build_openai_reranker():
    return OpenAIRerankerClient(
        config=LLMConfig(
            api_key=_first_env(
                "GRAPHITI_RERANKER_API_KEY",
                "OPENAI_RERANKER_API_KEY",
                "OPENAI_API_KEY",
                "MEMMARK_API_KEY",
            ),
            base_url=_first_env(
                "GRAPHITI_RERANKER_BASE_URL",
                "OPENAI_RERANKER_BASE_URL",
                "OPENAI_BASE_URL",
                "MEMMARK_BASE_URL",
            ),
            model=_first_env(
                "GRAPHITI_RERANKER_MODEL",
                "OPENAI_RERANKER_MODEL",
                "OPENAI_MODEL",
                "MEMMARK_MODEL",
            ),
        )
    )
