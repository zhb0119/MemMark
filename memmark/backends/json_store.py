from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from memmark.backends.base import MemoryBackendAdapter


class JsonMemoryStore(MemoryBackendAdapter):
    """File-backed JSON store for development / smoke tests.

    Supports the full operation vocabulary the carriers may emit:
      - add_memory
      - update_memory
      - delete_memory
    """

    # Json store is a thin stub — turn-level ingestion is the simplest
    # default (matches Graphiti's path so smoke results approximate
    # Graphiti's protocol).
    preferred_ingestion_mode = "turn"

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = Path(path) if path else None
        self._memories: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            self._memories = json.loads(self.path.read_text(encoding="utf-8"))

    def snapshot(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._memories]

    def apply(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        op = operation.get("op")
        evidence = list(operation.get("dia_ids", []))
        session_index = operation.get("session_index")
        speaker = operation.get("speaker", "")
        if op == "add_memory":
            memory = {
                "id": f"m{len(self._memories) + 1}",
                "text": operation["text"],
                "links": list(operation.get("links", [])),
                "dia_ids": evidence,
                "session_index": session_index,
                "speaker": speaker,
                "session_date_time": operation.get("session_date_time", ""),
            }
            self._memories.append(memory)
            self._persist()
            return memory
        if op == "update_memory":
            target_id = operation["memory_id"]
            new_text = operation["text"]
            updated: Dict[str, Any] = {}
            for record in self._memories:
                if record["id"] == target_id:
                    record["text"] = new_text
                    if "links" in operation:
                        record["links"] = list(operation["links"])
                    if evidence:
                        # Accumulate evidence across updates
                        record["dia_ids"] = list(
                            dict.fromkeys(list(record.get("dia_ids", [])) + evidence)
                        )
                    if session_index is not None:
                        record["session_index"] = session_index
                    if speaker:
                        record["speaker"] = speaker
                    updated = dict(record)
                    break
            self._persist()
            return updated or {"id": target_id, "text": new_text, "missing": True}
        if op == "delete_memory":
            target_id = operation["memory_id"]
            before = len(self._memories)
            self._memories = [m for m in self._memories if m["id"] != target_id]
            self._persist()
            return {"id": target_id, "deleted": before != len(self._memories)}
        raise ValueError(f"Unsupported operation: {op}")

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
