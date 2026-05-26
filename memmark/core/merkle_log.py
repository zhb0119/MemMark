"""Per-session Merkle log + signed root (README §9.2).

Each evolve decision contributes one leaf = `commit_t`. Periodic seal()
computes a Merkle root and signs it with HMAC(K, root) (placeholder for
ed25519). The signed `header_T` is the *anchor* that goes into the
in-storage memory snapshot so R3 In-Record Attribution Verification
(README §10.5) can verify any subset of leaves without an external
audit store.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from memmark.core.context import sha256_text
from memmark.core.types import MerkleProof, SessionHeader


def _hash_pair(left: str, right: str) -> str:
    return sha256_text(left + right)


def merkle_root(leaves: List[str]) -> str:
    if not leaves:
        return ""
    layer = list(leaves)
    while len(layer) > 1:
        nxt: List[str] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            nxt.append(_hash_pair(left, right))
        layer = nxt
    return layer[0]


def merkle_proof(leaves: List[str], target_index: int) -> MerkleProof:
    """Build inclusion proof for `leaves[target_index]`."""

    if not leaves or target_index < 0 or target_index >= len(leaves):
        raise IndexError("target_index out of range")
    layer = list(leaves)
    idx = target_index
    siblings: List[Tuple[str, str]] = []
    leaf = layer[idx]
    while len(layer) > 1:
        nxt: List[str] = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else left
            nxt.append(_hash_pair(left, right))
        sibling_idx = idx + 1 if idx % 2 == 0 else idx - 1
        if sibling_idx >= len(layer):
            sibling_idx = idx  # duplicated leaf path
            side = "R"  # we are on the left, sibling is the duplicate (right)
            sibling = layer[idx]
        else:
            side = "R" if idx % 2 == 0 else "L"
            sibling = layer[sibling_idx]
        siblings.append((sibling, side))
        idx //= 2
        layer = nxt
    return MerkleProof(leaf=leaf, siblings=siblings, root=layer[0])


def sign_root(secret_key: str, root: str) -> str:
    return hmac.new(
        secret_key.encode("utf-8"), root.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def verify_signature(secret_key: str, root: str, signature: str) -> bool:
    return hmac.compare_digest(sign_root(secret_key, root), signature)


@dataclass
class SessionMerkleLog:
    agent_id: str
    user_id: str
    session_id: str
    secret_key: str
    watermark_version: str = ""
    leaves: List[str] = field(default_factory=list)
    leaf_index: Dict[str, int] = field(default_factory=dict)

    def append(self, commitment: str) -> int:
        idx = len(self.leaves)
        self.leaves.append(commitment)
        self.leaf_index[commitment] = idx
        return idx

    def proof_for(self, commitment: str) -> Optional[MerkleProof]:
        if commitment not in self.leaf_index:
            return None
        return merkle_proof(self.leaves, self.leaf_index[commitment])

    def root(self) -> str:
        return merkle_root(self.leaves)

    def seal(self) -> SessionHeader:
        root = self.root()
        return SessionHeader(
            agent_id=self.agent_id,
            user_id=self.user_id,
            session_id=self.session_id,
            leaf_count=len(self.leaves),
            root=root,
            signature=sign_root(self.secret_key, root) if root else "",
            watermark_version=self.watermark_version,
        )
