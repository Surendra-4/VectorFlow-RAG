# src/indexing/compatibility.py

"""
Safe index compatibility system (Phase 12g).

The platform's hardest safety rule: **a configuration change must never
silently invalidate or mutate an index.** Changing the embedding model, the
vector dimension, the chunking strategy, the backend, or the FAISS topology
makes an existing index unusable as-is — and the user must be told, not
surprised.

This module compares an existing :class:`IndexProfile` against a *target*
configuration and returns a structured :class:`CompatibilityReport` whose whole
job is to drive the required UX:

    "Current index is incompatible with the selected configuration.
     Create a new index?"

Severity model
--------------
* ``BLOCKING``  — vectors are in a different space / shape. The index cannot be
  reused or even rebuilt-in-place meaningfully for this config without
  re-embedding the corpus. → action **create_new**.
* ``REBUILD``   — the embeddings are compatible, but the index must be rebuilt
  (different chunking, topology, or a stale corpus). → action **rebuild**.
* ``INFO``      — advisory only (e.g. training thresholds); reuse still allowed.

The validator NEVER performs any mutation. It only reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.indexing.profile import IndexProfile


class Severity(str, Enum):
    BLOCKING = "blocking"
    REBUILD = "rebuild"
    INFO = "info"


class Action(str, Enum):
    REUSE = "reuse"            # fully compatible — use as-is
    REBUILD = "rebuild"        # re-ingest the corpus into a (re)built index
    CREATE_NEW = "create_new"  # make a separate index; this one stays valid


@dataclass(frozen=True)
class IndexTargetConfig:
    """The configuration an index is being checked against.

    ``vector_dimension`` may be 0/unknown at check time (the dimension is a
    property of the embedding model, only known once it loads). When unknown we
    rely on the stricter ``embedding_model`` identity check.
    """

    embedding_model: str
    backend: str
    index_type: str
    chunk_size: int
    chunk_overlap: int
    vector_dimension: int = 0
    normalize: bool = True
    corpus_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityIssue:
    field: str
    severity: Severity
    message: str
    index_value: Any = None
    target_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class CompatibilityReport:
    """Structured verdict. ``compatible`` ⇒ the index is usable as-is."""

    index_name: str
    compatible: bool
    action: Action
    message: str
    issues: List[CompatibilityIssue] = field(default_factory=list)

    @property
    def blocking_issues(self) -> List[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == Severity.BLOCKING]

    @property
    def rebuild_issues(self) -> List[CompatibilityIssue]:
        return [i for i in self.issues if i.severity == Severity.REBUILD]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_name": self.index_name,
            "compatible": self.compatible,
            "action": self.action.value,
            "message": self.message,
            "issues": [i.to_dict() for i in self.issues],
        }


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #


def check_compatibility(profile: IndexProfile, target: IndexTargetConfig) -> CompatibilityReport:
    """Compare ``profile`` against ``target`` and return a report.

    Pure and side-effect-free. The decision tree:

    1. Any BLOCKING issue (embedding model/dim, backend) → ``create_new``.
    2. Else any REBUILD issue (chunking, topology, stale corpus) → ``rebuild``.
    3. Else → ``reuse``.
    """
    issues: List[CompatibilityIssue] = []

    # --- BLOCKING: different vector space ---------------------------------- #

    if profile.embedding_model != target.embedding_model:
        issues.append(CompatibilityIssue(
            field="embedding_model", severity=Severity.BLOCKING,
            message=(
                f"Index was built with embedding model '{profile.embedding_model}', "
                f"but the selected configuration uses '{target.embedding_model}'. "
                "Different models produce incompatible vector spaces."
            ),
            index_value=profile.embedding_model, target_value=target.embedding_model,
        ))

    # Dimension check only when the target dimension is known (>0).
    if target.vector_dimension and profile.vector_dimension and \
            profile.vector_dimension != target.vector_dimension:
        issues.append(CompatibilityIssue(
            field="vector_dimension", severity=Severity.BLOCKING,
            message=(
                f"Index dimension is {profile.vector_dimension}; the selected "
                f"embedding produces {target.vector_dimension}-d vectors. These "
                "cannot be searched together."
            ),
            index_value=profile.vector_dimension, target_value=target.vector_dimension,
        ))

    if profile.backend != target.backend:
        issues.append(CompatibilityIssue(
            field="backend", severity=Severity.BLOCKING,
            message=(
                f"Index uses the '{profile.backend}' backend; the selected "
                f"configuration uses '{target.backend}'. Stored vectors are not "
                "portable across backends."
            ),
            index_value=profile.backend, target_value=target.backend,
        ))

    # --- REBUILD: same vectors, different structure/content ---------------- #

    if profile.chunk_size != target.chunk_size or profile.chunk_overlap != target.chunk_overlap:
        issues.append(CompatibilityIssue(
            field="chunking", severity=Severity.REBUILD,
            message=(
                f"Chunking changed (size {profile.chunk_size}->{target.chunk_size}, "
                f"overlap {profile.chunk_overlap}->{target.chunk_overlap}). The "
                "corpus must be re-chunked and re-indexed."
            ),
            index_value={"chunk_size": profile.chunk_size, "chunk_overlap": profile.chunk_overlap},
            target_value={"chunk_size": target.chunk_size, "chunk_overlap": target.chunk_overlap},
        ))

    if profile.index_type != target.index_type:
        issues.append(CompatibilityIssue(
            field="index_type", severity=Severity.REBUILD,
            message=(
                f"Index topology changed ('{profile.index_type}' -> "
                f"'{target.index_type}'). The index must be rebuilt (the embeddings "
                "themselves remain valid)."
            ),
            index_value=profile.index_type, target_value=target.index_type,
        ))

    if target.corpus_fingerprint is not None and profile.corpus_fingerprint is not None \
            and profile.corpus_fingerprint != target.corpus_fingerprint:
        issues.append(CompatibilityIssue(
            field="corpus_fingerprint", severity=Severity.REBUILD,
            message=(
                "The corpus has changed since this index was built. Rebuild to "
                "include the current documents."
            ),
            index_value=profile.corpus_fingerprint, target_value=target.corpus_fingerprint,
        ))

    # --- Verdict ----------------------------------------------------------- #

    has_blocking = any(i.severity == Severity.BLOCKING for i in issues)
    has_rebuild = any(i.severity == Severity.REBUILD for i in issues)

    if has_blocking:
        action = Action.CREATE_NEW
        message = (
            f"Index '{profile.name}' is incompatible with the selected configuration. "
            "Create a new index?"
        )
        compatible = False
    elif has_rebuild:
        action = Action.REBUILD
        message = (
            f"Index '{profile.name}' needs to be rebuilt to match the selected "
            "configuration. Rebuild now?"
        )
        compatible = False
    else:
        action = Action.REUSE
        message = f"Index '{profile.name}' is compatible with the selected configuration."
        compatible = True

    return CompatibilityReport(
        index_name=profile.name, compatible=compatible, action=action,
        message=message, issues=issues,
    )


# --------------------------------------------------------------------------- #
# Bridge from runtime config → target
# --------------------------------------------------------------------------- #


def target_from_index_settings(
    index_settings,
    *,
    vector_dimension: int = 0,
    corpus_fingerprint: Optional[str] = None,
) -> IndexTargetConfig:
    """Build an :class:`IndexTargetConfig` from a runtime
    ``IndexConstructionSettings`` (Phase 12c) + the embedding choice.

    ``vector_dimension`` is supplied by the caller when known (e.g. the running
    embedder's dimension); otherwise 0 and the model-name check carries the
    compatibility decision.
    """
    return IndexTargetConfig(
        embedding_model=index_settings.embedding.model,
        backend=index_settings.vector_backend,
        index_type=index_settings.faiss_index_type,
        chunk_size=index_settings.chunk_size,
        chunk_overlap=index_settings.chunk_overlap,
        vector_dimension=vector_dimension,
        normalize=index_settings.embedding.normalize,
        corpus_fingerprint=corpus_fingerprint,
    )
